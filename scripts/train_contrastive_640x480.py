"""
Revised contrastive and regression training script for 640×480 silhouettes.

This minimal training loop handles argument parsing, dataset loading, model
construction, training, validation, and checkpointing without unnecessary
complexity.  Configuration is exposed via command-line flags, but only
parameters required by the underlying model are kept.  Comments throughout
the script outline the main stages of the workflow.
"""

import argparse
import pathlib
import sys
from typing import List

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

# Append the project root so imports resolve when running this script directly
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.model.contrastive_dualview import DualViewContrastive
from src.train.data import SilhouetteDataset
from src.train.losses import total_loss


def _resolve_csv_path(p: str) -> str:
    cands = [
        p,
        "data/pairs.csv",
        "data /pairs.csv",
        "data/pairs_full.csv",
        "data /pairs_full.csv",
    ]
    for c in cands:
        if pathlib.Path(c).exists():
            return c
    raise FileNotFoundError(f"CSV not found. Tried: {cands}")


def parse_args() -> argparse.Namespace:
    """Define and parse command‑line arguments.

    Only essential options are exposed here.  Defaults favour moderate
    compute budgets at 640×480 resolution.  The `--convit_shared` flag is
    implicitly true unless `--no_convit_shared` is supplied, avoiding
    redundant toggles.
    """
    parser = argparse.ArgumentParser(
        description="Contrastive + regression training at 640×480 resolution",
    )
    # Dataset & resolution
    parser.add_argument("--csv", type=str, default="data/pairs_10_percent.csv")
    parser.add_argument("--measurement_cols", type=str, default="bmi")
    parser.add_argument("--height", type=int, default=640)
    parser.add_argument("--width", type=int, default=480)
    # Training hyperparameters
    parser.add_argument("--batch_size", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=18)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--lambda_reg", type=float, default=2.0)
    parser.add_argument("--tau", type=float, default=0.07)
    parser.add_argument("--no_contrastive", action="store_true")
    # Model configuration
    parser.add_argument("--use_large", action="store_true")
    parser.add_argument("--base_dim", type=int, default=80)
    parser.add_argument("--use_bbox_features", action="store_true")
    parser.add_argument(
        "--encoder",
        "--enc",
        dest="encoder",
        type=str,
        choices=["cnn", "resnet18", "resnet34", "convnextv2_tiny"],
        default="cnn",
    )
    parser.add_argument("--pt", action="store_true")
    # ConViT settings (ignored when encoder='cnn')
    parser.add_argument("--convit_patch_size", type=int, default=16)
    parser.add_argument("--convit_dim", type=int, default=256)
    parser.add_argument("--convit_depth", type=int, default=6)
    parser.add_argument("--convit_heads", type=int, default=4)
    parser.add_argument("--convit_mlp_dim", type=int, default=512)
    parser.add_argument("--convit_drop", type=float, default=0.0)
    parser.add_argument("--convit_pool", type=str, choices=["mean", "cls"], default="mean")
    parser.add_argument("--convit_gpsa_layers", type=int, default=2)
    # ConViT weights are shared across views unless explicitly disabled
    parser.add_argument("--no_convit_shared", action="store_true")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    # Data loading & reproducibility
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    # Runtime limits & checkpointing
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--ckpt_tag", type=str, default="")
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()
    # Derive convit_shared from --no_convit_shared
    args.convit_shared = not args.no_convit_shared
    return args


def build_dataloaders(
    csv_path: str,
    target_hw: tuple[int, int],
    measurement_cols: List[str],
    augment: bool,
    seed: int,
    batch_size: int,
    num_workers: int,
) -> tuple[DataLoader, DataLoader, SilhouetteDataset]:
    """Construct training and validation DataLoaders from the full dataset.

    A 20% validation split is used by default.  A seeded generator ensures
    reproducibility of the random split across runs.
    """
    full_ds = SilhouetteDataset(
        csv_path,
        target_hw=target_hw,
        measurement_cols=measurement_cols,
        augment=augment,
    )
    # Determine split sizes
    n_total = len(full_ds)
    n_val = max(1, int(0.2 * n_total))
    n_train = n_total - n_val
    # Reproducible random split
    g = torch.Generator()
    g.manual_seed(seed)
    train_ds, val_ds = random_split(full_ds, [n_train, n_val], generator=g)
    # Create data loaders
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader, full_ds


def compute_normalization_stats(full_ds: SilhouetteDataset, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute mean and standard deviation for measurement and body‑fat targets.

    Stats are aggregated over the entire dataset and moved to the specified
    device.  A small epsilon avoids division by zero during standardization.
    """
    meas_list, bf_list = [], []
    for item in full_ds:
        meas_list.append(item["y_meas"])
        bf_list.append(item["y_bf"])
    meas_cat = torch.stack(meas_list)
    bf_cat = torch.stack(bf_list)
    meas_mean = meas_cat.mean(dim=0).to(device)
    meas_std = meas_cat.std(dim=0).clamp(min=1e-6).to(device)
    bf_mean = bf_cat.mean(dim=0).to(device)
    bf_std = bf_cat.std(dim=0).clamp(min=1e-6).to(device)
    return meas_mean, meas_std, bf_mean, bf_std


def train_epoch(
    model: DualViewContrastive,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    device: torch.device,
    step_count: int,
) -> tuple[float, int]:
    """Run one training epoch and return the average loss and updated step count."""
    model.train()
    losses = []
    for batch in loader:
        if args.max_steps is not None and step_count >= args.max_steps:
            break
        front = batch["front"].to(device)
        side = batch["side"].to(device)
        y_meas = batch["y_meas"].to(device)
        y_bf = batch["y_bf"].to(device)
        optimizer.zero_grad()
        out = model(front, side)
        loss_dict = total_loss(
            out,
            y_meas,
            y_bf,
            lambda_reg=args.lambda_reg,
            tau=args.tau,
            use_contrastive=not args.no_contrastive,
        )
        total_loss_val = loss_dict["total"]
        total_loss_val.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        losses.append(total_loss_val.item())
        step_count += 1
    avg_loss = float(np.mean(losses)) if losses else 0.0
    return avg_loss, step_count


def evaluate(
    model: DualViewContrastive,
    loader: DataLoader,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[float, float, float]:
    """Evaluate the model on the validation set and return loss and MAEs."""
    model.eval()
    val_losses, mae_meas, mae_bf = [], [], []
    with torch.no_grad():
        for batch in loader:
            front = batch["front"].to(device)
            side = batch["side"].to(device)
            y_meas = batch["y_meas"].to(device)
            y_bf = batch["y_bf"].to(device)
            out = model(front, side)
            loss_dict = total_loss(
                out,
                y_meas,
                y_bf,
                lambda_reg=args.lambda_reg,
                tau=args.tau,
                use_contrastive=not args.no_contrastive,
            )
            val_losses.append(loss_dict["total"].item())
            # Compute MAE in native units (measurement targets) and fraction (body‑fat)
            mae_meas.append(torch.nn.functional.l1_loss(out["meas"], y_meas).item())
            mae_bf.append(torch.nn.functional.l1_loss(out["bf"], y_bf).item())
    val_loss = float(np.mean(val_losses)) if val_losses else 0.0
    val_mae_meas = float(np.mean(mae_meas)) if mae_meas else 0.0
    val_mae_bf = float(np.mean(mae_bf)) if mae_bf else 0.0
    return val_loss, val_mae_meas, val_mae_bf


def save_checkpoint(
    path: pathlib.Path,
    model: DualViewContrastive,
    optimizer: torch.optim.Optimizer,
    meas_mean: torch.Tensor,
    meas_std: torch.Tensor,
    bf_mean: torch.Tensor,
    bf_std: torch.Tensor,
    args: argparse.Namespace,
    epoch: int | None = None,
    val_loss: float | None = None,
) -> None:
    """Serialize the model and training state to a checkpoint file."""
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "meas_mean": meas_mean,
            "meas_std": meas_std,
            "bf_mean": bf_mean,
            "bf_std": bf_std,
            # Store parsed arguments for reproducibility
            "args": vars(args),
            "use_large": args.use_large,
            "base_dim": args.base_dim,
            "use_bbox_features": args.use_bbox_features,
            "encoder": args.encoder,
            "pt": args.pt,
            "convit_patch_size": args.convit_patch_size,
            "convit_dim": args.convit_dim,
            "convit_depth": args.convit_depth,
            "convit_heads": args.convit_heads,
            "convit_mlp_dim": args.convit_mlp_dim,
            "convit_drop": args.convit_drop,
            "convit_pool": args.convit_pool,
            "convit_gpsa_layers": args.convit_gpsa_layers,
            "convit_shared": args.convit_shared,
            "input_size": (args.height, args.width),
            "epoch": epoch,
            "val_loss": val_loss,
        },
        path,
    )


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    args.csv = _resolve_csv_path(args.csv)
    # Seed everything for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    # Parse measurement columns from comma‑separated string
    meas_cols = [c.strip() for c in args.measurement_cols.split(",") if c.strip()]
    # Build data loaders
    train_loader, val_loader, full_ds = build_dataloaders(
        args.csv,
        target_hw=(args.height, args.width),
        measurement_cols=meas_cols,
        augment=args.augment,
        seed=args.seed,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    # Instantiate model
    model = DualViewContrastive(
        out_meas=len(meas_cols),
        proj_dim=128,
        use_large=args.use_large,
        base_dim=args.base_dim,
        use_bbox_features=args.use_bbox_features,
        encoder=args.encoder,
        pt=args.pt,
        convit_patch_size=args.convit_patch_size,
        convit_dim=args.convit_dim,
        convit_depth=args.convit_depth,
        convit_heads=args.convit_heads,
        convit_mlp_dim=args.convit_mlp_dim,
        convit_drop=args.convit_drop,
        convit_pool=args.convit_pool,
        convit_gpsa_layers=args.convit_gpsa_layers,
        convit_shared=args.convit_shared,
        convit_img_hw=(args.height, args.width),
    ).to(device)
    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    # Compute normalisation statistics once over the full dataset
    meas_mean, meas_std, bf_mean, bf_std = compute_normalization_stats(full_ds, device)
    # Prepare checkpoint directory
    ckpt_dir = pathlib.Path("checkpoints")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    tag = args.ckpt_tag or ""
    best_val = float("inf")
    step_count = 0
    start_epoch = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if ckpt.get("val_loss") is not None:
            best_val = float(ckpt["val_loss"])
        if ckpt.get("epoch") is not None:
            start_epoch = int(ckpt["epoch"]) + 1
        print(f"Resumed from {args.resume} at epoch {start_epoch} (best_val={best_val:.4f})")
    # Main training loop
    for epoch in range(start_epoch, args.epochs):
        train_loss, step_count = train_epoch(model, train_loader, optimizer, args, device, step_count)
        val_loss, val_mae_meas, val_mae_bf = evaluate(model, val_loader, args, device)
        scheduler.step()
        # Logging summary metrics for the epoch
        print(
            f"Epoch {epoch+1}/{args.epochs} | Train {train_loss:.4f} | "
            f"Val {val_loss:.4f} | Val MAE meas {val_mae_meas:.3f} | "
            f"Val MAE BF {val_mae_bf*100:.3f}%",
        )
        # Checkpoint the best model based on validation loss
        if val_loss < best_val:
            best_val = val_loss
            best_path = ckpt_dir / f"best_640x480{tag}.pt"
            save_checkpoint(
                best_path,
                model,
                optimizer,
                meas_mean,
                meas_std,
                bf_mean,
                bf_std,
                args,
                epoch=epoch,
                val_loss=val_loss,
            )
            print(f"New best model saved to {best_path}")
        # Stop early if max_steps reached
        if args.max_steps is not None and step_count >= args.max_steps:
            print(f"Reached max_steps={args.max_steps}, stopping training.")
            break
    # Always save final model at the end of training
    final_path = ckpt_dir / f"final_640x480{tag}.pt"
    save_checkpoint(
        final_path,
        model,
        optimizer,
        meas_mean,
        meas_std,
        bf_mean,
        bf_std,
        args,
    )
    print(f"Training complete. Final checkpoint: {final_path}")


if __name__ == "__main__":
    main()
