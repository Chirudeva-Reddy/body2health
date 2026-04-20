# BodyFit: Dual-View Contrastive Anthropometry

Silhouette-based system to predict BMI and body-fat % from front + side views, aligned with WHO anthropometric standards. Pipeline: RGB → segmentation/smoothing/standardization → (optional) SMPL canonicalization → dual-branch contrastive encoder → regression heads for BMI/BF%.

## Objectives
- Learn body-shape embeddings robust to clothing/lighting via supervised contrastive loss on dual views.
- Map embeddings to BMI/BF% with a lightweight regressor.
- Produce WHO-aligned health interpretations (BMI/BF% categories, optional waist/WHR cut-offs).

## Data capture assumptions (BodyM-like)
- Camera height fixed; subject stands on marked floor spots; distance ≈ 2.5–3.0 m, consistent across subjects.
- Height normalization to 256 px removes scale differences; longer distances reduce perspective distortion.

## Pipeline (high level)
1) Input RGB front/side  
2) Segment person (`2-pipeline/pipeline/segmentation.py`, compatibility shim)  
3) Guided filter refinement + pose carving to separate arms; multi-resolution smoothing + hole fill  
4) Standardize silhouette (`standardize_silhouette`): crop, resize to 256×128, center  
5) Optional SMPL fit (`src/smpl/fitter.py`) → canonical renders (`src/render/canonical.py`)  
6) Dual-view model (`src/model/dualview.py`): front/side branches → fused embedding  
7) Losses: `L = L_contrastive (InfoNCE/NT-Xent) + λ * L_reg (BMI/BF%)`  
8) Evaluation: MAE/MSE/R²/Pearson; WHO category alignment; baselines for comparison

## Repository map
```
bodyfit/
|-- 1-data/                      # dataset building, resizing, stats, training-log extraction
|-- 2-pipeline/                  # iPhone RGB -> silhouette preprocessing app
|-- 3-train/                     # training entrypoints
|-- 4-infer/                     # inference entrypoints
|-- 5-eval/                      # evaluation and plotting scripts
|-- configs/                     # defaults (silhouette size, SMPL fit, render, training stages)
|-- src/
|   |-- model/                   # dual-view CNN + baselines (baseline.py)
|   |-- train/                   # training loops, contrastive losses, data loading
|   |-- eval/                    # metrics (IoU now; extend with MAE/MSE/R²/MAPE/Pearson)
|   `-- utils/                   # I/O helpers
|-- data /                       # CSVs (labels/pairs_full/...), raw masks under data /raw/mask[_left]/
|-- out/                         # checkpoints, renders
`-- docs/                        # pipeline, literature, experiments, eval, dataset stats
```

## Quick starts
- Canonicalize single image: `python scripts/fit_and_render.py <image> --height_cm <cm> --engine mediapipe`
- Two-view fit + IoU gate: `python scripts/fit_and_render_two_view.py --front_img <front> --side_img <side> --height_cm <cm>`
- Toy training (fake data): `python scripts/train_minimal.py`
- Contrastive+regression training on CSV (BMI+BF%, with target normalization and stronger regression weight):  
  `python scripts/train_contrastive_reg.py --csv "data /pairs.csv" --measurement_cols bmi --batch_size 32 --epochs 20 --lr 3e-4 --lambda_reg 1.0 --tau 0.1 --augment`
- Refine noisy silhouettes via SMPL re-render: `python 3-train/refine_silhouettes_with_smpl.py --front_mask <path> --side_mask <path> --height_cm <cm>`
- Dataset stats: `python 1-data/dataset_stats.py --labels "data /labels.csv" --pairs "data /pairs.csv"`

## Experimental defaults (target)
- Optimizer Adam, lr=1e-4, batch=32, embed dim 128.
- Augmentations: small rotations (±3°), scale (±5%), erode/dilate jitter.
- Splits: 70/15/15 subjects.
- Loss: InfoNCE (positives = same subject/view aug; negatives = others) + MAE/MSE for BMI/BF%; tune λ.

## Evaluation & baselines
- Metrics: MAE, MSE/RMSE, R², MAPE (guarded), Pearson; IoU for masks.
- WHO mapping: BMI/BF% categories, optional waist/WHR cut-offs in report.
- Baseline: `src/model/baseline.py` (width/height/area/aspect) vs plain CNN regression vs contrastive model; store plots/tables in `results/baselines/`.

## Status / TODO
- Done: segmentation/smoothing/standardization; SMPL fit/render; docs scaffold; baseline stub; dataset stats script.
- TODO: real dataloader for `labels.csv`/`pairs.csv`; contrastive loss + metrics; evaluation script; generate qualitative/prelim results into `results/prelim/`.
