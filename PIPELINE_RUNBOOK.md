# BodyFit Pipeline Runbook

## What To Capture
- For dual-view work, capture 2 full-body images per subject: 1 front view and 1 side view.
- Keep camera height fixed and keep subject distance consistent, roughly 2.5-3.0 m.
- Use tight-fitting clothing. Avoid coats, dresses, abayas, or anything that changes the body outline.
- Keep the full body in frame from head to feet.
- Use a neutral standing pose and keep the background simple enough that a person mask is recoverable.
- Training scripts in this repo expect binary silhouette masks, not raw RGB photos.

## What Already Exists Vs What You Must Provide
- Already in repo: model code in `src/`, the iPhone preprocessing app in `2-pipeline/pipeline/`, training/inference/eval entrypoints, and existing artifact folders such as `checkpoints/`, `out/`, and `outputs/`.
- You must provide, for training: front and side silhouette masks plus metadata CSVs under `data /raw/metadata/`.
- You must provide, for silhouette-only inference: 640x480 or resizeable front and side mask images.
- You must provide, for raw iPhone evaluation: a trained checkpoint plus a single RGB image path. The current raw-image CLI processes one image at a time.

## Default Order For Training
1. Put front masks in `data /raw/mask/` and side masks in `data /raw/mask_left/`, with matching metadata CSVs in `data /raw/metadata/`.
2. Build the paired dataset CSV:

```bash
python 1-data/1build.py
```

Produces `data /pairs_full.csv`.

3. Resize masks to 640x480 and optionally make a subset:

```bash
python 1-data/2prep.py --input_csv "data /pairs_full.csv" --output_dir "data /640x480_processed" --create_subset
```

Produces `data /640x480_processed/pairs_640x480.csv` and, when `--create_subset` is used, `data /640x480_processed/pairs_10percent.csv`.

4. Inspect dataset stats if needed:

```bash
python 1-data/3stats.py --labels "data /labels.csv" --pairs "data /pairs_full.csv"
```

5. Train the model:

```bash
python 3-train/train_contrastive_640x480.py --csv "data /640x480_processed/pairs_640x480.csv"
```

Writes checkpoints into `checkpoints/` and logs to stdout. If you redirect logs to `out/train_640x480.log`, the later metrics scripts can parse them directly.

6. Extract training summaries from the log:

```bash
python 1-data/extract_training_metrics.py --log-file out/train_640x480.log
```

Writes CSV/JSON summaries under `data/`.

7. Plot training curves:

```bash
python 5-eval/plot_training_curves.py --log out/train_640x480.log --out_dir out/plots
```

Writes PNG plots under `out/plots`.

## Inference Order
- If you already have silhouette masks and want the simplest prediction path:

```bash
python 4-infer/predict_from_silhouette.py --front "path/to/front_mask.png" --side "path/to/side_mask.png" --ckpt "checkpoints/latest.pt"
```

- If you only have one silhouette and want it duplicated across both branches:

```bash
python 4-infer/predict_from_silhouette.py --mask "path/to/mask.png" --ckpt "checkpoints/latest.pt"
```

- If you want checkpoint-stat de-normalization and direct BMI/BF output:

```bash
python 4-infer/predict_bf.py --front "path/to/front_mask.png" --side "path/to/side_mask.png" --ckpt_dir checkpoints
```

- If you want constrained anthropometric output with optional assisted mode:

```bash
python 4-infer/infer_anthro.py --front "path/to/front_mask.png" --side "path/to/side_mask.png" --ckpt "checkpoints/latest.pt"
```

Add `--height_cm` and `--weight_kg` to use assisted mode.

- If you want the stricter BodyM-envelope path:

```bash
python -c "import sys; sys.path.append('4-infer'); from corrected_inference import corrected_inference"
```

Use `corrected_inference.py` as a library-style helper rather than a CLI.

## Raw iPhone Image Path
- For a single raw iPhone image, run:

```bash
python 5-eval/evaluate_640x480_model.py --model_path "checkpoints/latest.pt" --image_path "path/to/image.jpg"
```

- This route runs the preprocessing app and then evaluates the checkpoint.
- The current CLI accepts one raw image at a time. It is useful for checking the preprocessing pipeline on a single capture, not for building a full dual-view training dataset.
- If you already have a processed silhouette instead of RGB, use:

```bash
python 5-eval/evaluate_640x480_model.py --model_path "checkpoints/latest.pt" --silhouette_path "path/to/mask.png"
```

## Evaluation Scripts
- `5-eval/ablation_table.py`: compare single-view vs multi-view and simple post-hoc corrections on one checkpoint.
- `5-eval/contrastive_vs_no_contrastive_table.py`: compare two checkpoints, usually a contrastive run against a regression-only run.
- `5-eval/compute_silhouette_iou.py`: compute preprocessing IoU from a CSV of predicted and ground-truth mask paths.
- `5-eval/generate_figures_single_subject.py`: generate paper-style figures when you already have training and IoU logs.

## Notes
- `3-train/refine_silhouettes_with_smpl.py` is optional and not part of the default path. In this checkout it still depends on `src.smpl` and `src.render`, which are not present.
- `1-data/2prep.py` imports `pandas` at module import time. If `pandas` is missing, even `--help` will fail until that dependency is available.
- The preprocessing package is now imported as `pipeline`. From the repo root, this works:

```bash
python -c "from pipeline.iphone_pipeline import process_iphone_image"
```
