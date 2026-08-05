# Body2Health: Dual-View Silhouette Anthropometry

Silhouette-based system for recovering body dimensions from front and side phone images, then deriving central-adiposity indices from those dimensions. The current research target is not direct body-fat measurement; BodyM-style data provides silhouettes, height/weight metadata, and tape-measured dimensions, but not clinical body-fat labels.

## Research Position
- Learn dual-view body-shape embeddings from standardized silhouettes.
- Predict measurable anthropometric dimensions such as waist and hip girth.
- Derive WHR, WHtR, BRI, and related central-adiposity indices from predicted dimensions and known height.
- Use NLF feed-forward SMPL-X recovery plus render-back comparison as a reliability/abstention gate.
- Treat body-fat percentage as a future anchor-label task only if explicit labels are collected, for example clinical measurements or trained annotator distributions.

The key claim is: **2D phone images can support dimension recovery and clinically interpretable shape indices only when the capture passes a geometry reliability gate.** The system should not claim true body-fat percentage without a real body-fat label source.

## Data capture assumptions (BodyM-like)
- Camera height fixed; subject stands on marked floor spots; distance ≈ 2.5–3.0 m, consistent across subjects.
- Height-aware silhouette resizing keeps the 640x480 model input comparable across captures; longer distances reduce perspective distortion.
- Minimal or tight clothing is assumed for the supervised BodyM baseline. Loose-clothing capture should be evaluated as a separate domain shift, not mixed into the clean baseline claim.

## Pipeline (high level)
1) Input RGB front/side  
2) Segment person through strict YOLO + padded box-only SAM2 in the canonical `pipeline/` package  
3) Standardize silhouette: crop, resize, and center on the 640x480 model canvas  
4) Run NLF on the front RGB image to recover a personalized SMPL-X mesh  
5) Render the SMPL-X mesh back to the front silhouette and compute reliability mismatch  
6) Reject the report if the render-back agreement is too weak  
7) Dual-view model (`src/model/contrastive_dualview.py`): front/side branches → fused embedding  
8) Regression heads predict body dimensions, not body-fat percentage  
9) Derived indices are computed arithmetically from predicted dimensions and height  
10) Evaluation reports dimension error, index error, silhouette consistency, and abstention/reliability behavior

## What Is Novel Here
- **Dimension-first framing:** the supervised target is what the dataset actually labels: body dimensions. BMI and body-fat percentage are not treated as the main learned outputs.
- **Central-adiposity index recovery:** WHR, WHtR, and BRI are computed from predicted waist/hip dimensions, giving clinically interpretable outputs without inventing body-fat labels.
- **SMPL-X-backed reliability:** a predicted measurement is accepted only when a personalized SMPL-X render agrees with the observed silhouette. Failed agreement becomes an abstention signal instead of a hidden bad prediction.
- **Clean baseline for clothing/domain shift:** tight-clothing BodyM results define the recoverable anthropometry baseline; phone and loose-clothing inputs can be measured against that baseline.
- **Optional anchor-label extension:** Patent-2-style probability distributions are possible only after collecting anchor body-fat labels. In this repo, that should be a separate experiment, not the default claim.

## Target Outputs
- Predicted dimensions: waist, hip, chest, shoulder, inseam, arm, and other BodyM-labeled measurements available in the CSV.
- Derived indices: WHR (`waist / hip`), WHtR (`waist / height`), and BRI from waist and height.
- Central-adiposity risk summary: WHtR-primary screening interpretation for estimated cardiometabolic risk, WHR and waist-circumference support, BRI as exploratory context only.
- Reliability outputs: SMPL-X render IoU, contour mismatch, foreground-quality checks, accept/reject status, and fitted `.obj` export. The legacy proxy gate is kept only for BodyM-style mask-only coverage experiments.
- Optional future output: body-fat probability distribution trained from explicit anchor-label distributions.

## Repository map
```
bodyfit/
|-- 3-train/                     # training entrypoints
|-- 4-infer/                     # inference entrypoints
|-- 5-eval/                      # evaluation scripts
|-- dashboard/                   # local Streamlit visual demo
|-- configs/                     # YAML/settings only
|   `-- segmentation/            # segmentation model configs, including SAM2 YAML
|-- models/                      # binary model checkpoints only
|   `-- segmentation/            # segmentation weights, including YOLO and SAM2 checkpoint
|-- src/
|   |-- model/                   # dual-view CNN + feature extractors
|   |-- train/                   # training loops, contrastive losses, data loading
|   |-- eval/                    # metrics (IoU now; extend with MAE/MSE/R²/MAPE/Pearson)
|   |-- smpl/                    # legacy lightweight geometry reliability gate
|   |-- smplx_fit/               # NLF feed-forward SMPL-X + render-back reliability
|   `-- utils/                   # I/O helpers
|-- models/nlf/                  # NLF TorchScript model
|-- models/smplx/                # SMPLX .npz model assets
|-- data/
|   `-- bodym/                   # cleaned readable BodyM dataset and dimension labels
|-- outputs/                     # generated figures and analysis artifacts
`-- pipeline/                    # iPhone RGB -> silhouette preprocessing
```

## Quick starts
- Train a dimension model on BodyM-style CSV labels:
  `PYTHONPATH=. python 3-train/1train.py --csv "data/bodym/pairs_dimensions.csv" --measurement_cols "waist_cm,hip_cm,chest_cm" --encoder resnet18 --batch_size 12 --epochs 40 --lr 3e-4 --lambda_reg 2.0 --augment --ckpt_tag _v4_resnet`
- Run metadata ablations for leakage checks:
  `PYTHONPATH=. python 5-eval/4ablate.py --csv "data/bodym/pairs_dimensions.csv" --ckpt checkpoints/best_640x480_v4_resnet.pt --measurement_cols "waist_cm,hip_cm,chest_cm" --tp_on waist_cm`
- Run the final RGB-to-report demo with NLF SMPL-X reliability:
  `PYTHONPATH=. python 4-infer/1infer.py --front_rgb TestPhoto/deva_front.png --side_rgb TestPhoto/deva_side.png --ckpt checkpoints/best_640x480_v4_resnet.pt --height_cm 175 --sex male --smplx_fit --save_silhouettes outputs/final/deva --save_smplx outputs/final/deva/smplx --json outputs/final/deva/result.json`
- Run the local visual dashboard:
  `.venv/bin/streamlit run dashboard/app.py`
  The dashboard supports both preferred front+side mode and a quick single-front-photo mode. Single-front mode duplicates the front silhouette for the model side branch, so it is useful for demos and SMPL-X visualization but should be labeled less reliable than true dual-view capture.
- Run cached-mask inference with NLF SMPL-X reliability:
  `PYTHONPATH=. python 4-infer/1infer.py --front out/deva_front_silhouette.png --side out/deva_side_silhouette.png --front_rgb TestPhoto/deva_front.png --ckpt checkpoints/best_640x480_v4_resnet.pt --height_cm 175 --sex male --smplx_fit --save_smplx outputs/final/deva_cached/smplx --json outputs/final/deva_cached/result.json --skip-envelope-check`
- Run the official TestPhoto segmentation smoke:
  `PYTHONPATH=. python 5-eval/7segmentation_smoke.py --root TestPhoto --debug_dir outputs/final/segmentation --json outputs/final/segmentation/report.json`
- Run the reliability coverage table:
  `PYTHONPATH=. python 5-eval/6gate_eval.py --csv data/bodym/pairs_dimensions.csv --ckpt checkpoints/best_640x480_v4_resnet.pt --measurement_cols waist_cm,hip_cm,chest_cm --max_rows 50 --device cpu`
- Primary cleaned dataset: `data/bodym/pairs_dimensions.csv`

## Experimental defaults (target)
- Optimizer Adam, lr=1e-4, batch=32, embed dim 128.
- Augmentations: small rotations (±3°), scale (±5%), erode/dilate jitter.
- Splits: 70/15/15 subjects.
- Loss: InfoNCE (positives = same subject/view augmentation; negatives = other subjects) plus MAE/Smooth L1 for dimension regression.
- Validation split should be subject-disjoint. Row-level splits can leak the same subject across train and validation.

## Evaluation & baselines
- Dimension metrics: MAE, RMSE, R², Pearson, and threshold accuracy for each predicted girth.
- Index metrics: WHR/WHtR/BRI error computed from predicted dimensions versus the same index computed from tape-measured dimensions.
- Geometry metrics: SMPL-X front render IoU, contour Chamfer distance, foreground-ratio checks, and retained-error-vs-coverage.
- Leakage checks: report silhouette-only, dual-view, dual-view + height, dual-view + weight, and dual-view + height + weight separately.
- Baselines: bounding-box/area features, plain CNN regression, contrastive dual-view model, legacy proxy-gated reporting for BodyM masks, and NLF SMPL-X-gated phone reporting.

## Status / TODO
- Done: canonical `data/bodym/pairs_dimensions.csv`; waist/hip/chest training; subject-disjoint split; derived-index evaluation; structured central-adiposity risk summary; strict YOLO + padded box-only SAM2 RGB path; NLF feed-forward SMPL-X render-back gate; local Streamlit dashboard.
- TODO: calibrate the SMPL-X IoU threshold and add optional tape-measure extraction from mesh vertex rings as a diagnostic, not a replacement for the trained dimension outputs.
