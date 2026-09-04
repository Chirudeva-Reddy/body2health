# Body2Fit Pipeline Runbook

## Canonical Dataset

Use `data/bodym/pairs_dimensions.csv` for training and evaluation. It contains renamed front/side mask paths, subject/capture keys, height/weight metadata, and tape-measured BodyM dimensions.

Traceability files:

- `data/bodym/manifest.csv`: renamed file to original photo ID mapping.
- `data/bodym/subject_key_map.csv`: `sub_XXXX` to original subject ID mapping.
- `data/bodym/metadata/measurements_renamed.csv`: one row per subject with normalized measurement column names.

## Training

Train the active dimension model:

```bash
PYTHONPATH=. python 3-train/1train.py \
  --csv "data/bodym/pairs_dimensions.csv" \
  --measurement_cols "waist_cm,hip_cm,chest_cm" \
  --encoder resnet18 \
  --batch_size 12 \
  --epochs 40 \
  --lr 3e-4 \
  --lambda_reg 2.0 \
  --augment \
  --ckpt_tag _v4_resnet
```

The trainer uses a subject-disjoint validation split and writes checkpoints to `checkpoints/`. Checkpoints store the target names in `measurement_cols`.

## Inference

Run the final RGB-to-report path:

```bash
PYTHONPATH=. python 4-infer/1infer.py \
  --front_rgb "TestPhoto/deva_front.png" \
  --side_rgb "TestPhoto/deva_side.png" \
  --ckpt "checkpoints/best_640x480_v4_resnet.pt" \
  --height_cm 175 \
  --sex male \
  --smplx_fit \
  --save_silhouettes outputs/final/deva \
  --save_smplx outputs/final/deva/smplx \
  --json outputs/final/deva/result.json
```

Run the dimension-first silhouette path when masks already exist:

```bash
PYTHONPATH=. python 4-infer/1infer.py \
  --front "path/to/front_mask.png" \
  --side "path/to/side_mask.png" \
  --ckpt "checkpoints/best_640x480_v4_resnet.pt" \
  --height_cm 175
```

Run cached silhouettes plus front RGB through the NLF SMPL-X reliability gate:

```bash
PYTHONPATH=. python 4-infer/1infer.py \
  --front "out/deva_front_silhouette.png" \
  --side "out/deva_side_silhouette.png" \
  --front_rgb "TestPhoto/deva_front.png" \
  --ckpt "checkpoints/best_640x480_v4_resnet.pt" \
  --height_cm 175 \
  --sex male \
  --smplx_fit \
  --save_smplx outputs/final/deva_cached/smplx \
  --json outputs/final/deva_cached/result.json \
  --skip-envelope-check
```

The output is named dimensions plus WHR, WHtR, BRI, risk categories, `health_summary`, SMPL-X reliability status, segmentation paths, and a fitted `.obj` when `--save_smplx` is supplied. If the gate rejects the capture, `health_summary.overall_risk` becomes `not_reported`; treat the measurements and indices as diagnostic only and recapture rather than reporting them.

## Local Dashboard

Install dashboard dependencies into the project environment:

```bash
.venv/bin/pip install -r requirements-dashboard.txt
```

Run the visual demo:

```bash
.venv/bin/streamlit run dashboard/app.py
```

The dashboard uploads front/side RGB photos, runs the existing segmentation and dimension pipeline, shows source images, silhouettes, predicted waist/hip/chest, WHR/WHtR/BRI, the central-adiposity risk summary, SMPL-X reliability overlays, and downloadable `result.json` / `.obj` artifacts.

For quick demos, the dashboard also supports a single-front-photo mode. That mode generates the front silhouette and SMPL-X `.obj` from one image, then duplicates the front silhouette for the model's side branch to produce diagnostic dimensions and risk output. Use true front+side capture for reportable dual-view results.

## Segmentation Smoke Check

The active segmentation path is YOLO person box + padded box-only SAM2 + light binary cleanup. There are no point prompts, GrabCut steps, or fallback masks in the research path.

```bash
PYTHONPATH=. python 5-eval/7segmentation_smoke.py \
  --root TestPhoto \
  --debug_dir outputs/final/segmentation \
  --json outputs/final/segmentation/report.json
```

This writes individual `*_final_silhouette.png` masks plus `outputs/final/segmentation/contact_sheet.png`.

## Raw Image Smoke Check

For a single RGB image through the preprocessing path:

```bash
PYTHONPATH=. python 5-eval/2eval.py \
  --model_path "checkpoints/best_640x480_v4_resnet.pt" \
  --image_path "path/to/image.jpg" \
  --height_cm 175
```

This is useful for checking preprocessing on one image. The research training path still uses paired front/side silhouettes from `data/bodym/pairs_dimensions.csv`.

## Evaluation

Run metadata leakage and derived-index ablations:

```bash
PYTHONPATH=. python 5-eval/4ablate.py \
  --csv "data/bodym/pairs_dimensions.csv" \
  --ckpt "checkpoints/best_640x480_v4_resnet.pt" \
  --measurement_cols "waist_cm,hip_cm,chest_cm" \
  --tp_on waist_cm
```

Run the legacy proxy SMPLX reliability coverage table for BodyM mask-only rows:

```bash
PYTHONPATH=. python 5-eval/6gate_eval.py \
  --csv "data/bodym/pairs_dimensions.csv" \
  --ckpt "checkpoints/best_640x480_v4_resnet.pt" \
  --measurement_cols "waist_cm,hip_cm,chest_cm" \
  --max_rows 50 \
  --device cpu
```

Compare two checkpoints:

```bash
PYTHONPATH=. python 5-eval/5compare.py \
  --csv "data/bodym/pairs_dimensions.csv" \
  --ckpt_contrastive "checkpoints/contrastive.pt" \
  --ckpt_regonly "checkpoints/regonly.pt" \
  --measurement_cols "waist_cm,hip_cm"
```

Mask IoU evaluation remains in `5-eval/3iou.py`.

## Segmentation Assets

- Binary weights live under `models/segmentation/`.
- YAML/config files live under `configs/segmentation/`.

## SMPL-X and NLF Assets

- SMPLX `.npz` model files live under `models/smplx/`.
- The default NLF model lives at `models/nlf/nlf_l_multi.torchscript`.
- The NLF reliability gate uses the front RGB image to recover SMPL-X, then renders it back to the front silhouette for abstention.
- The legacy `--smpl_gate` path remains available for BodyM mask-only retained-error experiments, but the active phone-reporting path is `--smplx_fit`.
