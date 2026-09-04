# Body2Fit Project Summary

Body2Fit is now a dimension-first silhouette anthropometry pipeline. The active v1 target is not BMI or direct body-fat percentage. The model predicts tape-measured body dimensions from paired front/side silhouettes, then computes clinically interpretable central-adiposity indices from those dimensions.

## Active Pipeline

```text
data/bodym/pairs_dimensions.csv
-> front/side silhouettes
-> DualViewContrastive
-> waist_cm, hip_cm, chest_cm
-> WHtR, WHR, BRI
-> central-adiposity risk summary
-> NLF SMPL-X render-back reliability gate
-> dashboard/report or reject
```

## Canonical Data

- Primary table: `data/bodym/pairs_dimensions.csv`
- File provenance: `data/bodym/manifest.csv`
- Subject traceability: `data/bodym/subject_key_map.csv`
- Subject measurements: `data/bodym/metadata/measurements_renamed.csv`

The primary table uses readable `sub_XXXX` and `cap_XXX` keys and stores image paths under `data/bodym/raw_masks/...`.

## Current Implementation

- Training defaults to `waist_cm,hip_cm`; the active best checkpoint uses `waist_cm,hip_cm,chest_cm` with the ResNet-18 encoder.
- Validation split is subject-disjoint through `subject_key`.
- The active contrastive model emits only named measurement targets.
- Checkpoints store `measurement_cols`.
- Inference prints predicted dimensions, WHR/WHtR/BRI, central-adiposity risk categories, risk summary, and optional SMPL-X render-back reliability.
- `dashboard/app.py` provides the local visual demo: uploaded front/side images, generated silhouettes, dimensions, indices, central-adiposity risk summary, SMPL-X overlays, and downloadable JSON/OBJ artifacts.
- Evaluation reports dimension errors, derived-index errors, metadata leakage rows, and reliability coverage/error rows.

## Research Anchor

The strongest current contribution is reliable anthropometry from 2D silhouettes:

- predict real tape-measured dimensions,
- derive WHtR/WHR/BRI without inventing labels,
- use NLF SMPL-X recovery and render-back silhouette mismatch as a reliability/abstention gate.

SMPL-X reliability is now implemented as the prototype research step. The active phone path is feed-forward NLF from front RGB, followed by front-view render-back comparison to the observed silhouette. The older lightweight proxy gate remains only for BodyM mask-only retained-error tables.

## Current Commands

Train:

```bash
PYTHONPATH=. python 3-train/1train.py \
  --csv "data/bodym/pairs_dimensions.csv" \
  --measurement_cols "waist_cm,hip_cm,chest_cm" \
  --encoder resnet18 \
  --epochs 40 \
  --ckpt_tag _v4_resnet
```

Evaluate:

```bash
PYTHONPATH=. python 5-eval/4ablate.py \
  --csv "data/bodym/pairs_dimensions.csv" \
  --ckpt checkpoints/best_640x480_v4_resnet.pt \
  --measurement_cols "waist_cm,hip_cm,chest_cm" \
  --tp_on waist_cm
```

Evaluate the reliability gate:

```bash
PYTHONPATH=. python 5-eval/6gate_eval.py \
  --csv "data/bodym/pairs_dimensions.csv" \
  --ckpt checkpoints/best_640x480_v4_resnet.pt \
  --measurement_cols "waist_cm,hip_cm,chest_cm" \
  --max_rows 50 \
  --device cpu
```

Infer with the final phone-reporting path:

```bash
PYTHONPATH=. python 4-infer/1infer.py \
  --front_rgb TestPhoto/deva_front.png \
  --side_rgb TestPhoto/deva_side.png \
  --ckpt checkpoints/best_640x480_v4_resnet.pt \
  --height_cm 175 \
  --sex male \
  --smplx_fit \
  --save_silhouettes outputs/final/deva \
  --save_smplx outputs/final/deva/smplx \
  --json outputs/final/deva/result.json
```

Run the local dashboard:

```bash
.venv/bin/streamlit run dashboard/app.py
```

Segment TestPhoto samples:

```bash
PYTHONPATH=. python 5-eval/7segmentation_smoke.py \
  --root TestPhoto \
  --debug_dir outputs/final/segmentation \
  --json outputs/final/segmentation/report.json
```

Infer with cached masks plus NLF SMPL-X reliability:

```bash
PYTHONPATH=. python 4-infer/1infer.py \
  --front out/deva_front_silhouette.png \
  --side out/deva_side_silhouette.png \
  --front_rgb TestPhoto/deva_front.png \
  --ckpt checkpoints/best_640x480_v4_resnet.pt \
  --height_cm 175 \
  --sex male \
  --smplx_fit \
  --save_smplx outputs/final/deva_cached/smplx \
  --json outputs/final/deva_cached/result.json \
  --skip-envelope-check
```
