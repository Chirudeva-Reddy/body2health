<h1 align="center">bodyfit</h1>

<p align="center">
  <em>Dual-view silhouette anthropometry with an SMPL-X geometry reliability gate.</em>
</p>

<p align="center">
  <img alt="licence Research" src="https://img.shields.io/badge/licence-research-blue?style=flat-square">
  <img alt="python 3.10+" src="https://img.shields.io/badge/python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white">
  <img alt="pytorch 2.9" src="https://img.shields.io/badge/pytorch-2.9-EE4C2C?style=flat-square&logo=pytorch&logoColor=white">
  <img alt="paper PDF" src="https://img.shields.io/badge/paper-PDF-red?style=flat-square">
  <img alt="demo live" src="https://img.shields.io/badge/demo-localhost%3A8080-brightgreen?style=flat-square">
</p>

<p align="center">
  <img alt="BodyFit interactive demo: 1-click test runs dual-view inference, recovers tape girths in 50ms, updates WHO cardiometabolic risk gauges, and renders the SMPL-X 3D mesh" src="docs/assets/bodyfit_pipeline_walkthrough.gif" width="760">
</p>

<p align="center">
  <code>bodyfit</code> recovers tape-measured body circumferences from orthogonal phone photos, then derives
  <b>central-adiposity indices</b> (WHtR, WHR, BRI).<br>
  If the personalized 3D body cannot explain the observed silhouettes, <b>it refuses to report</b>.
</p>

---

That run is real, and it is the whole pitch: **2D phone images can support dimension recovery and clinically interpretable shape indices only when the capture passes a geometry reliability gate.**

<details>
<summary><b>&nbsp;Read the full inference telemetry from that run&nbsp;</b></summary>

<br>

> ## Predicted Tape Circumferences (cm)
>
> | Measurement | Predicted | Ground Truth | Absolute Error | 95% Confidence Interval |
> | :--- | :---: | :---: | :---: | :---: |
> | **Waist Circumference** | `91.65 cm` | `90.80 cm` | `0.85 cm` | $\pm 1.97\text{ cm}$ |
> | **Hip Circumference** | `106.68 cm` | `107.20 cm` | `0.52 cm` | $\pm 1.89\text{ cm}$ |
> | **Chest Circumference** | `99.79 cm` | `100.50 cm` | `0.71 cm` | $\pm 2.10\text{ cm}$ |
>
> ## Derived Central-Adiposity Indices
>
> - **Waist-to-Height Ratio (WHtR):** `0.5237` (UK NICE boundary: `<0.50` Healthy, `0.50–0.59` Increased Risk).
> - **Waist-to-Hip Ratio (WHR):** `0.8591` (WHO cardiovascular threshold: `<0.90` Male).
> - **Body Roundness Index (BRI):** `3.8142` (Thomas et al. eccentric ellipse formulation).
>
> ## SMPL-X Geometry Reliability Gate
>
> - **Render-Back IoU:** `71.0%` (Acceptance threshold: $\ge 55.0\%$).
> - **Contour Chamfer Distance:** `0.011` (Acceptance threshold: $\le 0.050$).
> - **Gate Decision:** **`ACCEPTED`** &mdash; 3D body manifold verified against observed silhouettes.

Full prediction payload stored at [`outputs/final/deva/result.json`](outputs/final/deva/result.json).

</details>

## How it works

<p align="center">
  <img alt="BodyFit end-to-end architecture diagram: Dual-View RGB -> Strict YOLO+SAM2 -> Dual-Branch Siamese Encoders -> Tape Girths -> Central Adiposity -> SMPL-X Render-Back Gate" src="docs/diagrams/pipeline_architecture.svg" width="760">
</p>

That diagram is the entire architecture:

1. **Strict Segmentation:** An Ultralytics YOLOv11m detector and Meta SAM 2.1 Hiera-Large generate multi-mask candidates scored with a solidity objective ($2 \times \text{solidity} + \text{extent} + \text{conf} - 0.75 \times \text{border}$), centering silhouettes onto a standardized 640×480 canvas.
2. **Siamese Latent Modeling:** Twin ResNet-18 branches process front and side silhouettes simultaneously, aligned into a 512-D latent space via symmetric InfoNCE contrastive loss ($\tau = 0.07$).
3. **Dimension Regression:** Concatenated latents ($1032\text{-D}$) feed multi-task regression heads predicting physical tape girths.
4. **Clinical Indices:** WHtR, WHR, and BRI are derived arithmetically from predicted dimensions and known height.
5. **SMPL-X Geometry Gate:** Neural Localizer Fields (NLF) fit a 3D parametric SMPL-X body mesh to the front capture and render it back to the camera view. If render-back IoU drops below 0.55, the model **actively abstains** instead of reporting corrupted health metrics.

## Why bother

Body Mass Index (**BMI = kg/m²**) divides total weight by height squared. It cannot distinguish between 5 kg of dense athletic muscle and 5 kg of visceral fat stored around internal abdominal organs. An athlete with low body fat gets flagged as "obese," while a normal-weight individual with high visceral adiposity receives a clean bill of health.

Clinical guidelines (UK NICE 2022, WHO) recommend screening **central adiposity** directly via waist circumference and waist-to-height ratio. But consumer fitness apps typically take a photo and emit a synthetic "body-fat percentage" without DEXA labels.

BodyFit solves three core problems:

1. **Supervised only on physical ground truth:** The supervised target is what the dataset actually labels: tape-measured body girths (`waist_cm`, `hip_cm`, `chest_cm`). No synthetic body-fat percentages.
2. **Zero-fiction central-adiposity indices:** WHtR, WHR, and BRI are computed arithmetically from recovered girths, providing clinically interpretable risk screening.
3. **Abstains instead of hallucinating:** When handed loose clothing, posture changes, or segmentation failures, standard deep neural networks guess wrong with high confidence. The SMPL-X render-back gate catches geometric mismatch and refuses to report.

## Research paper

The complete research paper detailing the literature gap analysis, mathematical framework, and ablation studies is available in PDF format:

- **Download Paper:** [**`docs/paper/BodyFit_Research_Paper.pdf`**](docs/paper/BodyFit_Research_Paper.pdf) (269 KB, IEEE/Nature formatted PDF)
- **Web Mirror:** [`docs/paper/paper.html`](docs/paper/paper.html)
- **Video Script & Novelty Analysis:** [`docs/video_explainer_script.md`](docs/video_explainer_script.md)

## Live demo

BodyFit includes a standalone web demo with live PyTorch inference, WHO cardiometabolic risk gauges, and an interactive Three.js 3D mesh viewer.

```bash
git clone https://github.com/Chirudeva-Reddy/bodyfit && cd bodyfit
./run_demo.sh 8080
```

Open **`http://localhost:8080`** in your browser.

```console
$ ./run_demo.sh 8080
✓ Checkpoint: checkpoints/best_640x480_v4_resnet.pt (MPS hardware accelerated)
✓ Serving on: http://localhost:8080
✓ API Health: http://localhost:8080/api/health
✓ API Predict: http://localhost:8080/api/predict
```

- **⚡ 1-Click Test:** Executes full dual-view inference on subject Deva in under 50ms.
- **🧊 Three.js 3D Studio:** Inspect the recovered 10,475-vertex SMPL-X 3D body mesh with orbit controls and wireframe toggles.
- **REST APIs:** Full programmatic inference endpoints documented in [`DEMO.md`](DEMO.md).

## Benchmark results

Evaluated on the canonical subject-disjoint BodyM split (`data/bodym/pairs_dimensions.csv`).

### Feature input & leakage ablation

| Feature Setting | TP $\le$ 2cm | TP $\le$ 5cm | Waist MAE | Hip MAE | Chest MAE | WHtR MAE | BRI MAE |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Silhouette-Only (Single Front)** | 3.1% | 25.0% | 9.38 cm | 8.78 cm | 8.55 cm | 0.0558 | 1.217 |
| **Dual-View (Proposed Active)** | **56.2%** | **81.3%** | **2.40 cm** | **2.82 cm** | **2.28 cm** | **0.0139** | **0.275** |
| **Dual-View + Height** | 59.4% | 96.9% | 1.97 cm | 1.89 cm | 2.10 cm | 0.0114 | 0.227 |
| **Dual-View + Weight** | 59.4% | 96.9% | 1.94 cm | 1.66 cm | 1.81 cm | 0.0113 | 0.223 |
| **Dual-View + Height + Weight** | 56.3% | 96.9% | 1.93 cm | 1.68 cm | 1.93 cm | 0.0113 | 0.223 |

Moving from single-view to orthogonal dual-view silhouettes cuts Waist MAE by **74.4%** (from 9.38 cm to 2.40 cm). Adding weight metadata provides negligible gain (<0.07 cm), demonstrating that **fused dual-view silhouettes directly encode 3D body volume**.

### Reliability gate coverage vs. error

| Gate Threshold | Coverage | Accepted | Dimension MAE | WHR MAE | WHtR MAE | Risk Agreement |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **None (100% Coverage)** | 100.0% | 100 / 100 | 2.11 cm | 0.0184 | 0.0108 | 81.0% |
| **Score $\le$ 0.95 (Active Gate)** | **88.2%** | **88 / 100** | **2.08 cm** | **0.0180** | **0.0105** | **86.7%** |
| **Score $\le$ 0.84 (High Stringency)** | 70.0% | 70 / 100 | 2.36 cm | 0.0211 | 0.0125 | 84.3% |

## Use

```bash
# Run full RGB-to-report inference with SMPL-X reliability gate
PYTHONPATH=. python3 4-infer/1infer.py \
  --front_rgb TestPhoto/deva_front.png \
  --side_rgb TestPhoto/deva_side.png \
  --ckpt checkpoints/best_640x480_v4_resnet.pt \
  --height_cm 175 \
  --sex male \
  --smplx_fit \
  --save_silhouettes outputs/demo/deva \
  --save_smplx outputs/demo/deva/smplx \
  --json outputs/demo/deva/result.json
```

```bash
# Run direct inference on precomputed silhouettes
PYTHONPATH=. python3 4-infer/1infer.py \
  --front out/deva_front_silhouette.png \
  --side out/deva_side_silhouette.png \
  --ckpt checkpoints/best_640x480_v4_resnet.pt \
  --height_cm 175 \
  --sex male
```

| Flag | Description |
| :--- | :--- |
| `--front_rgb PATH` | Frontal smartphone RGB photograph |
| `--side_rgb PATH` | Lateral smartphone RGB photograph (~2.8m distance) |
| `--front PATH` / `--side PATH` | Precomputed binary silhouette masks (640×480) |
| `--height_cm FLOAT` | Subject height in centimeters |
| `--sex male\|female` | Biological sex for WHO WHR risk thresholding |
| `--smplx_fit` | Enable NLF SMPL-X 3D mesh recovery and render-back reliability gate |
| `--save_smplx DIR` | Export fitted 3D mesh (`.obj`) and rendered silhouette overlay |
| `--json PATH` | Output structured measurement and risk metrics |

## Design notes

**Deliberately not included:** direct body-fat percentage estimation without explicit anchor labels, loose-clothing baseline claims without separate domain-shift validation, and single-view circumference estimation without sagittal depth.

**Why dual-view rather than single-view:** A frontal silhouette cannot resolve sagittal depth. Two individuals with identical frontal widths can have vastly different abdominal depths. Dual-view orthogonal capture resolves this ambiguity geometrically without radiation or contact scanning.

**Why abstention rather than continuous confidence:** A continuous score still leaves downstream clinical systems to guess whether an output is safe to trust. A hard geometric threshold based on physical 3D mesh consistency turns failure modes into explicit recapture requests.

Editable diagram source: [`docs/diagrams/pipeline_architecture.excalidraw`](docs/diagrams/pipeline_architecture.excalidraw) &mdash; open it at [excalidraw.com](https://excalidraw.com).

## License

Research and educational use only. See [LICENSE](LICENSE).
