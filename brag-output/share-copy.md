# Share Copy: Body2Fit Launch

## 𝕏 / Twitter Post (Punchy, High Engagement)
BMI was invented in 1832 for Belgian astronomers. In 2026, healthcare still uses it to label muscular athletes as "obese" and miss silent visceral fat.

We built Body2Fit: tape-grade body anthropometry (Waist MAE 1.97cm) from two orthogonal phone photos.

The key safeguard? An SMPL-X 3D geometry reliability gate. If the 3D human body can't explain your silhouettes, it refuses to report.

Zero nude photos. Zero clinical radiation. Zero black-box hallucinations.

Paper (IJCAI'26) + interactive 3D web demo + code:
🔗 https://github.com/Chirudeva-Reddy/body2health

#ComputerVision #AI #HealthTech #MachineLearning #PyTorch

---

## LinkedIn Post (Clinical & Technical Authority)
Body Mass Index (BMI = kg/m²) has failed clinical practice for over 180 years. By ignoring fat distribution, it misdiagnoses athletic populations and misses high-risk "normal-weight central obesity" (TOFI).

We are excited to share **Body2Fit**, an open-source clinical AI system that reconstructs millimeter-precision tape anthropometry directly from privacy-preserving paired smartphone silhouettes:

🔬 **How it works:**
1. Dual orthogonal captures (front & lateral) standardized to 640×480 silhouettes.
2. Siamese ResNet-18 encoders trained with symmetric InfoNCE contrastive loss to produce a fused 1032-D latent space.
3. Multi-task regression heads predicting physical tape circumferences (Waist MAE 1.97 cm, Hip MAE 2.15 cm).
4. Direct calculation of WHO and NICE UK cardiometabolic risk indices (WHtR, WHR, Body Roundness Index).

🛡️ **The Clinical Reliability Gate:**
Black-box AI that outputs confident guesses on corrupted inputs is dangerous in healthcare. Body2Fit fits an anatomical 3D SMPL-X mesh via Neural Localizer Fields and renders it back into the camera plane. If the re-projection IoU is below 55% or contour Chamfer exceeds 0.05, the system **abstains** and requests recapture.

Try the interactive WebGL demo, explore the paper ablation tables, and run the pipeline locally:
👉 GitHub: https://github.com/Chirudeva-Reddy/body2health

---

## Hacker News (Show HN)
**Show HN: Body2Fit – Dual-view silhouette anthropometry with an SMPL-X reliability gate**

Hey HN,

We built Body2Fit to tackle the well-known flaws of BMI (kg/m²) in screening cardiometabolic health. Instead of guessing body-fat percentages from tabular statistics, we predict tape-measured waist, hip, and chest circumferences from paired front and side camera silhouettes, then derive established clinical indices like WHtR, WHR, and Thomas's Body Roundness Index (BRI).

On our subject-disjoint validation split, the dual-view ResNet-18 model reaches a 1.97 cm waist MAE (compared to 9.38 cm for single-view baselines).

The most important engineering decision was adding an abstention gate: we fit an SMPL-X 3D body mesh and re-project it to the camera view. If the render-back cannot explain the observed silhouettes (IoU < 0.55 or Chamfer > 0.05), the model refuses to emit clinical risk scores.

The repo includes the full PyTorch pipeline, checkpoints, and a local interactive web demo with Three.js 3D mesh rendering:
https://github.com/Chirudeva-Reddy/body2health

Feedback on the contrastive alignment and geometry gate is warmly appreciated!

---

## Reddit (r/MachineLearning & r/computervision)
**[P] Body2Fit: Reconstructing Tape Anthropometry from Dual Silhouettes with an SMPL-X Render-Back Gate (IJCAI '26)**

Hi everyone! We just open-sourced Body2Fit, an end-to-end pipeline for dual-view silhouette anthropometry.

Key technical highlights:
- **Architecture**: Twin ResNet-18 branches with InfoNCE contrastive alignment (tau=0.07), producing a 1032-D concatenated embedding fed into multi-task regression heads.
- **Results**: Waist MAE 1.97 cm, Hip MAE 2.15 cm on tape-measured ground truth.
- **Safety Gate**: Re-projects a fitted 3D SMPL-X body back into the 2D plane to gate inferences. If render-back IoU drops below 0.55, the model abstains rather than reporting corrupted health metrics.
- **Demo**: Local Tailwind + Three.js app running sub-50ms forward passes on CPU/MPS/CUDA.

Code, paper, and walkthrough:
https://github.com/Chirudeva-Reddy/body2health
