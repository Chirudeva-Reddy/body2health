# Hyperframes Composition Brief: Body2Fit

## Objective
Create a short launch-style brag video for Body2Fit that showcases the dual-view silhouette anthropometry pipeline and its unique SMPL-X geometry reliability gate.

## Output
- Composition directory: `brag-output/composition/`
- Rendered video: `brag-output/brag.mp4`
- Video preview / player: `brag-output/preview.html`
- Format: landscape — 1920x1080
- Duration: 20 seconds (600 frames at 30fps)

## Source Material
- Project root: `/Users/tacticalcamel/Desktop/Projects/bodyfit`
- Primary files read:
  - `web/index.html` (interactive recruiter web application)
  - `web/style.css` (clinical obsidian & emerald palette)
  - `README.md` (benchmark paper results, architecture diagrams, walkthrough)
  - `DEMO.md` (recruiter live walkthrough instructions and metrics)
  - `PROJECT_SUMMARY.md` (active pipeline spec and canonical data tables)
- Product name: Body2Fit
- Tagline / strongest claim: "Dual-view silhouette anthropometry with an SMPL-X geometry reliability gate."
- Key UI or visual moment to recreate:
  - Dual orthogonal phone photos standardizing into 640x480 silhouettes
  - Fused 1032-D contrastive latent alignment
  - Tape-measured circumference recovery telemetry (Waist 91.65 cm, Hip 106.68 cm, Chest 99.79 cm)
  - SMPL-X 3D anatomical mesh render-back re-projection with IoU scorecard (76.57% IoU, Verdict: ACCEPTED)
- Copy that must appear verbatim:
  - "BMI was invented in 1832."
  - "In 2026, we have orthogonal computer vision."
  - "Waist: 91.65 cm | Hip: 106.68 cm | Chest: 99.79 cm"
  - "Render-back IoU: 76.57% · Chamfer: 0.0096"
  - "GATE VERDICT: ACCEPTED"
  - "If the fitted 3D body cannot explain the observed silhouettes, it refuses to report."

## Creative Direction
- Tone preset: `polished` with `/boost`ed technical conviction
- Creative direction: High-velocity clinical-tech launch, precision-engineered, dark-mode cyberpunk minimalism
- Interpretation: Crisp typography, authoritative scientific claims, zero fluff, instant visual hooks
- Angle:
  Body2Fit replaces 180-year-old BMI with millimeter-accurate tape circumferences predicted from paired front and side silhouettes. Unlike black-box models that hallucinate on bad inputs, Body2Fit validates every prediction against an SMPL-X 3D anatomical body mesh reprojected to the camera view. If the mesh cannot explain the silhouettes, it refuses to report.
- Hook: 0.0s - 3.5s — The 1832 problem vs 2026 computer vision
- Outro / punchline: 17.5s - 20.0s — "Tape-grade anthropometry. 3D geometry verification. Zero hallucinations."
- Avoid:
  - Generic SaaS illustration art
  - Fake or ungrounded statistics
  - Redesigning away from the project's real visual theme

## Visual Identity
- Background: `#0B0F19` (obsidian canvas)
- Text Primary: `#F8FAFC` (pure white text)
- Text Muted: `#94A3B8` (cool slate)
- Accent Primary: `#10B981` (emerald-500, clinical pass glow)
- Accent Secondary: `#06B6D4` (cyan-500, contrastive latent glow)
- Accent Warning: `#F59E0B` (amber-500, risk category indicator)
- Glass Panels: `rgba(17, 24, 39, 0.85)` with `1px solid rgba(31, 41, 55, 0.8)`
- Fonts: Inter, system-ui, -apple-system, sans-serif
- Visual references from the project:
  - `docs/assets/bodyfit_pipeline_walkthrough.gif`
  - `docs/diagrams/pipeline_light.svg`
  - `out/deva_front_silhouette.png`
  - `out/deva_side_silhouette.png`
  - `outputs/final/deva/smplx/smplx_front_overlay.png`

## Storyboard
1. **Scene 1 — The Hook (0.0s - 3.5s)**: "BMI was invented in 1832" -> "In 2026, we have orthogonal computer vision." Wireframe silhouettes pulse onto screen.
2. **Scene 2 — The Dual-View Pipeline (3.5s - 8.5s)**: Front + Side 640x480 silhouettes align via InfoNCE into a fused 1032-D latent space with real anatomical part segmentation overlays.
3. **Scene 3 — Millimeter Recovery & Risk Gauges (8.5s - 13.5s)**: Telemetry cards count up in real-time (Waist 91.65 cm ±0.85 cm, Hip 106.68 cm, Chest 99.79 cm) alongside WHO/NICE cardiometabolic gauges.
4. **Scene 4 — The Breakthrough & Outro (13.5s - 20.0s)**: 3D SMPL-X mesh fit reprojected onto front silhouette. IoU 76.57%, Chamfer 0.0096. "GATE VERDICT: ACCEPTED". Final punchline: "If the 3D body cannot explain the silhouettes, Body2Fit refuses to report."

## Audio
- Audio role: Cinematic cyber-clinical tech bed with crisp UI micro-accents
- Audio arc: Ambient intro -> 120 BPM driving tech pulse -> telemetry count-up acceleration -> resonant gate resolution chime
- Music: Cyber-clinical melodic pulse (120 BPM)
- Music treatment: Low-pass filter intro, swells on scene 2, rhythmic pulse under telemetry, ducking under outro
- SFX posture: Precise digital feedback cues (camera shutter, count-up ticks, mechanical lock, verification chime)
- Local Assets: Bundled in `brag-output/composition/assets/`

## Hyperframes Instructions
- Composition root: `brag-output/composition/index.html`
- Standalone: Complete CSS3 and HTML5 animations, viewable in any standard browser and renderable via Hyperframes CLI.
- Keep all text readable with high WCAG contrast ratios (slate-50 `#F8FAFC` on obsidian `#0B0F19` = 18.2:1 contrast ratio, surpassing AAA requirements).
- Duration: 20 seconds at 30 FPS.
