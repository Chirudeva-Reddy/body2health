# UAE deployable VLM project tracks

Research date: 4 September 2026.

## Executive recommendation

Propose **a plate-waste and food-surplus visual measurement assistant**. A diner or cafeteria worker photographs a plate before and after a meal. The system segments visible food, assigns a constrained food-group label, estimates a *leftover fraction band*, explains its confidence, and asks for human correction when uncertain. It is a sustainability tool, not a calorie counter or nutrition/health adviser.

This is a stronger replacement for Body2Health: it uses real multimodal vision engineering, collects a small UAE-specific evaluation set rather than relying on incomplete historical series, has measurable failure modes, and can be deployed as a privacy-conscious PWA.

The UAE Circular Economy Policy explicitly prioritises food-loss/waste reduction and biological-waste recycling. Dubai Municipality has also called for innovation in sustainable food systems and publishes food-waste guidance for retailers. [UAE Circular Economy Policy 2021-2031](https://uaelegislation.gov.ae/en/policy/details/sy-s-dol-l-m-r-t-laarby-lmthd-ll-kts-d-ld-ry-2021-2031) · [Dubai Municipality food-systems initiative](https://www.dm.gov.ae/safe-food-systems-for-a-sustainable-future/) · [Dubai Municipality food-waste guide](https://www.dm.gov.ae/wp-content/uploads/2024/04/Guide-to-Reducing-Food-Waste-in-Supermarkets-and-Hypermarket.pdf)

## What makes it substantial

- **Vision pipeline:** food-region detection/segmentation, constrained VLM food-group labelling, leftover-fraction estimation, confidence calibration, and a human-correction loop.
- **Local evaluation dataset:** 250-400 consented plate photographs gathered over 4-6 weeks at one university cafeteria, event, or household cohort. Record before/after image, food group, plate type, and measured leftovers using a kitchen scale. No multi-year public dataset is required.
- **Research question:** does grounding the VLM to segmented food regions and limiting it to a fixed taxonomy reduce error and hallucination compared with direct VLM prompting?
- **Evaluation:** macro-F1 for food groups; mIoU for annotated food masks; MAE for leftover fraction against weighed/annotated ground truth; calibration and abstention coverage; and median end-to-end latency. Compare VLM-only with VLM + segmentation + constrained labels.
- **Deployment:** mobile PWA -> image resize and EXIF removal -> FastAPI worker -> overlay/result -> optional human correction. Use temporary storage only for consented research images; do not accept faces and do not retain images by default.

## Pretrained-model stack

Use a VLM for constrained image understanding, not as an unchecked chatbot. Florence-2 supports prompt-based captioning, detection, segmentation and OCR; its model card documents local/server serving options. [Microsoft Florence-2 model card](https://huggingface.co/microsoft/Florence-2-base)  
Use SAM 2 only where it materially improves the food-region overlay; Meta documents promptable image/video segmentation and Apache-2.0 model licensing. [Meta SAM 2 repository](https://github.com/Segment-Anything/segment-anything-2)

For Arabic/English text-bearing images, PaddleOCR offers multilingual recognition with Arabic support. [PaddleOCR multilingual documentation](https://swhl.github.io/PaddleOCR/main/en/version3.x/algorithm/PP-OCRv5/PP-OCRv5_multi_languages.html)

Run a local demo with a small vision model through Ollama, then move the inference worker to a GPU-backed OpenAI-compatible vLLM endpoint only if benchmarking shows it is needed. [Ollama vision documentation](https://docs.ollama.com/capabilities/vision) · [vLLM multimodal documentation](https://docs.vllm.ai/en/latest/features/multimodal_inputs/)

## Other credible tracks

| Track | Core VLM/pretrained work | Local data and evaluation | Go/no-go condition |
|---|---|---|---|
| **Bilingual packaging-to-disposal evidence assistant** | Arabic/English OCR plus VLM constrained to a small material taxonomy; visible evidence overlay; abstain on unclear packaging. | 250-400 locally photographed packages; macro-F1, OCR character error, calibration, latency. | Proceed if disposal guidance can be cited from a maintained authority. Never present a disposal decision as authoritative. |
| **Visual accessibility and alt-text quality auditor** | VLM creates Arabic/English draft alt text; OCR grounds embedded text; rubric flags missing essential content or hallucination. | 300-500 public/consented images with human reference descriptions; factuality, coverage, hallucination rate, reviewer agreement. | Proceed if framed as an audit assistant, not formal accessibility certification. The UAE policy explicitly supports accessible digital products. [National Digital Accessibility Policy](https://uaelegislation.gov.ae/en/policy/details/the-national-digital-accessibility-policy) |
| **Date-palm symptom screen with expert referral** | VLM quality gate and broad visible-symptom label; SAM evidence overlay; official-guide retrieval. | 150-300 UAE palm photos; expert-labelled holdout; sensitivity for referral, false-negative rate, calibration. | Only proceed after securing a farm/landscape partner and an agricultural expert. Do not diagnose or prescribe treatment. |
| **Campus litter survey and cleanup-prioritisation tool** | Grounding-DINO proposes a limited litter vocabulary; SAM 2 creates masks; VLM verifies uncertain detections. | 300-600 approved-route frames; mAP@50, mask IoU, count MAE, false positives, reviewer time saved. | Use approved sites only, blur faces/plates, and prohibit enforcement or public surveillance. |
| **Pedestrian-accessibility field-survey assistant** | Vision checklist for visible curb ramp, tactile paving, crossing marking, obstruction, or not-assessable; human-correctable overlay. | 150-image independently labelled holdout; checklist precision/recall, abstention quality, correction time. | Never declare a route safe or accessible: a photo cannot establish slope, width, continuity, or traffic safety. |

## What to exclude

- No project requiring complete multi-year Bayanat-style series; each option works from a documented, bounded local collection.
- No body-image, diagnosis, facial-recognition, surveillance, or automated enforcement system. UAE personal-data law treats facial images used to identify a person as biometric data. [Federal Decree-Law No. 45 of 2021](https://www.uaelegislation.gov.ae/en/legislations/1972/download)
- No “model training from scratch.” Fine-tuning is optional and only after a zero-shot baseline and a held-out local evaluation exist.
- No ungrounded VLM answers: constrain classes, expose evidence, measure confidence, and abstain on ambiguity.

## Professor-ready pitch

> I would like to work on a UAE-focused, deployable computer-vision project rather than a simple dashboard. My preferred direction is a privacy-conscious plate-waste measurement assistant for cafeterias or households. It would use pretrained vision-language and segmentation models to identify visible leftover food, estimate a leftover fraction, and provide uncertainty-aware reduction feedback. I would build a small locally collected, consented evaluation dataset rather than rely on incomplete multi-year public data. The project would be evaluated with food-group F1, segmentation IoU, leftover-estimation MAE, calibration, and deployment latency. Would this align with the level and research direction you have in mind, or is there a related UAE problem you would prefer me to tackle?

