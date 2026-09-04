# Research source and claim ledger: UAE deployable VLM tracks

## Scope and assumptions

- **Decision:** select a substantial, deployable UAE course project using pretrained VLM/CV models.
- **Geography/time:** UAE, current sources checked 4 September 2026.
- **Exclusions:** incomplete multi-year Bayanat-style time series, sensitive biometrics, health diagnosis, public surveillance, physical hardware, and custom training as a prerequisite.
- **Success:** a privacy-aware PWA/web deployment, a bounded locally collected evaluation set, a clear baseline, objective metrics, and an abstention path.

## Claim ledger

| Claim | Evidence | Confidence | Limitation |
|---|---|---|---|
| Food-loss/waste reduction and biological-waste recycling are policy priorities. | [UAE Circular Economy Policy 2021-2031](https://uaelegislation.gov.ae/en/policy/details/sy-s-dol-l-m-r-t-laarby-lmthd-ll-kts-d-ld-ry-2021-2031), UAE Legislation, issued 2021 and current page updated June 2026. | High | A broad policy does not validate a particular application. |
| Dubai Municipality has recently promoted innovation and retailer guidance for food-waste reduction. | [Safe Food Systems for a Sustainable Future](https://www.dm.gov.ae/safe-food-systems-for-a-sustainable-future/), Dubai Municipality, 27 September 2024; [Food-waste guide](https://www.dm.gov.ae/wp-content/uploads/2024/04/Guide-to-Reducing-Food-Waste-in-Supermarkets-and-Hypermarket.pdf), Dubai Municipality. | High | Neither source establishes plate-image estimation accuracy. |
| UAE digital accessibility policy covers products, websites, applications, software and interfaces for People of Determination and older people. | [National Digital Accessibility Policy](https://uaelegislation.gov.ae/en/policy/details/the-national-digital-accessibility-policy), UAE Legislation, issued 2024 and current page updated June 2026. | High | An audit tool cannot certify compliance. |
| UAE food-labelling policy is a reasonable context for a bilingual label evidence tool. | [Food Labelling Policy](https://uaelegislation.gov.ae/en/policy/details/sy-s-tosym-lkym-lghth-y-llmntg-t), UAE Legislation, issued 2019 and current page updated June 2026; [Cabinet Resolution 83 of 2024](https://www.uaelegislation.gov.ae/en/legislations/2594/download). | High | It is not legal advice or an official compliance verdict. |
| Florence-2 is a prompt-based vision foundation model with captioning, detection, segmentation and OCR tasks. | [Microsoft Florence-2 model card](https://huggingface.co/microsoft/Florence-2-base), Microsoft, current. | High | Model-card capability is not local-domain accuracy. |
| SAM 2 supports prompted segmentation in image and video and has an Apache-2.0 model licence. | [SAM 2 repository](https://github.com/Segment-Anything/segment-anything-2), Meta FAIR, current. | High | Interactive performance depends on hardware. |
| PaddleOCR provides Arabic-capable multilingual recognition. | [PP-OCRv5 multilingual documentation](https://swhl.github.io/PaddleOCR/main/en/version3.x/algorithm/PP-OCRv5/PP-OCRv5_multi_languages.html), PaddleOCR, current. | High | Published benchmarks do not establish UAE-package-label accuracy. |
| Privacy precautions are necessary where a project could contain identifiable people. | [Federal Decree-Law No. 45 of 2021](https://www.uaelegislation.gov.ae/en/legislations/1972/download), UAE Legislation. | High | Institutional review may impose additional requirements. |

## Search and stopping note

Research used official UAE legislation and municipal sources for problem relevance, plus model-owner/first-party technical documentation. It stopped after the sources converged on three viable families: food-waste measurement, packaging/recycling evidence, and digital accessibility. The remaining material uncertainty is not a source gap: it is whether the student can recruit a small consented local image set and, for the palm option, an expert labeler. Those are go/no-go checks before implementation.
