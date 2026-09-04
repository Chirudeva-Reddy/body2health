# UAE VLM products that can be built from public data

Research date: 4 September 2026.  Scope: deployable, product-shaped course work for the UAE; no locally collected training set; no dependency on an incomplete Bayanat-style historical series. “Patent candidate” below means a **narrow engineering method worth a prior-art search**, not that it is patentable or that a patent will be granted.

## Decision

The three best candidates are:

1. **FloodRoute Evidence** — an all-weather, post-event road-access evidence service for municipalities, facilities, insurers and fleets.
2. **CoastWatch UAE** — a coastal-marine debris triage service that turns public satellite passes into ranked inspection jobs, rather than claiming to be an enforcement or pollution-certification system.
3. **ShelfProof** — an image-to-catalogue reconciliation product for supermarkets and distributors, built around product and price-tag evidence rather than a generic “AI shelf dashboard.”

They each have downloadable, commercially usable starting data; a concrete user workflow; a model stack that can run in a deployed service; and a specific technical seam that could be investigated for novelty. They do **not** need a time-complete UAE government data series or new photographs to establish the prototype and held-out benchmark.

The UAE use case is genuine: its climate-adaptation plan identifies infrastructure, water management, trade and logistics as impact areas; its 2024 NDC notes recent heavy rains and floods and the need to reduce infrastructure risk. Its Circular Economy Policy prioritises sustainable infrastructure, transport, manufacturing, and food consumption. [UAE National Climate Adaptation Action Plan](https://u.ae/en/about-the-uae/strategies-initiatives-and-awards/strategies-plans-and-visions/environment-and-energy/national-climate-adaptation-action-plan) · [UAE NDC 3.0](https://unfccc.int/sites/default/files/2024-11/UAE-NDC3.0.pdf) · [UAE Circular Economy Policy](https://u.ae/en/about-the-uae/strategies-initiatives-and-awards/policies/economy/uae-circular-economy-policy)

## Ranked product candidates

### 1. FloodRoute Evidence — recommended

**Product.** After heavy rain, a customer selects a facility, depot, district, or logistics route. The product obtains pre-/post-event Sentinel-1 SAR scenes, produces an inundation-change mask, intersects it with the customer’s route network, and issues a ranked review queue: *likely affected*, *clear*, or *insufficient evidence*. It is an operational triage tool, never a public flood warning or a claim that a road is safe.

**Ready data and terms.**

- [FloodNet](https://github.com/BinaLab/FloodNet-Supervised_v1.0) provides 2,343 post-flood aerial images with ten semantic classes, including flooded/non-flooded roads and buildings, and is released under **CDLA-Permissive-1.0**. It is downloadable from the project’s linked source; a maintained Hugging Face mirror records the same licence. [FloodNet licence/source record](https://huggingface.co/datasets/torchgeo/floodnet/commit/9330627ba945050fd340bdbfa0680f8c0591e843)
- Current UAE Sentinel-1 GRD imagery is available through Copernicus. Sentinel data are available on a free, full and open basis, including reproduction, modification, combination and distribution; Sentinel-1 provides day/night, all-weather SAR imagery. [Copernicus Sentinel terms](https://cds.climate.copernicus.eu/licences/ec-sentinel) · [Sentinel-1 collection](https://dataspace.copernicus.eu/data-collections/copernicus-sentinel-missions/sentinel-1)
- Customer-owned road/facility layers are an integration input, not training data. A demo can use OpenStreetMap only if it complies with the [ODbL](https://www.openstreetmap.org/copyright); do not mix an ODbL-derived database with incompatible proprietary geometry.

**Pretrained stack.** Fine-tune an Apache-2.0 [Prithvi-EO-1.0-100M](https://huggingface.co/ibm-nasa-geospatial/Prithvi-100M) temporal Vision Transformer on synthetic SAR/optical-aligned flood tasks or use a small Siamese change head; use [SAM 2](https://github.com/facebookresearch/sam2) (Apache-2.0) only to refine an analyst-selected visible boundary on optical context. The model card states Prithvi accepts a temporal image tensor and lists flood segmentation among its downstream examples.

**Deployable shape.** Web map/PWA -> API job -> Copernicus scene adapter -> GPU inference worker -> PostGIS route intersection -> review queue with source scene/date, confidence, and mask overlay. Cache raw scenes by tile/date and expire the derived queue after an event window.

**Product evaluation.** On FloodNet’s official held-out split: flooded-road IoU, macro-F1 for road state, calibration error, and abstention precision. On fixed UAE Sentinel-1 scenes selected *before* tuning: latency per 10 km route, percentage of route segments with evidence, and analyst agreement on a blinded 50-segment review set. The last measure is usability validation, not a claim of ground truth.

**Narrow invention to investigate.** A *route-segment evidence compositor* that (a) requires spatial agreement between a bi-temporal SAR change mask and a road’s buffered geometry, (b) discounts static water and radar shadow using a pre-event persistence score, and (c) emits a decision only when a traceable scene/date/evidence threshold is met. The potential claim is the evidence-gated route-ranking method, **not** “using AI to detect floods.”

**Caveat.** FloodNet is Hurricane Harvey aerial imagery, not UAE SAR. It is good for a product proof-of-concept but insufficient for a safety-critical UAE launch; the deployed product must retain *insufficient evidence* and needs an authorised UAE partner for operational validation. Copernicus’ flood-data licence also says its information is not a flood warning and that warnings are for authorised national/regional institutions. [CEMS-FLOODS licence](https://cds.climate.copernicus.eu/licences/cems-floods)

### 2. CoastWatch UAE — recommended

**Product.** A coastal operator gets a map of *candidate floating-debris patches*, each with imagery, confidence, look-alikes (foam, waves, ships, turbid water), and a recommended inspection order based on proximity to protected/coastal assets. The value is dispatch prioritisation and reproducible evidence, not an unsupported statement that a patch is plastic or a regulatory determination.

**Ready data and terms.**

- [MARIDA](https://doi.org/10.5281/zenodo.5151941) is a downloadable Sentinel-2 marine-debris archive. Its project repository documents patch-level images, segmentation masks, confidence masks, public download options, and the exact class mapping; the dataset documentation records **CC BY 4.0**. [MARIDA project/download instructions](https://github.com/marine-debris/marine-debris.github.io) · [MARIDA data documentation](https://data.source.coop/ntua/marida/documentation.pdf)
- Use current Sentinel-2 imagery under the same open Copernicus terms above. The public AWS COG archive has global coverage and continuously adds scenes; global Level-2A coverage begins in December 2018. [Sentinel-2 COG archive](https://registry.opendata.aws/sentinel-2-l2a-cogs/)
- The UAE government identifies marine/coastal protection and plastic-waste reduction as active waste-management concerns. [UAE waste-management policy page](https://u.ae/en/information-and-services/environment-and-energy/waste-management)

**Pretrained stack.** Prithvi temporal features or a compact U-Net head for multispectral segmentation, plus an Apache-2.0 Qwen [Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) only to turn fixed model outputs and structured metadata into a human-readable evidence card. Do not ask the VLM to discover debris directly from a low-resolution image; it should be a constrained explainer.

**Deployable shape.** Map UI -> AOI/date selection -> Sentinel-2 cloud-quality filter -> multispectral inference -> candidate clustering -> work-order API/export. It can be deployed as a GPU-backed API plus a static React/MapLibre front end; no ongoing bespoke imagery collection is required.

**Product evaluation.** MARIDA test split: debris-class average precision, IoU, false-positive rate against the “look-alike” categories, calibration, and time-to-ranked-queue. Product acceptance criterion: all surfaced candidates include a scene ID/date, segmentation overlay, confidence, and a distinct “uncertain” state.

**Narrow invention to investigate.** A *look-alike contradiction gate* that combines segmentation logits with an explicit water-condition classifier and local temporal consistency; it suppresses a debris alert when foam/wake/ship evidence contradicts it, while preserving the reason and source pixels for inspection. The candidate claim is the contradiction-aware alert gating and evidence record, not marine-debris segmentation itself.

**Caveat.** Sentinel-2’s 10 m pixels cannot reliably identify small items or establish material type. UAE coast-specific precision needs partner validation before any commercial pollution-monitoring claim. Do not train on [MADOS](https://zenodo.org/records/10664073) for this product without a separate licence decision: it is downloadable but published as **CC BY-NC-SA 4.0**, so it fails this commercial-ready screen.

### 3. ShelfProof — recommended

**Product.** A supermarket/distributor uploads an existing shelf photograph plus its catalogue/price list. ShelfProof detects product and price-tag regions, reads visible text, reconciles it against the supplied catalogue, and produces a review queue for: *missing facing*, *price-label mismatch*, *unreadable label*, and *insufficient evidence*. It is a catalogue and merchandising QA product, not an autonomous checkout or a price-compliance determination.

**Ready data and terms.**

- [FineGrainOCR](https://github.com/Tubbias/finegrainocr) is a downloadable multimodal grocery-product dataset with original images and OCR JSON, released under **CC0-1.0**. It contains deliberately similar grocery products where packaging/OCR distinctions matter.
- The [Supermarket Shelves Dataset](https://www.kaggle.com/datasets/humansintheloop/supermarket-shelves-dataset) contains 45 copyright-free shelf images and 11,743 product/price annotations, dedicated to the public domain under **CC0-1.0**. It is small, but suitable as a strictly legal end-to-end demo/evaluation supplement.
- If product facts are useful for a demo, [Open Food Facts’ documented export/API](https://openfoodfacts.github.io/documentation/docs/Product-Opener/api/tutorials/license-be-on-the-legal-side/) is available under ODbL (database) and CC BY-SA (images). Those obligations are workable, but a commercial product should use the customer’s catalogue as the system of record and keep Open Food Facts optional.

**Pretrained stack.** Product/label proposal model (Grounding DINO or RT-DETR under their individually verified terms) -> [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) (Apache-2.0, multilingual) -> Qwen2.5-VL (Apache-2.0) restricted to schema-constrained reconciliation explanations. The key is evidence-grounded OCR/crop matching, not “ask a VLM what is on the shelf.”

**Deployable shape.** Browser image upload -> API -> detector/OCR worker -> catalogue matching service -> review UI showing the original crop and matched field -> CSV/webhook export. CPU OCR plus a small GPU detector makes it feasible to deploy on a modest cloud GPU.

**Product evaluation.** FineGrainOCR split: item retrieval top-1/top-5, OCR character error rate, and attribute-match F1. Shelf dataset: product/price-tag detection AP and mismatch precision at a fixed review-budget. Add median image turnaround and percentage of flags with a reviewable crop.

**Narrow invention to investigate.** A *two-sided product-evidence graph*: a catalogue assertion is accepted only if its matching product crop and price-tag crop independently agree on required fields, while occlusion/unreadability becomes a first-class output rather than a guessed mismatch. The candidate claim is this evidence-graph decision rule and review protocol, not OCR, object detection, or planogram analysis generally.

**Caveat.** The public shelf datasets are not UAE stock and do not establish Arabic label accuracy. Start with language-agnostic product-count/visible-text reconciliation; add Arabic compliance requirements only after an authorised retailer supplies licensed, representative material. Do not use SKU-110K: it is widely downloadable but explicitly **research-only**. [SKU-110K terms](https://github.com/eg4000/SKU110K_CVPR19#dataset)

## Other viable, lower-priority products

| Product | Public data actually usable | VLM/CV product workflow | Evaluation and narrow technical seam | Important caveat |
|---|---|---|---|---|
| **WasteStream** — visual contamination/litter triage for campuses, malls, or facilities | [TACO](https://github.com/pedropro/TACO) supplies 1,500 real-world litter images and 4,784 annotations under **CC BY 4.0**; [TrashNet](https://github.com/garythung/trashnet) supplies 2,527 item images under **MIT**. | Detector/segmenter -> constrained VLM class verifier -> reviewable bin/zone work order. Deployed against existing approved CCTV or uploaded maintenance photos; no training collection needed. | mAP/IoU, count MAE, false-dispatch rate, calibration. Candidate seam: combine region-level material evidence with a “cannot classify” state so unobserved contamination never becomes a disposal instruction. | It needs camera/privacy approval in deployment and should not identify or enforce against people. UAE waste policy supports better segregation, not camera surveillance. |
| **CropTriage UAE** — greenhouse tomato/pepper visible-symptom routing | [PlantVillage](https://data.mendeley.com/datasets/tywbtsjrjv/1) makes 54,000+ labelled leaf images available under **CC BY 3.0**; its documented classes cover tomato and pepper health/disease labels. | Plant classifier + SAM lesion overlay + VLM-generated *observation report* with a “seek agronomist” route, deployed as a grower PWA. | Per-class macro-F1, abstention recall on unknown/poor-quality images, lesion-overlay IoU, and latency. Candidate seam: confidence is accepted only when classification and lesion geometry agree; otherwise route for inspection. | PlantVillage is largely controlled-background imagery and does not cover date palms; do not market as diagnosis or treatment recommendation before UAE field validation. The UAE food strategy explicitly supports technology-enabled sustainable food production. [Food Security Strategy 2051](https://u.ae/en/about-the-uae/strategies-initiatives-and-awards/strategies-plans-and-visions/environment-and-energy/national-food-security-strategy-2051) |
| **TrayMap** — ingredient-aware food-service waste-prevention workflow | [FoodSeg103](https://huggingface.co/datasets/EduardoPacheco/FoodSeg103) is a downloadable **Apache-2.0** segmentation benchmark: 7,118 images with 103 ingredient labels and official train/test folders. | Segmentation -> constrained VLM description -> kitchen/menu analyst dashboard that identifies menu components frequently left untouched in submitted tray images. | Ingredient macro-IoU, per-component recall, abstention/overlay quality, and time-to-correct an analyst suggestion. Candidate seam: a cross-image component matcher that reports a menu pattern only when ingredient segmentation and layout similarity jointly meet thresholds. | This is viable only as a kitchen analytics workflow. It must not present caloric, dietary, allergy, or medical advice; public data do not prove UAE menu performance. |
| **MangroveProof** — restoration-plot evidence and change dossier | [Global Mangrove Watch v4.1](https://www.eorc.jaxa.jp/ALOS/en/dataset/gmw_e.htm) provides downloadable global mangrove maps for **41 annual epochs (1985–2025)** under free public-use terms with credit; use current open Sentinel-1/2 imagery for an AOI. | Multi-temporal geospatial Vision Transformer -> change candidates -> evidence dossier for an authorised conservation operator; map a restoration plot, not an estimated carbon-credit ledger. | Agreement with withheld GMW epochs, temporal stability, change-detection F1 where labels are available, and source/evidence completeness. Candidate seam: a two-sensor temporal-consistency rule that refuses a restoration-change conclusion unless SAR and optical evidence agree or a documented cloud exception applies. | The global map has varying local accuracy and is not proof of carbon stock or restoration success. UAE policy relevance is strong: official sources describe UAE mangrove restoration guidance and a 100-million-mangrove goal. [EAD/MOCCAE guidelines](https://www.ead.gov.ae/en/Media-Centre/News/MANGROVE-INITIATIVE-GUIDELINES) |
| **PermitSight** — bilingual document packet completeness assistant | IBM [DocLayNet](https://github.com/DS4SD/DocLayNet) has 80,863 manually annotated document pages under **CDLA-Permissive-1.0**. | Layout detector -> PaddleOCR -> Qwen schema extraction -> evidence-linked “missing/ambiguous field” review UI, deployed as a document-upload service. | Layout AP, key-value field accuracy, evidence-box IoU, and reviewer minutes saved. Candidate seam: field completeness is calculated only from cross-page visual evidence and pre-declared packet rules, preserving a counterexample crop for each missing-field claim. | There is no verified public UAE Arabic permit benchmark in this screen. The prototype can demonstrate layout/evidence handling, but not claim UAE regulatory compliance without an authorised document partner. |

## What I rejected despite public access

- **S1GFloods**: a valuable global Sentinel-1 benchmark (5,360 image pairs across 46 events) but the dataset authors state it is **non-commercial/research only**; its repository’s code licence does not change the dataset restriction. [S1GFloods terms](https://github.com/Tamer-Saleh/S1GFlood-Detection)
- **Sen1Floods11**: downloadable, but its upstream repository currently has no explicit dataset licence and an open “Missing LICENSE file” issue. It should not be used in a commercial/patent-oriented training pipeline until the owners grant terms. [Repository issue list](https://github.com/cloudtostreet/Sen1Floods11/issues)
- **MADOS**: public and technically excellent, but the dataset is **CC BY-NC-SA 4.0**; that conflicts with a commercial-ready product path. [MADOS record](https://zenodo.org/records/10664073)
- **SKU-110K, Grocery Dataset, DeepGlobe and WorldFloods**: each is public in some sense, but its terms are research-only, challenge-only, or non-commercial. They are not substitutes for a deployable product dataset. [SKU-110K](https://github.com/eg4000/SKU110K_CVPR19#dataset) · [Grocery Dataset](https://github.com/gulvarol/grocerydataset) · [DeepGlobe FAQ](https://deepglobe.org/resources.html) · [WorldFloods terms](https://github.com/spaceml-org/ml4floods/blob/main/jupyterbook/content/worldfloods_dataset.md)

## Patent and launch boundary

UAE law requires a new invention with an inventive step and industrial applicability; publicly disclosed prior art defeats novelty. It is therefore unsafe to promise a patent for a broad “VLM application” or to publish a detailed claim before a professional prior-art review. [Federal Law No. 11 of 2021, Article 5](https://www.uaelegislation.gov.ae/en/legislations/1506/download)

The disciplined course-project path is:

1. Choose one of the three top products and build the constrained evidence workflow using only the stated licensed data/models.
2. Record the exact narrow mechanism, baselines, and measurable improvement; do not claim ownership of a pretrained model or a public dataset.
3. Before demo disclosure, commission a UAE/IP professional to run a claims-focused prior-art search around that mechanism and advise whether to file before presenting it.

That keeps the project product-first, demonstrably deployable, and honest about both licence and patent risk.

## Source and licence ledger

| Asset | Owner/source | Availability checked | Terms to preserve |
|---|---|---|---|
| FloodNet | BinaLab | Official repository and maintained mirror | CDLA-Permissive-1.0 |
| Sentinel-1/2 | Copernicus/European Commission | Official collection/terms | Free, full, open; required attribution and terms apply |
| MARIDA | Authors/Zenodo | Project download instructions and documentation | CC BY 4.0 attribution |
| FineGrainOCR | Dataset authors | Official GitHub repository/download | CC0-1.0 |
| Supermarket Shelves Dataset | Humans in the Loop | Publisher dataset page | CC0-1.0 |
| TACO | Dataset authors | Official repository / licence file | CC BY 4.0 attribution |
| TrashNet | Dataset authors | Official repository | MIT |
| PlantVillage | PlantVillage/Mendeley | Dataset record | CC BY 3.0 attribution |
| FoodSeg103 | Dataset authors/Hugging Face distribution | Dataset card | Apache-2.0 |
| Global Mangrove Watch | JAXA/GMW partners | Official download and terms page | Free public use with credit; location/scale accuracy caveat |
| DocLayNet | IBM Research | Official repository/licence | CDLA-Permissive-1.0 |
| Prithvi-EO, SAM 2, Qwen2.5-VL, PaddleOCR | Respective model publishers | Model/repository cards | Apache-2.0; retain notices |

### Search stop note

I stopped after two focused passes because the main uncertainty was not finding more ideas; it was licensing. The candidates above have a direct publisher/repository/official-data path and a stated reusable term. Additional hits repeatedly led to research-only, non-commercial, challenge-only, or missing-licence data and were rejected.
