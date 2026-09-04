# UAE computer-vision product directions

## Executive decision

**LabelProof UAE has been removed from the active shortlist.** Its data availability is strong, but a normal barcode/product-lookup app is a simpler substitute for its consumer workflow. An opaque “integrity score” would also be unjustified: a barcode cannot establish that a photographed package is current, authentic, safe or compliant.

The only potentially useful version would be a private supplier-to-retailer packaging-revision workflow. That depends on a partner's approved product master data, so it fails the public self-serve requirement.

## Other viable directions

| Direction | Public data and runtime input | Product/evidence report | Constraint to respect |
|---|---|---|---|
| **AccessLedger UAE** | [Google SANPO](https://github.com/google-research-datasets/sanpo_dataset) accessibility-navigation data (CC BY 4.0); venue owner uploads guided entrance/route photos. | `visible`, `owner-confirmed`, `not observed` features such as steps, ramps, obstructions or tactile cues; bilingual venue disclosure. | Strong social value and policy fit, but no UAE visual benchmark and it must never certify “wheelchair accessible” or code compliance. [DCT Shumool](https://dct.gov.ae/en/what.we.do/tourism/industry.initiatives.emiratisation/Shumool%20-%20People%20of%20Determination%20Training%20Programme%20.aspx) |
| **SolarProof UAE** | Public solar-condition benchmark data can support a proof of concept; Shams Dubai owner/contractor supplies runtime panel photos. | Visible dust, obstruction or possible crack evidence with an overlay and contractor-handoff draft. | Good Dubai relevance, but image data is not UAE-specific and visual output must not become a fault, electrical-safety or energy-yield diagnosis. [DEWA Shams Dubai](https://www.dewa.gov.ae/en/consumer/solar-community/shams-dubai/shams-dubai-faq) |
| **ChangeLens UAE** | Continuous, commercially reusable [Copernicus Sentinel](https://cds.climate.copernicus.eu/licences/ec-sentinel) data and [Dynamic World](https://www.dynamicworld.app/about/index.html); no runtime upload needed. | Dated map-scale evidence for land-cover changes: vegetation, water, unpaved land or built surface. | Best data continuity, but lower product differentiation and only 10 m resolution—never individual-property, permit or investment claims. |
| **ManeuverCoach UAE** | Public driving data for development; a normal dashcam clip is runtime input. | Evidence clip plus a versioned RTA practice card for a merge or roundabout event. | Strong product experience, but live coaching/driver scoring is already crowded. It remains a credible alternative, not the best new evidence-led direction. |

## Rejected directions

- **Date-palm diagnosis:** available UAE-specific images commonly prohibit commercial use; disease claims are also high-risk.
- **Property snagging/used-car damage:** existing UAE products already occupy the core workflow, and permissively licensed UAE visual training data was not verified.
- **Municipal litter/surveillance:** local collection/privacy burden is high and related municipal camera products already exist.
- **Health, halal, expiry or allergy scanner:** misleading certainty creates avoidable consumer-safety and regulatory risk.
- **LabelProof UAE:** current UAE catalogue data is not enough to overcome barcode lookup as the simpler consumer alternative; a differentiated B2B packaging-revision product would need a private partner catalogue.

## Deployment path

```text
Mobile/web camera -> local quality gate -> encrypted upload -> OCR/CV worker
-> catalogue retrieval -> evidence graph -> bilingual evidence card -> user deletion/export
```

Start with a browser PWA and one catalogue snapshot. Keep raw images only for the processing window; retain a user-selected report or no data at all. Add account history only after the core evidence card works.

No custom model training, vendor integration, social feed, retailer API, legal rules engine, or health recommendation is needed for the first release.

## Responsible launch conditions

- Get a clear camera/upload notice; minimise retention and protect image-derived personal data under the [UAE PDPL](https://www.uaelegislation.gov.ae/en/legislations/1972/download).
- Display source, date, crop and confidence for every claim.
- Make uncertainty visible rather than converting it into a guess.
- Keep a human/user review step before any real-world action.
- Use generative AI only to write from the evidence schema, consistent with the [UAE AI Charter](https://u.ae/en/about-the-uae/strategies-initiatives-and-awards/policies/Ai/The-UAE-Charter-for-the-Development-and-Use-of-Artificial-Intelligence).

## IP posture

Do not promise a patent for a “food label scanner.” If pursuing IP, investigate the narrow evidence-graph mechanism: acceptance only when independently observed package fields and a catalogue candidate agree, with source-crop provenance and a mandatory unresolved state. Confirm university ownership and perform a claims-focused prior-art search before public technical disclosure.

## Recommendation

Keep **ManeuverCoach UAE**, **AccessLedger UAE**, **SolarProof UAE**, and **ChangeLens UAE** as the active directions. None has yet earned selection: each must next be tested against an existing simpler substitute, a concrete user decision, and a defensible IP seam before being presented as the primary track.
