# body2fit: venture and commercialisation review

Reviewed 2026-09-14 against the tests in `CONTEXT.md` and the shortlist in
`research/uae-cv-opportunities/uae-cv-product-directions.md`.

## Summary

Two blockers sit in front of any commercialisation of the current artifact, and
both are decidable today without market research.

1. The trained checkpoint cannot be sold. `BodyM` is CC BY-NC 4.0.
2. The product as specified is the category `uae-cv-product-directions.md`
   already rejected: a health scanner whose output is a risk determination.

Neither blocker is fatal to the underlying work. Both change what the venture
would have to be. The recommendation is at the end.

## Blocker 1: the training data forbids commercial use

`README.md` evaluates on "the subject-disjoint BodyM split
(`data/bodym/pairs_dimensions.csv`)", and that file's columns match BodyM's
schema. The AWS Registry of Open Data states BodyM's licence as
**Creative Commons Attribution-Non Commercial 4.0 International**. It does not
permit commercial use.

That reaches further than the CSV. `checkpoints/best_640x480_v4_resnet.pt` is a
derivative of BodyM. A venture selling access to that checkpoint, or to any
service running it, is selling a non-commercial derivative.

This is the same failure that removed date-palm diagnosis from the shortlist:
*"available UAE-specific images commonly prohibit commercial use."* The test was
applied there and it applies here.

The work is not lost. The architecture, the gate, the evaluation harness and the
capture protocol are all yours. Only the weights are encumbered. Retraining on
commercially licensed or self-collected data restores the asset, and that is a
funded-dataset problem rather than a research problem.

**Immediate action, unrelated to the venture question.** The weights are
currently public at `v1.0.0-weights` with no provenance notice. NC restricts
commercial use rather than redistribution, so publishing for research is
defensible, but the release should state that the weights derive from BodyM
under CC BY-NC 4.0 and are not licensed for commercial use. I published that
release, so I should fix it. Say the word and I will.

## Blocker 2: your own framework already rejected this category

`uae-cv-product-directions.md` rejects:

> **Health, halal, expiry or allergy scanner:** misleading certainty creates
> avoidable consumer-safety and regulatory risk.

And `CONTEXT.md` defines an evidence report as one that *"never makes a
compliance, legal, medical, safety, insurance, financial, or enforcement
determination."*

The demo currently renders "Elevated Central Risk" and "Increased Risk
(0.50-0.60)". Those are medical determinations. Under your own definition the
product does not currently emit an evidence report.

There is one honest counter-argument, and it is the reason this direction is
worth more than the scanner you rejected. Your responsible-launch conditions
require: *"Make uncertainty visible rather than converting it into a guess"* and
an *"explicit unavailable or unreadable state."* The SMPL-X render-back gate is
exactly that mechanism, implemented and working. The rejected scanner had no
such construct. body2fit is the one health direction that already satisfies the
uncertainty test rather than promising to.

So the category rejection holds for the *risk labels*, not for the *measurement
and abstention machinery*.

## The regulatory line runs through the risk labels, not the measurements

This is the load-bearing distinction for the whole venture question.

**Measurements are not a device.** FDA treats software confined to general
wellness claims as outside device regulation so long as it makes no
disease-specific diagnostic or treatment claim. A vision-based body composition
tool has been run under that general-wellness path explicitly.

**Risk categories probably are.** The UAE moved medical device oversight to the
Emirates Drug Establishment under Federal Decree-Law No. 38 of 2024, and UAE
classification rules align with the EU, not with FDA's lighter wellness
approach. Under EU MDR **Rule 11**, software providing information used to take
decisions with diagnostic purposes is **Class IIa**, which requires a Notified
Body. A tool that prints a WHO or NICE cardiometabolic risk band is squarely in
the intended-use language that triggers Rule 11.

The practical consequence: `waist 91.65 cm` is a measurement. `Elevated Central
Risk` is a regulated claim. The same pipeline produces both, and the second one
is roughly two years and a Notified Body away from being sellable in the UAE or
the EU.

**The pivot away from the paper accidentally improved this.** The IJCAI proposal
predicts BMI, BAI and BF%, supervised against `BAI/100` as a proxy. The current
repo predicts tape girths and nothing else. Girths are physical dimensions with
a non-medical buyer. BF% has only a health buyer. The change was made for
scientific-honesty reasons, and it happens to have moved the artifact to the
cheap side of the regulatory boundary.

## The non-medical path exists but is occupied

Girth-from-two-photos has an established commercial market in apparel fit, and
it is not an open field. 3DLOOK generates measurements and a 3D model from two
smartphone photos, ships a production API (FitXpress), and reports 80+ long-term
customers including large fashion and e-commerce brands. Bodygram occupies
adjacent ground.

Against your **non-substitutable problem test**, "predict waist and hip from two
photos for sizing" fails. A funded incumbent with a live API already delivers
that decision.

## Where a defensible seam actually is

Strip out what is commoditised and what is regulated, and one thing is left that
neither incumbents nor the rejected scanner have: **a measurement that refuses
itself on geometric grounds.**

3DLOOK's published positioning is accuracy and coverage. Nothing in the material
reviewed offers a verified refusal: a per-capture proof that the returned
numbers are consistent with a 3D body that actually reprojects onto the observed
silhouettes. Your gate rejects two of three demo presets on real geometry, with
named reasons (`side_foreground_too_large`, `band_circumference_mismatch`,
`geometry_score`). That is an evidence-report primitive, and it matches your
`CONTEXT.md` definition almost line for line: observed evidence, provenance,
confidence, and an explicit unreadable state.

Buyers who care about a refusal rather than an average are the ones where a
wrong measurement has a cost:

- **Made-to-measure and uniform programmes.** A garment cut to a wrong waist is
  scrap plus a remake. "We decline 12% of captures and ask for a retake" is a
  cost argument, not a marketing claim.
- **Clinical trial and research capture.** Non-contact anthropometry where
  protocol compliance matters and an auditable reject is a feature.
- **Insurance and occupational health intake**, where the measurement feeds a
  human decision and a silent bad reading is the liability.

None of these requires a risk label. All of them consume centimetres.

## The three paths, scored against your own tests

| | Fit-tech measurement API | Reliability layer for capture | Clinical screening (paper's framing) |
|---|---|---|---|
| Public self-serve MVP | yes | yes | no, needs clinical partner |
| Public-source-only MVP | **fails, BodyM is NC** | **fails, BodyM is NC** | fails |
| Non-substitutable | **fails, 3DLOOK exists** | plausible, unvalidated | yes but regulated |
| Evidence report, no medical determination | yes | yes | **no** |
| Regulatory load | wellness, low | wellness, low | **EU MDR Class IIa, Notified Body** |
| Time to first revenue | short | short | long |

Every path currently fails the data test. Two of three clear the regulatory and
evidence tests once the data is replaced.

## IP posture

Do not pursue a patent on "body measurement from photos." That is heavily
occupied and the claims would be narrow to the point of uselessness.

The narrow mechanism worth a prior-art search, consistent with the posture you
already took on LabelProof, is the one in `src/smpl/gate.py`: **acceptance of a
predicted measurement only when a parametric 3D body fitted to the capture
reprojects onto the observed silhouettes within stated per-band tolerances, with
a mandatory unresolved state and named failure reasons.** The claim is about
conditional refusal tied to geometric self-consistency, not about measurement.

Confirm BITS Pilani Dubai's ownership position before any public technical
disclosure. Note that the repository and the weights are already public, which
may already constitute disclosure. Check this with the university before
spending on a filing.

## Recommendation

Do not promote body2fit to the active shortlist yet. It fails the same
commercial-data test that removed date-palm diagnosis, and its headline output
is the medical determination your evidence-report definition forbids.

Do keep the gate. It is the only asset here that is both differentiated and
aligned with the doctrine you have already written down, and it is more useful
to the three active directions than it is to body2fit. `AccessLedger` needs
exactly this: a principled, non-negotiable "not observed" state rather than a
confidence number. Consider extracting the gate concept as a shared evidence
primitive across directions instead of treating it as a body-measurement
feature.

If body2fit itself is to continue as a venture, the order is:

1. Replace BodyM with a commercially licensed or self-collected measurement set.
   Until this is done nothing else matters.
2. Drop the risk labels from the product surface. Keep them in the paper.
3. Validate one buyer who will pay for a refusal, in made-to-measure or trial
   capture, before building anything further.
4. Only then revisit clinical framing, with a Notified Body budget and a
   partner.

## Sources

- [BodyM Dataset, Registry of Open Data on AWS](https://registry.opendata.aws/bodym/)
- [FDA clarifies when software wellness products are not medical devices](https://www.mddionline.com/software/fda-clarifies-when-software-wellness-products-are-not-medical-devices)
- [FDA issues updated guidance loosening regulatory approach to certain digital health tools](https://www.lw.com/en/insights/fda-issues-updated-guidance-loosening-regulatory-approach-to-certain-digital-health-tools)
- [Evaluation of the accuracy of a computer vision-based tool for assessment of total body fat percentage](https://cdn.clinicaltrials.gov/large-docs/21/NCT04854421/Prot_SAP_002.pdf)
- [UAE EDE medical device registration guide 2026](https://meddeviceguide.com/blog/uae-ede-medical-device-registration-guide-2026)
- [United Arab Emirates medical device regulations, Qserve](https://qservegroup.com/en/united-arab-emirates-medical-device-regulations)
- [MDR Rule 11 classification rules for SaMD](https://revolve.healthcare/blog/rule-11-mdr)
- [MDCG 2019-11 qualification and classification of software](https://openregulatory.com/mdcg/mdcg-2019-11)
- [3DLOOK body scanning technology for apparel](https://3dlook.ai/content-hub/body-scanning-technology-for-apparel/)
- [3DLOOK FitXpress API reference](https://docs.fitxpress.3dlook.me/)
