# Renamed BodyM Dataset

This folder is the clean working dataset for BodyFit. It replaces the hash-heavy raw layout with readable subject and capture names while preserving traceability to the original BodyM IDs.

Naming rule:

```text
raw_masks/sub_0001/cap_001_front.png
raw_masks/sub_0001/cap_001_side.png
```

- `sub_XXXX` is a stable anonymized subject index.
- `cap_XXX` is the repeated photo/capture index for that subject.
- `front` comes from the original `raw/mask/` folder.
- `side` comes from the original `raw/mask_left/` folder.

Primary files:

- `pairs_dimensions.csv`: main training/evaluation table for the dimension-first pipeline.
- `metadata/measurements_renamed.csv`: one row per subject with normalized measurement column names.
- `manifest.csv`: traceability only; one row per image file mapping each renamed file back to the original subject and photo IDs.
- `subject_key_map.csv`: traceability only; one row per subject mapping `sub_XXXX` back to the original subject ID.

Old pseudo-label CSVs are intentionally not part of the primary training table.
