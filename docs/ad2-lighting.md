# ad 2 lighting groups

the definitions below come from sections iii-c/iii-e and table vi of the [versioned dataset paper](https://arxiv.org/html/2503.21622v1), checked on 2 november 2026. this is a mapping of documented lighting settings, not an inference from image appearance.

`regular`, `underexposed` and `overexposed` use the regular light sources, with exposure time changed for the latter two. additional `shift_1`/`shift_2`/`shift_3` settings are category-specific:

| category | regular / exposure changes | shift_1 | shift_2 | shift_3 |
|---|---|---|---|---|
| can | bar light | bar + spot 1 | bar + spot 2 | bar + spots 1/2 |
| fabric | flat light | flat + spot 1 | flat + spot 2 | flat + spots 1/2 |
| fruit_jelly | flat + spot 1 | flat + spots 1/2 | unavailable | unavailable |
| rice | flat light | flat + spot 1 | flat + spot 2 | flat + spots 1/2 |
| sheet_metal | ring light 1 | ring light 2 | ring light 2 | ring light 2 |
| vial | flat light | flat + spot 1 | flat + spot 2 | flat + spot 3 |
| wallplugs | flat light | flat + spot 1 | flat + spot 2 | flat + spots 1/2 |
| walnuts | flat light | flat + spot 1 | flat + spot 2 | flat + spots 1/2 |

for sheet_metal, the paper specifies three exposure times for each ring light. do not interpret its three shifted settings as three additional spotlights. `shift_1` is also not one physically identical intervention across all categories.

the actual vial archive also contains `shift_4` (five normal and fifteen defective public images). that group is not described in arxiv v1 table vi. retain it as the observed filename group `shift_4`, record its physical mapping as unverified, and do not silently merge it into another setting. the known exposure/shift names can still be inventoried without interpreting this extra lamp configuration.

the staged original public filenames use `<index>_<condition>.png`, matching these condition labels. verify the actual category/condition inventory with `scripts/build_public_conditions.py`; it reads file metadata only. the generated json is accepted by `scripts/evaluate.py --conditions`. its adjacent metadata records counts and the ordered image-id hash. neither step selects settings or decodes test pixels.

the paper states that public test data includes all lighting conditions. private regular and private mixed are separate sets; the mixed set contains seen and unseen lighting. do not label the whole mixed set as unseen. do not infer private labels or acquisition groups from filename patterns.
