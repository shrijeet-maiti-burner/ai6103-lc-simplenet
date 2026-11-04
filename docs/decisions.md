# Decisions

Updated 4 November 2026 (Singapore).

| Item | State |
|---|---|
| Project source | Revised `AI6103_Project_Proposal_SimpleNet (1).pdf` |
| Core method | Brightness/gamma pair; normal patch-score MSE; original discrimination loss retained |
| Initial comparisons | B0 baseline, B1 relighting without consistency, B2 relighting with consistency |
| Dataset progression | Selected AD reproduction, then separate AD 2 training/evaluation |
| Repetitions | Seeds 0, 1 and 2 completed in the released AD reproduction and AD 2 paired study |
| Roles | Shrijeet A + E and overall review; Athul + Umesh B; Harsh C; Aryaman D |
| Code provenance | Official SimpleNet pinned; original submodule unchanged |
| Visibility/access | Private team repository; all four requested collaborators have write access as checked 26 October |
| Professor approval / novelty judgement | Proposal acceptance reported by Shrijeet. Later professor reply leaves the approach to the group; no novelty or grade assurance |
| Current deadline / submission | 23 November 2026, confirmed by Shrijeet on 4 November; replaces provisional 8/15 November targets. Current announcement reviewed: report 30 points, video 10 points, separate code submission; see [requirements](requirements.md). Exact cutoff and registered team still need portal verification |
| Categories, training budget, image sizes | [AD reproduction](ad-reproduction-results.md): 15 categories, 40 meta-epochs with four discriminator passes each, resize 329 / crop 288. [AD 2 paired study](ad2-study-results.md): eight categories, 256×256 full-frame resize, 80 source passes and three seeds. Two views give 160 input-view passes; this does not establish equal compute to a single-view run |
| AD baseline input | Upstream `run.sh`: resize 329, center crop 288 for images and masks (B decision, 26 October) |
| Transform ranges / score-loss weight | Brightness/gamma 0.8 to 1.2, confirmed by C on 30 October; AD score-loss weight 1.0, chosen on normal validation (27 October) |
| Validation FPR target / tolerated recall drop | 5% target FPR on regular-light normal validation, one frozen threshold per category, method and seed. Recall is preserved when macro recall is within 5 percentage points of the comparison model in each lighting condition; B1 against B0 is the main comparison, each B2 weight is compared with B0 and B1 (agreed on #8, applied in [the AD 2 evaluation](ad2-test-evaluation.md)) |
| AD split and AD 2 lighting metadata | Frozen manifests and public filename/label mapping retained. Vial shift_4 is observed but its physical lamp setup remains unverified |
| Private server access / submission fields | Not tested; confirm before promising private metrics |
| Data foundation | Normal manifest builder/reader and continuous-map restoration implemented; see [the contract](data-contract.md) |
| Development training integration | Paired pilot and full paired study implemented; 45-run AD fixed-final reproduction released. Faithful single-view AD 2 replay remains unresolved and must not be replaced with an unqualified reference result |
| AD 2 evaluation loader / official PRO implementation | Complete 120-head public image evaluation and exact native RTX pixel AUROC/AU-PRO released and audited; official private-server evaluation remains pending |

Do not replace score MSE with feature cosine or score L1 because they appeared in an older chat answer. Such changes require an explicit experiment decision. No additional architectures or datasets are part of the initial commitment.
