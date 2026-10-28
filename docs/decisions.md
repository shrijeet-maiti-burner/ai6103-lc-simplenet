# Decisions

Updated 27 October 2026 (Singapore).

| Item | State |
|---|---|
| Project source | Revised `AI6103_Project_Proposal_SimpleNet (1).pdf` |
| Core method | Brightness/gamma pair; normal patch-score MSE; original discrimination loss retained |
| Initial comparisons | B0 baseline, B1 relighting without consistency, B2 relighting with consistency |
| Dataset progression | Selected AD reproduction, then separate AD 2 training/evaluation |
| Repetitions | Three independent training seeds; proposed values 0, 1, 2 |
| Roles | Shrijeet A + E and overall review; Athul + Umesh B; Harsh C; Aryaman D |
| Code provenance | Official SimpleNet pinned; original submodule unchanged |
| Visibility/access | Private team repository; all four requested collaborators have write access as checked 26 October |
| Professor approval / novelty judgement | Not found in supplied material |
| Current deadlines / team registration | Verify current course announcements; old slide dates are unsuitable |
| Categories, training budget, image sizes | AD study: bottle and grid, 80 source passes, resize 329 / crop 288, chosen from measured cost (B provisional decision, 27 October; see [plan](ad-study-plan.md)). AD 2 paper baseline is 256×256 full-frame resize and 160 SimpleNet epochs (arXiv:2503.21622v1); not adopted in shared trainers pending team agreement |
| AD baseline input | Upstream `run.sh`: resize 329, center crop 288 for images and masks (B decision, 26 October) |
| Transform ranges / score-loss weight | Brightness/gamma 0.8 to 1.2 until C confirms; AD score-loss weight 1.0, chosen on normal validation (27 October) |
| Validation FPR target / tolerated recall drop | Open; predeclare before final testing |
| AD split and AD 2 lighting metadata | Persist and verify before the experiment matrix |
| Private server access / submission fields | Not tested; confirm before promising private metrics |
| Data foundation | Normal manifest builder/reader and continuous-map restoration implemented; see [the contract](data-contract.md) |
| Development training integration | [Fixed-step paired pilot](pilot-training.md) implemented; full upstream reproduction remains pending |
| AD 2 evaluation loader / official PRO implementation | Pending; normal-only calibration does not replace defect evaluation |

Do not replace score MSE with feature cosine or score L1 because they appeared in an older chat answer. Such changes require an explicit experiment decision. No additional architectures or datasets are part of the initial commitment.
