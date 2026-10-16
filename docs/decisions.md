# Decisions

Updated 16 October 2026 (Singapore).

| Item | State |
|---|---|
| Project source | Revised `AI6103_Project_Proposal_SimpleNet (1).pdf` |
| Core method | Brightness/gamma pair; normal patch-score MSE; original discrimination loss retained |
| Initial comparisons | B0 baseline, B1 relighting without consistency, B2 relighting with consistency |
| Dataset progression | Selected AD reproduction, then separate AD 2 training/evaluation |
| Repetitions | Three independent training seeds; proposed values 0, 1, 2 |
| Roles | Unassigned at the user's request; members choose their research direction |
| Code provenance | Official SimpleNet pinned; original submodule unchanged |
| Visibility/access | Private team repository; four requested write invitations |
| Professor approval / novelty judgement | Not found in supplied material |
| Current deadlines / team registration | Verify current course announcements; old slide dates are unsuitable |
| Categories, training budget, image sizes | Open until baseline profiling; no score-driven category selection |
| Transform ranges / score-loss weight | Open; normal training/validation evidence only |
| Validation FPR target / tolerated recall drop | Open; predeclare before final testing |
| AD split and AD 2 lighting metadata | Persist and verify before the experiment matrix |
| Private server access / submission fields | Not tested; confirm before promising private metrics |
| Main training integration / AD 2 loader / official PRO implementation | Not implemented in this scaffold |

Do not replace score MSE with feature cosine or score L1 because they appeared in an older chat answer. Such changes require an explicit experiment decision. No additional architectures or datasets are part of the initial commitment.
