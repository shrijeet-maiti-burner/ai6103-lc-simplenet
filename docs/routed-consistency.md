# adapter-only score consistency

`adapter_only_score_consistency` evaluates an aligned normal pair through the
current discriminator with detached classifier parameters and cloned buffers.
the consistency gradient reaches the adapter. classification continues to train
both modules. the auxiliary forward uses running normalization and restores all
previous module modes.

pass the complete adapted auxiliary batch, with original patches first and
relit patches second. additional patches retain the caller's forward shape.
`patches_per_view` is the number of patches in one normal view.

`local_global_losses.local_global_objective` supplies the associated feature-loss
calculation. it combines normal/global synthetic bce, local focal supervision
and normal-score consistency. it uses the original raw normality sign and one
live batch-normalization update; mining and auxiliary forwards use running
statistics. global offsets start from caller-supplied gaussian noise, undergo
20 gradient-ascent steps of size 0.001, and are projected every five steps to
the 0.5--1.0 annulus. local focal loss uses gamma 2 and the empirical median
of squared anomaly-probability error, retaining ties.

the matched feature-loss ablations are:

| method | `lambda_score` | `route_consistency` | paired inputs |
|---|---:|---|---|
| l0 | 0 | true | identity pair |
| l1 | 0 | true | relit pair |
| l2 | 0.1 | false | relit pair |
| r0 | 0.1 | true | relit pair |

the caller supplies frozen embeddings, aligned local synthetic embeddings,
noise, visible-support masks, optimizers and data order. support labels mark
changed synthetic pixels, not real defects. the objective does not include a
defect-lift loss. these apis do not yet provide the image-synthesis pipeline or
a training cli. existing training commands, defaults and inference are
unchanged. shared adapter updates can still change later defect responses;
gradient routing does not guarantee recall preservation.

the reported AD 2 and blade results were produced by the archived trainers in
the execution-source releases, such as the
[AD 2 source](https://github.com/shrijeet-maiti-burner/ai6103-lc-simplenet/releases/tag/routed-reproduction-source-2026-11-16),
not by this module. this module is a checked copy of that objective: on cpu,
its r0, l1 and l2 losses, gradients and batch-normalization updates match the
archived AD 2 code bit for bit.

the tests check forward-value and adapter-gradient equality, absent classifier
consistency gradients, a classification-equivalent optimizer update, one live
batch-normalization update, and state restoration after an exception. full-loss
tests additionally check mining isolation, offset bounds, the normality sign,
and routed versus full-gradient and zero-weight controls.

global ascent and local focal supervision are adapted from
[glass](https://github.com/cqylunlun/GLASS), commit
`f788d567db552820d744ca2d93e989265aad8c45` (qiyu chen et al., eccv 2024).
its [mit notice](../third_party/licenses/GLASS-MIT.txt) is retained. the sign
convention, normalization path and paired consistency treatment differ from
the author system. this is not a full glass reproduction or a performance
claim. comparisons must retain the original gaussian-hinge baseline and the
matched synthesis controls; a gain over that baseline cannot all be attributed
to routing.
