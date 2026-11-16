# adapter-only score consistency

`adapter_only_score_consistency` evaluates an aligned normal pair through the
current discriminator with detached classifier parameters and cloned buffers.
the consistency gradient reaches the adapter. classification continues to train
both modules. the auxiliary forward uses running normalization and restores all
previous module modes.

pass the complete adapted auxiliary batch, with original patches first and
relit patches second. additional patches retain the caller's forward shape.
`patches_per_view` is the number of patches in one normal view.

this is an experimental training primitive. the existing training commands,
defaults and inference path are unchanged. it does not by itself implement the
local/global synthetic-defect recipe or establish a real-defect improvement.
shared adapter updates can still change defect responses in later steps.

the tests check forward-value and adapter-gradient equality, absent classifier
consistency gradients, a classification-equivalent optimizer update, one live
batch-normalization update, and state restoration after an exception.
