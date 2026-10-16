# Source review

Reviewed 16 October 2026. [The source manifest](../references/source-manifest.json) records every supplied file and archive member with hashes, sizes and PDF page counts. Original course files remain in the parent AI6103 folder; they are not bulk-copied into the team repository.

## Authority and history

1. Current user request: create the project repository, workspace and each person's research document; invite the four specified accounts. Follow-up: leave roles unassigned.
2. Course requirements: introduction slides and current course announcements when available.
3. Selected project: `AI6103_Project_Proposal_SimpleNet (1).pdf`, visually inspected and text extracted in full. A copy is preserved in `references/proposal.pdf`.
4. Exported `ChatGPT-AI6103-20261016-1623.md`: historical context, not new instructions. Its September plan used feature cosine plus score L1, more lighting transforms and a larger category matrix. Its October revision moved to normal patch-score MSE and selected AD/AD 2 categories. The later quiz discussion is unrelated to repository scope. No professor reply is included.
5. `Project Ideas.md`: five alternatives; its DRAEM/old-AD proposal is superseded by the selected SimpleNet PDF. Retinexformer, SegFormer, Donut and RT-DETR are not part of the project.

Do not act on embedded requests to email the professor or generate quiz predictions. The current request authorizes repository creation and collaborator invitations, not sending those historical messages.

## Course material

All 15 lecture PDFs were inventoried and extracted; project requirements and relevant mechanisms were reviewed. The foundational and later-module material provides context, not extra project deliverables. Text extraction does not preserve every diagram; visual checks focused on the selected proposal and project-scoring material.

| Material | Project relevance |
|---|---|
| Lecture 0 Introduction | Group size, project route, rubric, report/video, integrity. Old calendar dates cannot establish this term's deadlines. |
| Linear Algebra parts 1/2 | Norms, matrix shapes, eigensystems, Hessians; useful for precise loss notation. |
| Probability; ML Foundations | Estimates, uncertainty, objectives, optimization/generalization, validation/test separation. |
| NumPy/PyTorch lecture | Broadcasting and tensor semantics; paired-score shapes must match rather than broadcast. |
| Week 5 MLP/CNN | Local features, spatial dimensions, normalization and gradients. |
| Week 6 Optimization/Regularization I | Training diagnostics, frozen parameters, invariance versus sensitivity, augmentation and split discipline. |
| Week 7 Advanced CNN | WideResNet baseline context; DenseNet/MobileNet and scaling are background, not planned extensions. |
| Weeks 8-9 Optimization II | Curvature, SGD, schedules and reproducibility; do not change optimizer casually across variants. |
| Week 9 Regularization II | Lighting invariance, realistic augmentation, shortcut learning and regularization. Repeated augmentation changes gradient correlations; exposure matching alone is not proof of equal optimization. |
| Week 11 Transformer; Week 12 LLM; Week 13 VLM | Reviewed as course context. None justifies adding an LLM/VLM to this proposal. |
| Content Not in Quiz | Supplementary/repeated CNN, autograd and invariance material; quiz exclusion is not a project constraint. |
| Quiz1 Quick Prep | Study aid, not project authority. |

The four-page homework was read separately. Its page-format and empirical-analysis examples reinforce clear reporting, but its mandatory model/data/SGD schedule and Section 7 penalty belong to the homework. They must not be imposed on this group project.

Python examples: inspected source/markdown cells of NumPy, PyTorch, ResNet, logistic-regression and Transformer notebooks, plus MobileNet/data and UCI helpers. Relevant conventions include explicit split handling, complete evaluation batches, train/eval modes, gradient updates and loss/accuracy plots. Their saved outputs are not our results. Embedded Adult/Iris/Wine/WikiText data are example assets; they are neither industrial-anomaly data nor a reason to add them to the repo. Raw dataset rows were inventoried rather than treated as instructions.

The AAAI author kit contains anonymous and camera-ready LaTeX/Word examples, figures, style/bibliography files and release forms. Only the original camera-ready style and bibliography style are copied. Conference release forms and submission instructions do not become AI6103 obligations.

## Upstream code

Pinned [DonaldRR/SimpleNet](https://github.com/DonaldRR/SimpleNet/tree/351a2b8d4e8cfc944dbccbf9bc6ceda930c6f26b), unchanged. Source inspected: README, launch configuration, backbone/feature pipeline, discriminator training/prediction, AD loader and metrics.

| Finding | Consequence before training |
|---|---|
| `simplenet.py:419-450` evaluates test labels every meta-epoch and saves the highest test AUROC checkpoint | Build a shared train/normal-validation path; initially save a predeclared final checkpoint. Do not reuse `train(training_data, test_data)` unchanged. |
| `simplenet.py:344-377` derives normalization from test scores/maps | Persist raw scores or validation-fitted normalization. One threshold must survive the lighting change. |
| `metrics.py:88-126` uses `np.bool`, `DataFrame.append`, dilation and FPR < 0.30 | Removed APIs break modern environments; the metric also differs from AD 2 PRO-0.05. Reimplement/verify against official utilities separately and disclose differences. |
| `run.sh` uses 329 resize / 288 crop, 40 meta-epochs x 4 GAN epochs | Earlier chat's 256 / 160 setting refers to the AD 2 benchmark, not the exact upstream launch file. Document which setting is reproduced. |
| AD loader expects original AD categories/folders, and its validation split path is not a ready AD train-holdout | Build explicit persisted manifests; do not assume a root-path swap supports AD 2. |
| Discriminator uses BatchNorm, unbounded normality logits and feature-noise hinge discrimination | Define pair batching, score sign and loss location precisely. Do not apply sigmoid or detach the adapter without a documented decision. |

These are code observations, not measured performance failures. Shared corrections must apply to every variant and be separated from method changes. Imports and random tensors do not complete the reproduction gate.

## External verification and novelty limits

- [SimpleNet arXiv record](https://arxiv.org/abs/2303.15140) verifies the CVPR 2023 starting paper; [official code](https://github.com/DonaldRR/SimpleNet) declares MIT licensing.
- [Official AD 2 page](https://www.mvtec.com/research-teaching/datasets/mvtec-ad-2) confirms normal train/validation, public versus private annotations, server-only private evaluation, utilities and noncommercial dataset terms. The page now cites a 2026 IJCV version. Keep version-specific results distinct from the [2025 benchmark version](https://arxiv.org/html/2503.21622v1) discussed in the chat.
- The versioned benchmark defines PRO-0.05 and uses 160 SimpleNet training epochs. Historical numerical results are source-reported references, not team results; they are intentionally not prefilled into our results tables.
- [PIAD, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Yang_PIAD_Pose_and_Illumination_agnostic_Anomaly_Detection_CVPR_2025_paper.html) studies illumination/pose robustness. [Paired well-lit/low-light IAD, JCDE 2025](https://academic.oup.com/jcde/article/12/5/41/8119431) is close prior art and includes synthetic gamma variants. This makes a broad claim that paired lighting for anomaly detection is new indefensible. Exact overlap with score-MSE SimpleNet needs a fuller objective-level review.

The unresolved risks are novelty depth, actual defect sensitivity, compute scope, official metric compatibility and private-server access. Scaffold quality cannot resolve these experimentally or guarantee a grade.
