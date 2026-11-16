# frozen routed-simplenet reproduction

this launcher exposes the implementation used by the ongoing original ad2
comparison: b0, b1, b2, l0, l1, l2 and r0, three seeds and eight categories.
it prepares normal inputs, trains/resumes a fixed case and scores public images
with frozen normal-calibrated thresholds. it does not select a successful method.

the execution archive contains the frozen trainer, objectives, upstream source,
licences and public input fingerprints. the training class and objectives are
copied unchanged. the synthesis function and optimizer values are retained;
unused historical study plans and entry points are omitted. no dataset images,
model weights, credentials or private experiment results are included.

## setup and inputs

until [pr39](https://github.com/shrijeet-maiti-burner/ai6103-lc-simplenet/pull/39)
is merged, check out its branch with `gh pr checkout 39`, or use the tagged
source-code archive from the prerelease. the earlier main checkout has no launcher.

use the [locked environment](setup.md): python 3.12, torch 2.6.0,
torchvision 0.21.0, numpy 1.26.4 and pillow 11.3.0. full training uses the
original 256-pixel geometry, batch eight, 80 paired source passes and fixed-final
checkpoints. the experiment runs on an l40s with 48 gb; smaller hardware has not
been validated for the complete matrix. the mechanics check uses one fit image
and two normal validation images and carries a separate protocol identity.

download the execution source from the
[source prerelease](https://github.com/shrijeet-maiti-burner/ai6103-lc-simplenet/releases/tag/routed-reproduction-source-2026-11-16):

```powershell
uv sync --frozen --extra cuda
gh release download routed-reproduction-source-2026-11-16 --repo shrijeet-maiti-burner/ai6103-lc-simplenet --pattern routed-ad2-execution-source-20261116.zip --dir results/routed-source
uv run --frozen --extra cuda python scripts/reproduce_routed_ad2.py --source-archive results/routed-source/routed-ad2-execution-source-20261116.zip verify
```

archive sha256:
`74f2c9adb729e432237a6879ae82373a3dcee032e5517e7daabb515b5096cb8a`.
the launcher verifies every extracted file. extraction is temporary and leaves
the checkout and its simplenet submodule untouched.

acquire the original [mvtec ad2](https://www.mvtec.com/research-teaching/datasets/mvtec-ad-2)
normal/public images and [oxford dtd](https://www.robots.ox.ac.uk/~vgg/data/dtd/)
separately, retaining their licences. `--textures-root` points to the directory
containing `images/`. the fixed 256-donor selection is in the source archive;
input bytes, resized pixels and order must match the recorded fingerprints.
the normal dataset layout is described in [setup](setup.md).
torchvision obtains the pinned imagenet-v1 wrn50 weights through its normal
cache; the complete weight hash is checked before training.

```powershell
uv run --frozen --extra cuda python scripts/reproduce_routed_ad2.py --source-archive results/routed-source/routed-ad2-execution-source-20261116.zip prepare --data-root data/mvtec_ad2 --textures-root data/dtd --category sheet_metal --out results/routed-inputs
```

preparation reads only the pinned normal train/validation paths and dtd donors.
it never opens test images. staging is transactional: failed preparation does
not publish incomplete input files. subsequent categories can use the same
output directory; an existing identical texture archive is retained.

## mechanics check and full training

```powershell
uv run --frozen --extra cuda python scripts/reproduce_routed_ad2.py --source-archive results/routed-source/routed-ad2-execution-source-20261116.zip train --inputs results/routed-inputs --category sheet_metal --seed 0 --mechanics-only --stop-after 2 --out results/routed-mechanics
```

this executes the seven real objectives and checkpoint machinery on a small
normal-only fixture. it is not a trained detector result. scoring refuses
mechanics-only checkpoints. `--stop-after` also permits a short partial
production run; partial checkpoints cannot be scored as final models.

for a full case:

```powershell
uv run --frozen --extra cuda python scripts/reproduce_routed_ad2.py --source-archive results/routed-source/routed-ad2-execution-source-20261116.zip train --inputs results/routed-inputs --category sheet_metal --seed 0 --out results/routed-runs/sheet_metal/seed0
```

repeat preparation/training for all eight named categories and seeds 0, 1, 2
to obtain the 168-head matrix. there is no epoch search. `run.json` records the
fixed inputs and scope; each attempt has a separate loss log. snapshots retain
all heads, optimizer/sampler/rng states and a next-input feature fingerprint.
`latest.json` identifies the newest saved checkpoint. resume with the same
inputs, output, category and seed, adding for example:

```text
--resume results/routed-runs/sheet_metal/seed0/checkpoint-00256.pt
```

the original exact replay target is the locked l40s runtime. another gpu or
library combination may differ numerically; a failed replay check must not be
silently bypassed or fixed by recalibrating on targets.

local verification regenerated all 156 sheet-metal normal inputs and all 256
texture donors exactly. on an rtx 3060, all seven objectives ran and a resumed
two-update normal-only fixture matched an uninterrupted run in every saved
checkpoint field, including optimizer/rng states and the next-input fingerprint.
this is a mechanics check. replaying a completed l40s can checkpoint on the rtx
3060 differed by up to `1.69e-5` in normal calibration patch scores; it was not
bit exact. the strict scoring guard is retained. use the original runtime/gpu
for its recorded calibration, or reproduce a separately calibrated training run
on the new hardware. no genuine target was opened for these checks.

## fixed-threshold public scoring

```powershell
uv run --frozen --extra cuda python scripts/reproduce_routed_ad2.py --source-archive results/routed-source/routed-ad2-execution-source-20261116.zip score --inputs results/routed-inputs --run results/routed-runs/sheet_metal/seed0 --data-root data/evaluation-ad2 --out results/routed-scores/sheet_metal/seed0
```

scoring first cold-replays the complete saved normal calibration bit for bit.
only then does it read that category's `test_public/good` and `test_public/bad`
images. it saves raw patch arrays, image maxima, fixed thresholds and per-method
condition metrics. the rule is maximum negative raw discriminator patch score,
higher empirical q95 on regular normal validation, then strict `score > threshold`.
no model, threshold, epoch, category or seed is selected using those results.

this exports individual-case evidence. the original all8 claim still requires
the complete independent matrix audit, both the newly matched and previous
frozen paired b0 anchors, and the prespecified recall/false-alarm limits.
seven categories without sheet_metal are a sensitivity analysis, not a
replacement primary scope. exposed public results are not private-server results.

this launcher covers the ongoing original-ad2 transfer. it does not reproduce
the different aebad fit-subset budget or the earlier robustad experiment. the
original source screen and full procedural confirmation did not qualify; the
transfer remains explicitly exploratory. no new detection gain is claimed by
this code handoff. full-matrix scoring is still pending.
