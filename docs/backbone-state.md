# original-reference backbone state

the single-view checkpoint saves trained heads but reconstructs the frozen pretrained backbone. upstream's training-mode shape probe changes its batchnorm buffers. that initialization can differ across runtimes even when the weight file, parameters and seed match.

on 9 november, the complete original a40 reconstruction replayed all 24 heads, 906 saved normal predictions, complete maps and threshold decisions bit-exactly. an independent comparison checked every returned score/map against the original npz files, transport hashes and archive crc. the reference is replayable on its original device. this is a reproducibility result; no detector was retrained or improved.

restoring those buffers on the laptop passes the eight-case check for every head, but the complete audit passes only 19 of 24. all five failures occur on the final single-image batch in `vial` or `wallplugs`; maximum map error is 0.0008004. thirteen decisions change, all at saved q95 ties. partial replay is insufficient, and the laptop remains excluded from original-reference test scoring. neither tolerance nor threshold changes to accommodate these differences. recovering this reference does not create a matched comparison with the paired-study models.

| retained private record | sha256 |
|---|---|
| complete normal input manifest | `abf6e7c65772c9088531afb01cd8b2b0546d3983000a1e0a2a355248d1f03e3d` |
| a40 runtime receipt | `b421eea3d6d59efd7c8bf309250da0561297279fefb15fd1985b7da1a5900b46` |
| a40 completion receipt | `928be0578e6adf4bfcef99a6f83e9612914ab17a07a7a082bab8f8adeec0df28` |

the remote audit used the verified source archive at `a05eb9e41fca0106abf72aae0ff632248c1e0c42`, plus the inference helper from `c8c23ab` (file sha256 `5e742d18cc40c5dad94d6450a1a7f2330bf22c8ee59f57518f798d7128ba8510`). the exact cached computation-library versions matched the original runtime. the unused metrics module was imported lazily, and pinned pretrained weights were loaded from verified memory bytes; neither changes the prediction path. data and binary outputs remain outside git.

## capture and restore

capture after the original probe, on the original device and runtime:

```powershell
python scripts/replay_single_view.py capture --run <released-run> --out results/backbone-capture-001
```

then replay the complete saved normal population on the scoring machine:

```powershell
python scripts/replay_single_view.py replay --run <released-run> --snapshot <backbone-state.pt> --manifest <normal-manifest.jsonl> --manifest-sha256 <manifest-sha256> --data <dataset-root> --out results/normal-replay-001
```

the manifest must produce the same ordered validation identities. preprocessing uses the original cpu transform and batch size. replay compares image scores and every smoothed map value with `np.allclose(atol=1e-4, rtol=1e-4)`. it records any normal decisions that differ at the saved q95 threshold. a failure prevents test scoring; the tolerance is not a setting to tune.

[`capture_backbone_state`](../src/lc_simplenet/backbone_state.py) stores independent cpu buffers, the probe shape, pretrained-file identity, parameter fingerprint and complete-state fingerprint. restore validates every field before changing any buffer. it leaves parameters and forward modes alone. the binary snapshot stays outside git.

[`predict_reference_batch`](../src/lc_simplenet/reference_inference.py) omits only an unused dense feature visualization. the score/map computation stays the same; a cpu architecture test checks complete-array equality with upstream `_predict`. this memory adaptation is explicit, and the upstream submodule is unchanged.

## score the frozen public cohort

freeze every category/seed and complete the full normal matrix before opening public inputs. the original-reference public audit uses the stricter original-device requirement of bit-exact scores/maps and unchanged decisions. the numeric-tolerance replay above remains useful for diagnosing portability; a numeric-only pass does not qualify this scoring command.

```powershell
python scripts/score_single_view_public.py --run <released-run> --snapshot <backbone-state.pt> --manifest <normal-manifest.jsonl> --manifest-sha256 <manifest-sha256> --data <normal-dataset-root> --public-root <public-dataset-root> --out results/original-public-001
```

the command repeats the complete normal replay before listing any public input and rechecks snapshot, head and normal-file hashes afterward. it preserves cpu preprocessing, saved batch size, raw max-patch image scores and smoothed maps. outputs include native-map geometry, input/checkpoint hashes, conditions and the original normal calibration scores. the annotated public subset is already exposed; these outputs cannot select a method or become an untouched evaluation.

## limits

the recovered a40 snapshot is a verified reconstruction, not a binary exported by the original training run. future runs should capture their actual initialized state. matching state does not guarantee identical kernels across devices; saved-normal replay remains necessary. legacy checkpoint bytes and the frozen paired study are unchanged. the full validation audit and any subsequent public scoring must retain their own source, input, snapshot and output identities.
