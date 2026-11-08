# original-reference backbone state

the single-view checkpoint saves trained heads but reconstructs the frozen pretrained backbone. upstream's training-mode shape probe changes its batchnorm buffers. that initialization can differ across runtimes even when the weight file, parameters and seed match.

on 9 november, the original a40 reconstruction replayed six released heads' saved normal scores and complete maps bit-exactly. its post-probe backbone fingerprint differed from the laptop reconstruction. restoring the a40 buffers on the laptop made all 24 heads pass the unchanged normal score/map guard on eight cases each. this is a reproducibility result; no detector was retrained or improved.

the complete laptop audit then failed: 19 of 24 heads passed across 906 normal predictions. all five failures occurred on the final single-image batch in `vial` or `wallplugs`; the maximum map error was 0.0008004. thirteen normal decisions changed, all on images whose saved score equalled the saved q95 threshold. the eight-case check therefore does not authorize laptop test scoring. the full original-device replay remains required; neither the tolerance nor the threshold changes to accommodate these differences.

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

## limits

the recovered a40 snapshot is a verified reconstruction, not a binary exported by the original training run. future runs should capture their actual initialized state. matching state does not guarantee identical kernels across devices; saved-normal replay remains necessary. legacy checkpoint bytes and the frozen paired study are unchanged. the full validation audit and any subsequent public scoring must retain their own source, input, snapshot and output identities.
