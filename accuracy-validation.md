# Qwen3.5/Qwen3.8 CP validation

**Validation:** The development implementation has completed GPU numerical checks, Qwen3.5/Qwen3.8 model accuracy evaluations and end-to-end serving tests. Results and coverage are summarized below.

## Accuracy Tests

### Model accuracy through the serving API

These are completed development runs through the service endpoint, including CP prefill, PD transfer and decode. They are fixed 200-example evaluations, not full-dataset scores.

| Model | Serving configuration | GSM8K | MMLU |
| --- | --- | --- | --- |
| Qwen3.5-397B-A17B | CP4 + Mega; external encoder and separate decode | 196/200 (98.0%) | — |
| Qwen3.5-397B-A17B | CP4 + Mega on 4× H800 prefill; concurrency 8, non-thinking | 196/200 (98.0%) | — |
| Qwen3.8 | CP2 + Mega, PD; concurrency 8, non-thinking | 192/200 (96.0%) | 177/200 (88.5%) |
| Qwen3.8 | Two CP2 + Mega prefill workers and one 4-GPU decode worker; concurrency 64, template-default thinking | **194/200 (97.0%)** | **176/200 (88.0%)** |

Protocol: GSM8K 5-shot; Qwen3.8 MMLU 0-shot, fixed 200 questions covering 49 subjects. Temperature 0, top-p 1 and seed 0 for the H800 Qwen3.5 evaluation and both Qwen3.8 evaluations; output limits were 4,096 tokens for the non-thinking runs and 16,384 for the thinking run. Score extracted final answers; request failures, missing final answers and truncations count as incorrect. No retries or selective replacement of failed answers.

All 400 requests in the concurrency-64 Qwen3.8 evaluation returned HTTP 200, and workers remained healthy afterward. The reported scores include six empty final answers and five truncated responses as failures. The non-thinking Qwen3.8 evaluation also had zero request errors; its truncated and invalid answers remain in the denominator. The two Qwen3.8 runs use different thinking/output-budget settings and are not a like-for-like accuracy comparison.

### Complete end-to-end serving and GPU checks

- **Qwen3.5 EPD:** six CP4 + Mega prefill groups, separate 8-GPU decode and an external encoder. All **15 functional cases** passed, including a **70,311-token retrieval request**, short generation, image requests and cached follow-up turns. Eight additional routing requests covered all six prefill groups; post-run queues drained. Concurrent shared-prefix checks also passed, with observed **50,048-token cache hits**.
- **Sustained serving:** the same Qwen3.5 development campaign completed **6,462/6,462 requests**, zero failed requests, over **30 minutes** at 3.59 requests/s (mean input 51,839 tokens). This exercises the full serving chain and is a reliability result, not a claim of CP speedup or maximum capacity. NEXTN was enabled on the separate decode service; this does not enable speculative execution inside the CP prefill path.
- **Qwen3.8 PD and prefix reuse:** the fused-QSA campaign completed end-to-end accuracy requests and cold/repeated/branched-prefix checks. A 19,027-token repeated prompt reused **19,008 cached tokens**; branching after token 8,192 reused **8,192 cached tokens**, with expected generated answers.
- **GPU regressions:** the fused-QSA run recorded **192 passed** GPU cases, plus a separate **16 passed** cache-store regression run (counts are separate runs, not asserted to be disjoint).
- **Mapped Mega/mHC combine:** **368/368** comparisons against explicit combine + reorder were elementwise identical (`max_abs = 0`, `max_relative_rms = 0`) on real model inputs. This validates that layout transformation; it is not a blanket equivalence claim for quantization conversion.

The results above cover the listed development configurations. Regression checks are summarized below.

### Regression checks

**604 passed tests** (104 QSA/CP + 128 disaggregation + 372 server-argument tests), plus 431 passing subtests. CUDA cases: 348 skipped, 13 deselected. Three MLX/xgrammar-dependent argument cases were excluded after reproducing failures on the unchanged base.

All 124 changed Python files compiled; runnable pre-commit checks passed. Community CI remains outstanding. Matching non-CP accuracy baselines, an isolated quantization-conversion comparison and an upstream-reproducible FlashInfer/NVSHMEM dependency matrix remain follow-up coverage.

## Speed Tests and Profiling

The 30-minute end-to-end load run above establishes completed serving coverage. No general CP speedup is claimed: matched weights, quantization, hardware and request workloads are required for that comparison. Follow-up profiling should quantify TTFT/throughput/memory/transfer cost and actual communication/computation overlap across CP sizes and cold/warm prefixes.

Optional machine-readable result summary: [accuracy-evidence.json](accuracy-evidence.json).
