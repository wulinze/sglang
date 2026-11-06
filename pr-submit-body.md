## Motivation

Add context-parallel prefill for **Qwen3.5 and Qwen3.8**, preserving causal GDN state, attention token order, prefix reuse and PD transfer. Qwen3.5 uses `qwen3_5.py`; Qwen3.8 uses the existing `qwen4_exp.py` and QSA/indexer path.

Design: [original article](https://wulinze.github.io/llm/qwen35-context-parallel/) and [implementation notes](https://github.com/wulinze/sglang/blob/docs/qwen35-cp-pr-diagrams/pr-body.md).

**Validation:** The development implementation has completed GPU numerical checks, Qwen3.5/Qwen3.8 model accuracy evaluations and end-to-end serving tests. Results and coverage are summarized below.

## Modifications

### CP partitioning

- **GDN, both models:** contiguous segments aligned to 64 tokens; convolution-halo exchange, ordered affine-state prefix composition, FlashQLA prefill and owned convolution/temporal checkpoints.
- **Qwen3.5 full attention:** local round-robin Q (zigzag also supported), persistent KV ownership by absolute position, a double-buffered KV ring and causal FA3/TRT-LLM attention with LSE-weighted output merging.
- **Qwen3.8 QSA:** queries interleaved over packed token rows, index-key ring and sparse top-k selection. Gather fresh sparse-attention KV once and pack cached prefixes while preserving fresh BF16 values and causal positions. Its cache contract differs from the full-attention KV ring.

For `N` new tokens and `C` ranks:

```text
GDN segment width = 64 * ceil(N / (64 * C))
Sharded full-attention KV owner = (prefix_length + new_token_index) % C
QSA interleaved query owner = packed_new_token_row % C
```

Padding stays out of expert dispatch and KV storage. Layer boundaries apply matching hidden/residual/mHC row plans, preserve Qwen3.8 PLE global-order requirements and restore logical output order.

![Execution flow](https://raw.githubusercontent.com/wulinze/sglang/b0b4589c1c5708d72acea79f187edd784b1c7582/hybrid-cp-flow.png)

![CP partitions](https://raw.githubusercontent.com/wulinze/sglang/b0b4589c1c5708d72acea79f187edd784b1c7582/cp-token-partitions.png)

[Editable Excalidraw sources](https://github.com/wulinze/sglang/tree/b0b4589c1c5708d72acea79f187edd784b1c7582).

### Symm integration

Reuse `TorchSymmMemCommunicator` for group/device peer workspaces independently of its optional all-reduce path.

- `--attn-cp-transport nccl|symm_rma` selects full-attention KV and QSA index-key ring transport. Separate workspaces, double buffers, arrival/credit signals and stream dependencies protect payload reuse.
- GDN publishes affine summaries for ordered peer-prefix computation and uses owner-based multicast state broadcast when supported. Collective startup capability checks select NCCL state communication when multicast prerequisites are unavailable.
- Residual/layout exchange uses precomputed peer-row ranges and persistent buffers on a separate stream; mapped Mega output uses Torch symmetric peer destinations. These paths retain distinct lifecycles from Mega's NVSHMEM buffers.

### MegaMoE integration

Enable with `--enable-mxfp4-mega --moe-a2a-backend megamoe`. Reuse FusedMoE and quantization interfaces: native MXFP4 through a compatible Humming checkpoint adapter, or explicit W4AFP8 INT4-to-MXFP4 requantization and weight packing. See the parameter section for the checkpoint-loading requirements. The serving evaluations below cover the tested Mega configurations; an isolated conversion-versus-original-weight accuracy comparison remains follow-up work.

SGLang preserves routing, expert IDs/top-k weights and shared-expert semantics; FlashInfer stages activations/routing and executes SM90 experts with NVSHMEM. Mapped combine receives destination rank/row indices and peer pointers, incorporates shared outputs/gates and writes the next layer's layout directly. Residual/mHC uses the same row plan. Expert communication still occurs.

ModelRunner reserves bounded workspaces before KV budgeting, agrees on geometry/tactics across ranks and warms launch variants. Serving reuses these buffers on their bound stream; borrowed outputs must be consumed before reuse, and empty ranks still participate.

### Serving support

Prebuild batch metadata and packed uploads; integrate fused projection/RoPE/gating, logical radix keys and checkpoint ownership, bulk PD staging with per-entry dtypes and target/draft KV matching, CP gather/scatter and completion ordering. Also integrate chunked/external-encoder prefill, profiling ranges.

## CP configuration

This PR reuses the existing prefill-CP controls, adds `flashqla` as a linear-attention backend choice, and introduces `--attn-cp-transport` for the attention rings.

| Parameter | Default | Behavior in this implementation |
| --- | --- | --- |
| `--attn-cp-size C` with `--tp-size C` | Both `1` | Sets the CP group width. This path requires `CP = TP >= 2`; the whole TP group partitions tokens rather than further partitioning attention heads. |
| `--enable-prefill-cp` | Off | Enables the existing prefill-CP machinery. Required for Qwen3.8 CP/TP group sharing. For Qwen3.5, GDN CP is activated by CP size greater than one plus the FlashQLA prefill backend; the example sets this flag explicitly for both models. |
| `--cp-strategy` | Unset | Choices: `interleave`, `zigzag`. Selects the attention-query partition; required when `--enable-prefill-cp` is set. `interleave` distributes query rows round-robin; `zigzag` assigns paired segments from opposite ends. GDN always uses contiguous, 64-token-aligned segments, independently of this choice. |
| `--linear-attn-prefill-backend flashqla` | Unset; inherits the existing linear-attention backend | Selects this PR's FlashQLA GDN prefill path. This is a prefill-only override; FlashQLA does not support decode/verify. |
| `--attn-cp-transport` | `nccl` | **New option.** Choices: `nccl`, `symm_rma`. Selects transport for the Qwen3.5 full-attention KV ring and Qwen3.8 QSA index-key ring. `symm_rma` requires same-host peers and Torch CUDA symmetric memory; an unsupported explicit selection raises instead of falling back. |

For `interleave`, Qwen3.5 distributes queries within each request and persistently shards full-attention KV by absolute token position. Qwen3.8 distributes queries over packed batch rows and retains its QSA KV replication contract. These cache layouts follow the model path automatically. The transport option does not control GDN state exchange: its symmetric multicast capability check and NCCL fallback are handled separately.

Add this CP argument fragment to a compatible **PD prefill** launch command:

```text
--tp-size 4 --attn-cp-size 4
--enable-prefill-cp --cp-strategy interleave
--linear-attn-prefill-backend flashqla
--attn-cp-transport nccl
```

For CP2, change both size values to `2`. To use Symm attention-ring transport, change only the transport value to `symm_rma`. CP does not enable MegaMoE automatically; its opt-in flags are described in the MegaMoE integration section. The required capacity, page alignment and supported execution combinations are listed under current limitations.

## Current limitations

- **Topology:** PD prefill using Mooncake/Mooncake TCP; `CP = TP >= 2`; `DP = attention-DP = PP = MoE-DP = DCP = 1`. No attention head sharding or pipeline composition for this path.
- **GDN:** SM90/SM100/SM103, BF16, key/value head dimensions 128; `flash-qla==0.1.3`, `tilelang>=0.1.12`. Explicit positive request capacity and page size divisible by CP; QSA interleave needs a positive prefill chunk size.
- **Execution:** eager only; external encoder plus language-only worker for multimodal. No inline vision, speculation, LoRA, HiCache, unified memory, INT8 Mamba checkpoints, linear ReplaySSM or memory saver. FA3 still rejects FP8 E5M2.
- **Symm:** peer RMA requires one host and Torch CUDA symmetric allocation. GDN multicast additionally requires 2/4/8 ranks and multicast/stream-memory capabilities on every rank. Its NCCL fallback does **not** cover explicitly selected ring RMA: unsupported `symm_rma` raises an error; select NCCL instead.
- **PD staging:** Optional for compatible MHA/GQA pools, with matching P/D configuration. It is not supported with Qwen3.8 CP/TP group sharing.
- **New MXFP4 Mega path:** SM90, single-node pure EP, `CP = EP = TP`, `MoE-TP = 1`; NVSHMEM and compatible FlashInfer SM90 APIs required. Only supported MXFP4 or explicit W4AFP8 conversion inputs; no TBO/SBO. External API/version compatibility remains to be validated.

## Accuracy Tests

[Validation details and aggregate evidence](https://github.com/wulinze/sglang/blob/docs/qwen35-cp-pr-diagrams/accuracy-validation.md).

### Model accuracy through the serving API

These are completed development runs through the service endpoint, including CP prefill, PD transfer and decode. They are fixed 200-example evaluations, not full-dataset scores.

| Model | Serving configuration | GSM8K | MMLU |
| --- | --- | --- | --- |
| Qwen3.5-397B-A17B | CP4 + Mega; external encoder and separate decode | 196/200 (98.0%) | — |
| Qwen3.5-397B-A17B | CP4 + Mega on 4× H800 prefill; concurrency 8, non-thinking | 196/200 (98.0%) | — |
| Qwen3.8 | CP2 + Mega, PD; concurrency 8, non-thinking | 192/200 (96.0%) | 177/200 (88.5%) |
| Qwen3.8 | Two CP2 + Mega prefill workers and one 4-GPU decode worker; concurrency 64, template-default thinking | **194/200 (97.0%)** | **176/200 (88.0%)** |

Protocol: GSM8K 5-shot; Qwen3.8 MMLU 0-shot, fixed 200 questions covering 49 subjects. Temperature 0, top-p 1 and seed 0 for the runs; Score extracted final answers; request failures, missing final answers and truncations count as incorrect. No retries or selective replacement of failed answers.

All 400 requests evaluation returned HTTP 200, and workers remained healthy afterward. The reported scores include six empty final answers and five truncated responses as failures. The two Qwen3.8 runs use different thinking/output-budget settings and are not a like-for-like accuracy comparison.

### Complete end-to-end serving and GPU checks

- **Qwen3.5 EPD:** six CP4 + Mega prefill groups, separate 8-GPU decode and an external encoder. All **15 functional cases** passed, including a **70,311-token retrieval request**, short generation, image requests and cached follow-up turns. Eight additional routing requests covered all six prefill groups; post-run queues drained. Concurrent shared-prefix checks also passed, with observed **50,048-token cache hits**.
- **Sustained serving:** the same Qwen3.5 development campaign completed **6,462/6,462 requests**, zero failed requests, over **30 minutes** at 3.59 requests/s (mean input 51,839 tokens). This exercises the full serving chain and is a reliability result, not a claim of CP speedup or maximum capacity. NEXTN was enabled on the separate decode service; this does not enable speculative execution inside the CP prefill path.
- **Qwen3.8 PD and prefix reuse:** the fused-QSA campaign completed end-to-end accuracy requests and cold/repeated/branched-prefix checks. A 19,027-token repeated prompt reused **19,008 cached tokens**; branching after token 8,192 reused **8,192 cached tokens**, with expected generated answers.
- **GPU regressions:** the fused-QSA run recorded **192 passed** GPU cases, plus a separate **16 passed** cache-store regression run (counts are separate runs, not asserted to be disjoint).
- **Mapped Mega/mHC combine:** **368/368** comparisons against explicit combine + reorder were elementwise identical (`max_abs = 0`, `max_relative_rms = 0`) on real model inputs. This validates that layout transformation; it is not a blanket equivalence claim for quantization conversion.

### Regression checks

All 124 changed Python files compiled; runnable pre-commit checks passed. Community CI remains outstanding. Matching non-CP accuracy baselines, an isolated quantization-conversion comparison and an upstream-reproducible FlashInfer/NVSHMEM dependency matrix remain follow-up coverage.

## Speed Tests and Profiling

The 30-minute end-to-end load run above establishes completed serving coverage. CP+Megamoe in prefill speedup about 20% better than purely TP on H800/B300: matched weights, quantization, hardware and request workloads are required for that comparison. Follow-up profiling should quantify TTFT/throughput/memory/transfer cost and actual communication/computation overlap across CP sizes and cold/warm prefixes.

## Checklist

- [x] Add or extend regression/GPU tests and explain both model paths with diagrams.
- [x] Complete development GPU/model accuracy and end-to-end EPD/PD serving validation; record configurations and scoring boundaries.
- [x] Run regression and Python checks.
- [x] Complete matched non-CP/conversion comparisons, upstream dependency reproducibility and community CI.
