## Motivation

Add context-parallel prefill for **Qwen3.5 and Qwen3.8**, preserving causal GDN state, attention token order, prefix reuse and PD transfer. Qwen3.5 uses `qwen3_5.py`; Qwen3.8 uses the existing `qwen4_exp.py` and QSA/indexer path.

Design: [original article](https://wulinze.github.io/llm/qwen35-context-parallel/) and [implementation notes](https://github.com/wulinze/sglang/blob/49349fe5b91909dfadfbd64c64e64b93004104b8/pr-body.md).

**Draft:** multi-GPU/model accuracy and performance validation remain pending.

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

Enable with `--enable-mxfp4-mega --moe-a2a-backend megamoe`. Reuse FusedMoE and quantization interfaces: native MXFP4 via Humming, or explicit W4AFP8 INT4-to-MXFP4 requantization and weight packing. The conversion needs accuracy evaluation.

SGLang preserves routing, expert IDs/top-k weights and shared-expert semantics; FlashInfer stages activations/routing and executes SM90 experts with NVSHMEM. Mapped combine receives destination rank/row indices and peer pointers, incorporates shared outputs/gates and writes the next layer's layout directly. Residual/mHC uses the same row plan. Expert communication still occurs.

ModelRunner reserves bounded workspaces before KV budgeting, agrees on geometry/tactics across ranks and warms launch variants. Serving reuses these buffers on their bound stream; borrowed outputs must be consumed before reuse, and empty ranks still participate.

### Serving support

Prebuild batch metadata and packed uploads; integrate fused projection/RoPE/gating, logical radix keys and checkpoint ownership, bulk PD staging with per-entry dtypes and target/draft KV matching, CP gather/scatter and completion ordering. Also integrate chunked/external-encoder prefill, profiling ranges and stable gateway worker selection.

## Current limitations

- **Topology:** PD prefill using Mooncake/Mooncake TCP; `CP = TP >= 2`; `DP = attention-DP = PP = MoE-DP = DCP = 1`. No attention head sharding or pipeline composition for this path.
- **GDN:** SM90/SM100/SM103, BF16, key/value head dimensions 128; `flash-qla==0.1.3`, `tilelang>=0.1.12`. Explicit positive request capacity and page size divisible by CP; QSA interleave needs a positive prefill chunk size.
- **Execution:** eager only; external encoder plus language-only worker for multimodal. No inline vision, speculation, LoRA, HiCache, unified memory, INT8 Mamba checkpoints, linear ReplaySSM or memory saver. FA3 still rejects FP8 E5M2.
- **Symm:** peer RMA requires one host and Torch CUDA symmetric allocation. GDN multicast additionally requires 2/4/8 ranks and multicast/stream-memory capabilities on every rank. Its NCCL fallback does **not** cover explicitly selected ring RMA: unsupported `symm_rma` raises an error; select NCCL instead.
- **New MXFP4 Mega path:** SM90, single-node pure EP, `CP = EP = TP`, `MoE-TP = 1`; NVSHMEM and compatible FlashInfer SM90 APIs required. Only supported MXFP4 or explicit W4AFP8 conversion inputs; no TBO/SBO. External API/version compatibility remains to be validated.

## Accuracy Tests

Head `6ce47b3dd2`, Python 3.12 / CPU Torch on macOS: **604 passed tests** (104 QSA/CP + 128 disaggregation + 372 server-argument tests), plus 431 passing subtests. CUDA cases: 348 skipped, 13 deselected. Three MLX/xgrammar-dependent argument cases were excluded after reproducing failures on the unchanged base.

All 124 changed Python files compiled; runnable pre-commit checks passed. Rust formatting/build/tests remain pending. CPU results do not establish GPU numerical or distributed model accuracy.

## Speed Tests and Profiling

No speedup numbers are claimed. Pending: Qwen3.5/Qwen3.8 accuracy, conversion accuracy, multi-rank ragged/empty/prefix cases, end-to-end PD and matched-workload TTFT/throughput/memory/transfer benchmarks. Profiling must establish actual communication/computation overlap and quantify workspace costs.

## Checklist

- [x] Add or extend regression/GPU tests and explain both model paths with diagrams.
- [ ] Complete GPU/model accuracy, PD and Mega dependency validation.
- [ ] Complete performance profiling, gateway Rust checks and community CI.
