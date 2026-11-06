## Motivation

Add context-parallel prefill for **Qwen3.5 and Qwen3.8** (the existing `qwen4_exp.py` path), preserving causal state, token order, prefix reuse and PD transfer.

Design: [original article](https://wulinze.github.io/llm/qwen35-context-parallel/) and [implementation notes](https://github.com/wulinze/sglang/blob/49349fe5b91909dfadfbd64c64e64b93004104b8/pr-body.md).

**Draft:** multi-GPU/model accuracy and performance validation remain pending.

## Modifications

### CP partitioning

- **GDN:** 64-token-aligned contiguous segments, convolution halos, ordered affine-state prefix composition, FlashQLA prefill and convolution/temporal checkpoints.
- **Qwen3.5:** local round-robin Q (or zigzag), KV ownership by absolute position, a double-buffered KV ring and causal FA3/TRT-LLM attention merged with LSE weights.
- **Qwen3.8 QSA:** interleaved packed query rows, index-key ring and sparse top-k selection. Gather new KV once and pack prefixes, preserving fresh BF16 values and causal positions under the existing QSA cache contract.

For `N` new tokens and `C` ranks:

```text
GDN segment width = 64 * ceil(N / (64 * C))
Sharded full-attention KV owner = (prefix_length + new_token_index) % C
QSA interleaved query owner = packed_new_token_row % C
```

Padding stays out of dispatch/KV storage. Hidden/residual/mHC row plans agree, PLE retains global-order inputs, and model outputs return to logical order.

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

Prebuild metadata/uploads; integrate fused projection/RoPE/gating, prefix/checkpoint ownership, bulk PD staging with mixed dtypes, target/draft matching and CP scatter/completion. Add chunked/external-encoder prefill, profiling and stable gateway routing.

## Current limitations

- **Topology:** PD prefill using Mooncake/Mooncake TCP; `CP = TP >= 2`; `DP = attention-DP = PP = MoE-DP = DCP = 1`. No attention head sharding or pipeline composition for this path.
- **GDN:** SM90/SM100/SM103, BF16, key/value head dimensions 128; `flash-qla==0.1.3`, `tilelang>=0.1.12`. Explicit positive request capacity and page size divisible by CP; QSA interleave needs a positive prefill chunk size.
- **Execution:** eager only; external encoder plus language-only worker for multimodal. No inline vision, speculation, LoRA, HiCache, unified memory, INT8 Mamba checkpoints, linear ReplaySSM or memory saver. FA3 still rejects FP8 E5M2.
- **Symm:** peer RMA requires one host and Torch CUDA symmetric allocation. GDN multicast additionally requires 2/4/8 ranks and multicast/stream-memory capabilities on every rank. Its NCCL fallback does **not** cover explicitly selected ring RMA: unsupported `symm_rma` raises an error; select NCCL instead.
- **New MXFP4 Mega path:** SM90, single-node pure EP, `CP = EP = TP`, `MoE-TP = 1`; NVSHMEM and compatible FlashInfer SM90 APIs required. Only supported MXFP4 or explicit W4AFP8 conversion inputs; no TBO/SBO. External API/version compatibility remains to be validated.

## Accuracy Tests

Head `6ce47b3dd2`, Python 3.12 / CPU Torch: **604 tests + 431 subtests passed**. CUDA: 348 skipped, 13 deselected. Three MLX/xgrammar-dependent cases also failed on the unchanged base and were excluded.

124 Python files compiled; runnable pre-commit checks passed. Rust checks remain pending. GPU/distributed accuracy is not established.

## Speed Tests and Profiling

No speedup claims. Pending: both models' accuracy, requantization accuracy, multi-rank edge cases, end-to-end PD, and matched TTFT/throughput/memory/transfer benchmarks. Traces must verify overlap and workspace costs.

## Checklist

- [x] Add or extend regression/GPU tests and explain both model paths with diagrams.
- [ ] Complete GPU/model accuracy, PD and Mega dependency validation.
- [ ] Complete performance profiling, gateway Rust checks and community CI.
