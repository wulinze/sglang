## Motivation

Add context-parallel prefill for **Qwen3.5 and Qwen3.8**, covering their hybrid gated delta network (GDN), full/sparse attention and MoE execution. Long prompts need sequence sharding across these layers while preserving causal recurrence, query/KV correspondence, prefix-cache reuse and prefill-to-decode transfer.

The shared design is described in [Qwen3.5 Context Parallel: system design and computation flow](https://wulinze.github.io/llm/qwen35-context-parallel/) (Chinese). This PR includes both model integrations and their runtime, cache and transfer support. Qwen3.8 enters through the repository's existing `qwen4_exp` implementation, including QSA/indexer and model-specific residual/PLE handling.

**Draft:** CPU regression checks are available below. Multi-GPU numerical accuracy, model-level evaluation and performance measurements for this head are still pending.

## Modifications

![Hybrid CP execution flow](https://raw.githubusercontent.com/wulinze/sglang/b0b4589c1c5708d72acea79f187edd784b1c7582/hybrid-cp-flow.png)

[Editable Excalidraw flow diagram](https://github.com/wulinze/sglang/blob/b0b4589c1c5708d72acea79f187edd784b1c7582/hybrid-cp-flow.excalidraw)

### Token partitioning and layer boundaries

CP partitions token rows. In the `CP = TP` configuration, attention TP is one, so a rank retains the full hidden/head width for its local rows. Query ownership and KV storage follow each attention backend's contract:

| Consumer | Partition | Reason |
| --- | --- | --- |
| GDN | Per-request contiguous segments, with a segment width rounded up to a multiple of 64 | Preserve temporal order and kernel chunk boundaries |
| Full-attention queries | Round-robin by absolute token position by default; zigzag is also supported | Distribute causal attention work across ranks |
| Qwen3.5 sharded full-attention KV | Absolute token position modulo CP size | Keep ownership stable across prefix hits, chunk boundaries and query-layout changes |
| Qwen3.8 QSA queries | Interleaved rows from the packed batch; zigzag remains supported | Keep local queries aligned with the CP/indexer plan |
| Qwen3.8 sparse-attention KV | Gather fresh K/V and combine with cached prefix under the existing QSA cache contract | Preserve exact sparse top-k positions and fresh BF16 values |
| MoE / MLP | Valid rows from the current layout; output can target the next layer's layout | Preserve routing semantics while avoiding a separate layout exchange where supported |

![CP token partitions](https://raw.githubusercontent.com/wulinze/sglang/b0b4589c1c5708d72acea79f187edd784b1c7582/cp-token-partitions.png)

[Editable Excalidraw partition diagram](https://github.com/wulinze/sglang/blob/b0b4589c1c5708d72acea79f187edd784b1c7582/cp-token-partitions.excalidraw)

For `N` new tokens and `C` CP ranks:

```text
GDN segment width = 64 * ceil(N / (64 * C))
Sharded full-attention KV owner for new token i = (prefix_length + i) % C
QSA interleaved query owner = packed_new_token_row % C
```

For example, 1,000 new tokens on four ranks use a segment width of 256; the last GDN shard has 232 valid rows. Kernel padding is excluded from expert dispatch and persistent KV storage. In the sharded full-attention path, round-robin queries use the same prefix-aware ownership as new KV writes, allowing the corresponding redistribution to be skipped. Zigzag changes query ownership and packing while retaining that KV ownership rule. QSA's interleave layout instead uses the packed new-token stream; its query positions include each request's prefix for causal scoring.

The scheduler prepares a batch-owned plan for token routes, lengths and indices, packs read-only metadata into a pinned upload, and resolves cache slots from the device request table. Existing layer-boundary preparation/output hooks move hidden states and residuals using the same mapping. Model exit restores logical token order.

### GDN recurrence and full attention

GDN CP enters through the existing linear-attention backend's prefill dispatch using FlashQLA. It exchanges the causal-convolution halo separately from the recurrent state, computes a local affine state summary, and combines preceding summaries in sequence order to supply each shard's initial state.

For one head, `S` and `H` have shape `[dk, dv]` and `M` has shape `[dk, dk]`; `@` denotes matrix multiplication:

```text
F_r(S) = M_r @ S + H_r
F_b(F_a(S)) = (M_b @ M_a) @ S + (M_b @ H_a + H_b)
```

Composition is associative but order-sensitive. Cached prefix state supplies the starting state; an empty shard contributes the identity transformation. Convolution and temporal checkpoints retain their ownership for later prefix reuse and PD transfer.

Full attention keeps Q local and visits the persistent KV shards through a double-buffered ring. Existing FA3/TRT-LLM attention kernels produce partial outputs and log-sum-exp (LSE) values. These are merged with stable LSE weights, preserving causal masking by global position. KV-ring transport is selectable through `--attn-cp-transport nccl|symm_rma`; its resources are distinct from GDN state exchange and expert communication.

### Qwen3.8 QSA and model integration

The `qwen4_exp` model connects its GDN layers to the same FlashQLA CP dispatcher and contiguous row transport. Its QSA layers retain the existing sparse-attention/indexer entry points, using round-robin queries and an index-key ring for the interleave path. Sparse attention gathers fresh K/V once, restores token order and packs each request's cached prefix. Fused packing/cache stores preserve fresh BF16 inputs even when the cache uses FP8 or cache writes are disabled, together with absolute causal positions and logical top-k indices. The QSA index-key ring and the full-attention KV ring carry different data and serve different computations.

Model integration also handles strided GDN projections, external-encoder embeddings, full-head state/checkpoint ownership and native MXFP4 Mega execution through the existing loading/quantization path. Layer-boundary plans route attention/MoE outputs and residual or mHC streams to the next layer's layout; PLE operations that need global sequence order gather the relevant rows and select the rank-local result. Backend metadata and enabled-kernel warmup are prepared through the existing batch and ModelRunner paths.

### Runtime and serving integration

| Area | Changes and review entry points |
| --- | --- |
| Layouts and execution boundaries | Reusable token exchange/gather plans in `layers/cp/utils.py`; integration through `layers/layer_boundary/`, the hybrid attention backend and model forwards |
| Model integration | Qwen3.5 in `models/qwen3_5.py`; Qwen3.8 in `models/qwen4_exp.py`, with QSA/indexer changes in `layers/attention/qwen_sparse_attn_backend.py` and `layers/attention/qsa/` |
| Kernels and communication | Fused QK normalization/RoPE/gating and GDN projection preparation; shared communication buffers and lifecycle handling in `TorchSymmMemCommunicator` |
| MoE | Integrate optional SM90 MXFP4 Mega execution through the existing MoE/quantization path. Mapped combine output and asynchronous residual exchange target the next layer's layout while preserving expert IDs, top-k weights and shared-expert semantics |
| Cache and allocation | Shard logical KV and Mamba checkpoint storage, preserve logical radix keys, and track compatible KV/state boundaries for prefix hits |
| PD transfer | Bulk staging with per-entry byte sizes/dtypes, target/draft KV slot matching, CP page gather/scatter, state-owner selection and completion ordering; handle unwritten staging rows in the fallback path |
| Scheduling and warmup | Chunked and external-encoder multimodal prefill integration; reserve communication/expert workspaces before KV budgeting and run compilation/tuning through ModelRunner |
| Observability and routing | Runtime/HTTP profiling controls and execution ranges; stable worker selection and consistent hash-ring membership updates in the gateway to preserve session/cache locality |

### Symmetric-memory integration

Reuse `TorchSymmMemCommunicator` as the group/device resource owner for peer RMA, ring buffers, peer-state publication and multicast broadcast. These resources are independent of the optional symmetric all-reduce eligibility; enabling CP transport does not require enabling a new attention backend or changing every collective in the model.

- `--attn-cp-transport nccl|symm_rma` selects the transport for the full-attention KV ring and the QSA index-key ring. The two rings retain separate workspaces and message semantics. Double-buffered payloads, arrival/credit signals and explicit producer/consumer stream ordering protect slot reuse.
- GDN publishes local affine state summaries for ordered peer-prefix computation and broadcasts the initial/checkpoint state from its actual owner when multicast is available. Startup capability agreement is collective across the group; unavailable multicast/state prerequisites select the NCCL state path and log the reason. This is separate from the explicit attention-ring transport setting.
- Residual/token layout exchange can use persistent symmetric peer-row buffers on a dedicated stream, with precomputed source/destination ranges and events. Mapped Mega output also uses Torch symmetric peer destinations while retaining the Mega runtime's own NVSHMEM workspaces.

Peer RMA requires all ranks on one host, Torch's CUDA symmetric allocator and working peer mappings. The GDN multicast path additionally requires 2, 4 or 8 ranks, SM90-or-newer multicast capability and CUDA stream memory operations on every participating rank. Explicit `symm_rma` selection raises an error when its prerequisites fail; users must select NCCL for the ring in that situation. Communication buffers have fixed capacities and stream ownership; graph capture is not supported on these paths.

### MegaMoE integration

The new opt-in path is selected by `--enable-mxfp4-mega` together with `--moe-a2a-backend megamoe`. It enters the existing FusedMoE and quantization interfaces. Native MXFP4 experts are prepared through the Humming path; W4AFP8 experts use an explicit post-load INT4-to-MXFP4 requantization and Mega weight-packing step. That conversion changes the numerical representation and requires accuracy evaluation.

SGLang retains routing, top-k expert IDs/weights and shared-expert handling. FlashInfer stages activations/routing and executes the SM90 experts using NVSHMEM workspaces. When the next layer needs another token layout, the combine kernel receives destination rank/row indices and peer pointers, incorporating shared outputs/gates as it writes directly to that layout. Residual or mHC exchange uses the same row plan. This removes the corresponding standalone output redistribution where enabled; expert dispatch/combine communication still occurs.

ModelRunner warmup allocates capacity from the request/chunk/context bounds, agrees on workspace geometry/tactics across EP ranks, compiles both ordinary and mapped-combine launch variants, and accounts for reserved memory before KV budgeting. Serving reuses the warmed workspace on its bound stream. Outputs borrow that workspace and must be consumed before reuse; empty local ranks still participate in the EP operation.

### Current limitations and dependencies

The following limits describe this hybrid CP implementation, rather than the capabilities of every existing SGLang backend:

| Area | Current implementation boundary |
| --- | --- |
| Parallel topology / serving | PD prefill with `mooncake` or `mooncake_tcp`; `CP = TP >= 2`; `DP = attention-DP = PP = MoE-DP = DCP = 1`. These row plans do not compose with attention head sharding or pipeline-stage boundaries. Decode-side CP and speculative execution are outside this path. |
| GDN hardware / dependencies | FlashQLA explicitly checks SM90, SM100 or SM103; `flash-qla==0.1.3` and `tilelang>=0.1.12` are required. Key/value head dimensions must both be 128 and model execution must be BF16. |
| Storage / bounds | Positive explicit `max-running-requests` and page size divisible by CP; ring/Mega workspaces use bounded startup geometry. QSA interleave warmup requires a positive prefill chunk size. |
| Attention | Sharded full attention uses FA3 or TRT-LLM MHA; Qwen3.8 retains the QSA backend. FA3's FP8 E5M2 restriction remains. The prefill worker does not use a separately selected decode attention backend. |
| Multimodal / execution | External encoder plus language-only prefill worker; inline vision encoding is unsupported. CUDA graph phases are disabled for this GDN CP path. |
| Unsupported combinations | LoRA, speculation, HiCache, unified memory, INT8 Mamba checkpoints, linear ReplaySSM and memory saver are rejected by the CP argument checks. |
| Symm | Peer RMA is single-host. GDN multicast has the additional 2/4/8-rank capability constraints above; its NCCL fallback does not imply automatic fallback for explicitly selected ring RMA. |
| New MXFP4 Mega path | SM90 only, single-node pure EP with `CP = EP = TP`, `MoE-TP = 1`; requires both opt-in flags, NVSHMEM and compatible FlashInfer SM90 APIs. Other weights are rejected unless they use the supported native-MXFP4 or explicit W4AFP8 conversion path. TBO/SBO are unsupported by the Mega dispatch/combine interface. |
| Validation | CPU regressions cover selected contracts. Multi-GPU accuracy, quantization-conversion accuracy, end-to-end PD behavior and performance remain unvalidated for this head; gateway Rust checks are also pending. |

MoE TP and EP are resolved separately from CP; enabling CP alone does not enable Mega. FlashInfer SM90 staging, conversion and mapped-combine API/version compatibility still needs to be established for an upstream-reproducible environment. The optional SM90 path above does not extend the support matrix of other Mega implementations to this integration.

## Accuracy Tests

Validation for code head `6ce47b3dd20136dd0cfcf6154a2fb7037e37ad84`, on macOS with Python 3.12 / CPU Torch:

| Check group | Result |
| --- | --- |
| Selected QSA / CP regression suites | 104 passed, 348 skipped, 13 CUDA-only cases deselected |
| Disaggregation wire, bootstrap, batching and bulk-transfer regressions | 128 passed; 153 subtests passed |
| Server argument regressions | 372 passed; 278 subtests passed; 3 environment-dependent cases deselected |
| Changed Python files | 124 compiled successfully |
| Pre-commit | Runnable checks passed; gateway rustfmt skipped because the Rust toolchain is unavailable |

Total: **604 passed tests**, plus 431 passing subtests. The three server-argument exclusions require unavailable MLX/xgrammar components and also failed on the unchanged base in the same environment. CUDA-dependent tests were skipped/deselected; these results do not establish multi-rank numerical correctness. The gateway changes have not been built or tested with Rust here.

Before marking ready for review:

- [ ] Run GPU kernel and multi-rank CP comparisons, including round-robin/zigzag, ragged/empty shards, nonzero prefixes and chunk boundaries.
- [ ] Validate GDN convolution/state checkpoints and end-to-end PD reconstruction, including mixed KV dtypes and target/draft layouts.
- [ ] Run Qwen3.5 and Qwen3.8 model-level accuracy against matching non-CP baselines, plus the gateway Rust checks.

## Speed Tests and Profiling

No speedup numbers are claimed for this head. The design article explains the computation/communication structure; it is not benchmark evidence for this PR.

Pending measurements should compare matched model weights, quantization, hardware and request workloads across CP sizes, with cold/warm prefix caches and short/long incremental prefill. Report TTFT, prefill throughput, memory usage and PD transfer time, and use the added profiling ranges to inspect metadata upload, GDN state/halo exchange, the KV ring, expert dispatch/combine and residual layout exchanges.

## Checklist

- [x] Add or extend focused regression tests and GPU test coverage in the test tree.
- [x] Describe CP partitioning and execution flow, with editable diagrams.
- [ ] Complete GPU accuracy and performance validation.
- [ ] Validate external FlashInfer/NVSHMEM compatibility for the optional Mega path.
- [ ] Complete gateway Rust formatting/build/tests and required community CI.
- [ ] Add user-facing deployment documentation once the validated configuration is established.
