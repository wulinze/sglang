## Motivation

Add context-parallel prefill for **Qwen3.5 and Qwen3.8**, covering their hybrid gated delta network (GDN), full/sparse attention and MoE execution. Long prompts need sequence sharding across these layers while preserving causal recurrence, query/KV correspondence, prefix-cache reuse and prefill-to-decode transfer.

The shared design is described in [Qwen3.5 Context Parallel: system design and computation flow](https://wulinze.github.io/llm/qwen35-context-parallel/) (Chinese). This PR includes both model integrations and their runtime, cache and transfer support. Qwen3.8 enters through the repository's existing `qwen4_exp` implementation, including QSA/indexer and model-specific residual/PLE handling.

**Validation:** The development implementation has completed GPU numerical checks, Qwen3.5/Qwen3.8 model accuracy evaluations and end-to-end serving tests. Results and coverage are summarized below.

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
| Observability | Runtime/HTTP profiling controls and execution ranges |

### Symmetric-memory integration

Reuse `TorchSymmMemCommunicator` as the group/device resource owner for peer RMA, ring buffers, peer-state publication and multicast broadcast. These resources are independent of the optional symmetric all-reduce eligibility; enabling CP transport does not require enabling a new attention backend or changing every collective in the model.

- `--attn-cp-transport nccl|symm_rma` selects the transport for the full-attention KV ring and the QSA index-key ring. The two rings retain separate workspaces and message semantics. Double-buffered payloads, arrival/credit signals and explicit producer/consumer stream ordering protect slot reuse.
- GDN publishes local affine state summaries for ordered peer-prefix computation and broadcasts the initial/checkpoint state from its actual owner when multicast is available. Startup capability agreement is collective across the group; unavailable multicast/state prerequisites select the NCCL state path and log the reason. This is separate from the explicit attention-ring transport setting.
- Residual/token layout exchange can use persistent symmetric peer-row buffers on a dedicated stream, with precomputed source/destination ranges and events. Mapped Mega output also uses Torch symmetric peer destinations while retaining the Mega runtime's own NVSHMEM workspaces.

Peer RMA requires all ranks on one host, Torch's CUDA symmetric allocator and working peer mappings. The GDN multicast path additionally requires 2, 4 or 8 ranks, SM90-or-newer multicast capability and CUDA stream memory operations on every participating rank. Explicit `symm_rma` selection raises an error when its prerequisites fail; users must select NCCL for the ring in that situation. Communication buffers have fixed capacities and stream ownership; graph capture is not supported on these paths.

### MegaMoE integration

The new opt-in path is selected by `--enable-mxfp4-mega` together with `--moe-a2a-backend megamoe`. It enters the existing FusedMoE and quantization interfaces. Native MXFP4 experts require a compatible checkpoint adapter and Humming layer metadata; W4AFP8 experts use an explicit post-load INT4-to-MXFP4 requantization and Mega weight-packing step. That conversion changes the numerical representation. Completed serving accuracy is documented below; an isolated comparison against the original representation remains follow-up work.

SGLang retains routing, top-k expert IDs/weights and shared-expert handling. FlashInfer stages activations/routing and executes the SM90 experts using NVSHMEM workspaces. When the next layer needs another token layout, the combine kernel receives destination rank/row indices and peer pointers, incorporating shared outputs/gates as it writes directly to that layout. Residual or mHC exchange uses the same row plan. This removes the corresponding standalone output redistribution where enabled; expert dispatch/combine communication still occurs.

ModelRunner warmup allocates capacity from the request/chunk/context bounds, agrees on workspace geometry/tactics across EP ranks, compiles both ordinary and mapped-combine launch variants, and accounts for reserved memory before KV budgeting. Serving reuses the warmed workspace on its bound stream. Outputs borrow that workspace and must be consumed before reuse; empty local ranks still participate in the EP operation.

## Parameters and launch configurations

The tables below cover the CP, Symm, MegaMoE and disaggregated-serving controls used by this implementation. Defaults are the declared CLI defaults before model/device resolution; `unset` means that SGLang resolves the value later. **This PR adds** `--attn-cp-transport`, `--enable-mxfp4-mega`, `--disaggregation-require-bulk`, `SGLANG_DISAGG_REQUIRE_BULK`, and the `flashqla` linear-attention backend choice. The other options reuse existing interfaces.

### CP and attention backends

| Option | Default | Configuration for this path |
| --- | --- | --- |
| `--tp-size C` | `1` | Total TP group width. Use `C >= 2` and set `--attn-cp-size C`; this path requires `CP = TP`. |
| `--attn-cp-size C` | `1` | Number of context-parallel ranks. `--attention-context-parallel-size` is an alias. |
| `--enable-prefill-cp` | Off | Required for Qwen3.8 CP/TP group sharing. The examples also set it explicitly for Qwen3.5. Qwen3.5 GDN CP activation itself is determined by CP size, FlashQLA and the prefill role. |
| `--cp-strategy interleave` | Unset | Choices: `interleave`, `zigzag`. Must be explicit with `--enable-prefill-cp`. Interleave splits Qwen3.5 full-attention queries by request-local row and Qwen3.8 QSA queries by packed-batch row. GDN keeps its contiguous, 64-token-aligned partition under either strategy. |
| `--linear-attn-prefill-backend flashqla` | Unset; inherits `--linear-attn-backend` (`triton`) | Selects the new GDN CP prefill path. Use this prefill-specific override: FlashQLA is not a decode/verify backend. Do not set the global `--linear-attn-backend flashqla`. |
| `--attention-backend` / `--prefill-attention-backend` | Unset; model/device selection | Qwen3.5 sharded full attention requires `fa3` on SM90 or `trtllm_mha` on a supported newer GPU. Qwen3.8 uses its model-specific QSA path. `--prefill-attention-backend` takes precedence for prefill; any explicit `--decode-attention-backend` on the P worker must match the resolved prefill backend. |
| `--attn-cp-transport` | `nccl` | Choices: `nccl`, `symm_rma`. Selects the full-attention KV ring and QSA index-key ring transport; it does not select the GDN state algorithm or the MoE backend. |
| `--dtype bfloat16` | `auto` | The model must resolve to BF16; key/value GDN head dimensions must both be 128. This activation/model dtype is separate from weight quantization and KV-cache storage dtype. |
| `--kv-cache-dtype` | `auto` | Keep the P/D KV representations compatible. `bfloat16` is used in the examples to avoid an implicit FP8 setting. FA3 rejects `fp8_e5m2`; model-specific quantized-cache paths still require their checkpoint metadata/scales. |
| `--qsa-indexer-dtype` | `auto` | Qwen3.8 compressed QSA only: `auto`, `bfloat16`, `fp8_e4m3`. Changes index-key/query storage and scoring, not the main KV cache. Explicit FP8 indexer mode requires a compressed QSA profile and CUDA SM90/SM100. |
| `--dp-size`, `--attn-dp-size`, `--pp-size`, `--moe-dp-size`, `--dcp-size` | All `1` | Keep all at `1` for this GDN CP path. Qwen3.8 also rejects mixed prefill/decode chunks, DSA cache-layer splitting and waterfill routing. |

CP query partitioning and persistent cache ownership are separate. Qwen3.5 automatically uses the sharded full-attention KV contract; Qwen3.8 CP/TP group sharing retains its replicated QSA KV contract. Do not use `--cp-kv-layout` as a substitute for selecting one of these model paths. CP/TP group sharing is derived internally, not a separate launch flag.

### Capacity, pages and prefix caching

| Option | Default | Configuration for this path |
| --- | --- | --- |
| `--max-running-requests R` | Unset | Set an explicit positive bound. CP and mapped Mega workspace allocation use it; do not leave it to automatic memory sizing. |
| `--chunked-prefill-size T` | Unset; automatically resolved | Set a positive token budget, especially for Qwen3.8 QSA interleave warmup. The examples use `8192`; this is a configuration example, not a measured optimum. Workspace sizing includes per-rank CP padding. |
| `--max-prefill-tokens` | `16384` | Existing prefill batching budget. Tune together with the chunk budget and request capacity. |
| `--context-length` | Unset; model configuration | Maximum sequence length. Increasing it changes cache/metadata capacity requirements; the checkpoint must support the requested length. |
| `--page-size` | Unset; backend/model selection | Must be positive and divisible by CP. The compressed Qwen3.8 QSA profile resolves it to `64`. The examples use `64` for CP2/CP4. |
| `--mamba-radix-cache-strategy` | `auto` | Use `extra_buffer` for the illustrated page-aligned hybrid prefix-cache configuration. `no_buffer` requires page size `1`, which cannot satisfy this CP page contract. `extra_buffer_lazy` is not supported under PD disaggregation. |
| `--mamba-track-interval` | `256` | Existing state-checkpoint tracking interval; must be divisible by page size. Retain a chunk budget large enough to produce intermediate Mamba checkpoints. |
| `--mem-fraction-static` | Unset; automatic | Overall static memory budget. Mega reserves its workspaces before KV-pool budgeting; increasing concurrency/chunk size therefore reduces space available for KV. |
| `--mamba-full-memory-ratio` / `--max-mamba-cache-size` | Unset | Existing hybrid-state sizing controls: memory ratio versus full-attention KV, or an explicit Mamba entry bound. Choose from workload/concurrency needs; CP does not introduce a universal recommended ratio. |
| `--max-total-tokens` | Unset; memory-derived | Optional cap on the KV token pool. It is distinct from per-request context length and per-iteration prefill tokens. |

CUDA graphs are disabled automatically for this GDN CP execution path. Do not enable deterministic inference with FlashQLA. Speculation, LoRA, HiCache, unified memory, INT8 Mamba checkpoints, linear ReplaySSM and memory saver are unsupported on the CP prefill worker. A separate decode worker has its own supported configuration.

### Symm and MXFP4 MegaMoE

| Option | Default | Configuration for this path |
| --- | --- | --- |
| `--attn-cp-transport symm_rma` | `nccl` | Optional peer RMA rings. Requires peers on one host and working Torch CUDA symmetric allocation. An unsupported explicit RMA selection raises; select `nccl` when these prerequisites are absent. |
| `--enable-mxfp4-mega` | Off | Opts into the new FlashInfer SM90 MXFP4 Mega implementation. Requires `--moe-a2a-backend megamoe`, single-node pure EP, `CP = EP = TP >= 2`, `MoE-TP = 1`, PD prefill and a positive request bound. |
| `--moe-a2a-backend megamoe` | `none` | Selects Mega expert communication/compute. This option alone can select other existing Mega implementations; it is not equivalent to enabling the new MXFP4 path. |
| `--ep-size C` | `1` | Set equal to CP/TP in the examples. The existing Mega A2A normalization also resolves EP to TP. |
| `--moe-runner-backend` | `auto` | Leave `auto` for these examples. The existing `megamoe` runner alias normalizes to the Mega A2A backend; it does not replace `--enable-mxfp4-mega`. |
| `--quantization` | Unset; checkpoint-derived | For a compatible W4AFP8 checkpoint, `w4afp8` plus the Mega flag performs explicit INT4-to-MXFP4 conversion and packing. Do not apply that setting to arbitrary BF16/native-MXFP4 weights. Native MXFP4 Mega needs a compatible checkpoint adapter and Humming layer metadata; `--quantization humming` alone is not a generic RedSlim adapter. |
| `--nnodes` | `1` | Keep `1` for the new MXFP4 Mega path and explicit Symm RMA. |

The new MXFP4 Mega implementation is **SM90-only** and requires its FlashInfer/NVSHMEM APIs. GDN support on SM100/SM103 does not extend this Mega implementation to those devices. TBO/SBO must remain disabled. Its bounded workspace is calculated from CP padding, chunk/context limits and request capacity; `SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK` belongs to the other Mega implementation and does not size this workspace.

GDN symmetric state exchange is selected internally after collective capability checks. It falls back to NCCL if multicast prerequisites are absent. This fallback does not cover an explicitly requested `symm_rma` attention ring. Torch symmetric buffers for rings/layout exchange and NVSHMEM expert buffers have separate lifetimes; enabling optional Torch symmetric all-reduce is not required to activate the CP rings.

### PD/EPD roles and bulk transfer

| Option | Default | Configuration for this path |
| --- | --- | --- |
| `--disaggregation-mode` | `null` | Set `prefill` on the CP worker. Run a separate `decode` worker with CP size `1`; do not copy the CP-only FlashQLA/Mega flags onto it. |
| `--disaggregation-transfer-backend` | `mooncake` | This CP path accepts `mooncake` or `mooncake_tcp`. Configure compatible transport on P and D. |
| `--disaggregation-bootstrap-port` | `8998` | P-worker bootstrap port. Use distinct/reachable endpoints for separate prefill instances. |
| `--disaggregation-ib-device` | Unset | Optional Mooncake NIC selection: an HCA name, comma-separated list, per-GPU JSON mapping, or mapping file. Leave unset to retain backend selection. |
| `--disaggregation-decode-allocation-policy` | `early` | Keep `early` for this hybrid GDN path; the existing `prefill_complete` policy rejects non-DSA state descriptors. |
| `--disaggregation-require-bulk` | Off | Require bulk staging instead of allowing a direct-write fallback. Requires `SGLANG_DISAGG_STAGING_BUFFER=1`; enable consistently on compatible P/D workers. |
| `--language-only` | Off | Required for a multimodal checkpoint on the CP worker; moves encoding to a separate service. It is not the standalone `--language-model-only` option. |
| `--encoder-only` | Off | Launch a separate encoder service without CP/FlashQLA/Mega flags. |
| `--encoder-urls` | Empty list | Supply one or more encoder endpoints to the language-only worker, e.g. `--encoder-urls "$ENCODER_URL"`. |
| `--encoder-transfer-backend` | `auto` | Choices: `auto`, `zmq_to_scheduler`, `zmq_to_tokenizer`, `mooncake`. `auto` resolves by model/TP; E and P must use a compatible transfer route. |

Bulk staging is optional and requires compatible **MHA/GQA** pools. It must be enabled on both P and D because it changes the completion protocol. **Qwen3.8 CP/TP group sharing rejects staging:** leave `SGLANG_DISAGG_STAGING_BUFFER`, `SGLANG_DISAGG_REQUIRE_BULK` and `--disaggregation-require-bulk` off for that configuration. Its ordinary Mooncake transfer still handles the model's KV/state descriptors.

| Environment variable | Default | Purpose |
| --- | --- | --- |
| `SGLANG_DISAGG_STAGING_BUFFER` | `0` | Enable the compatible MHA/GQA bulk staging path on both P and D. |
| `SGLANG_DISAGG_REQUIRE_BULK` | `0` | Environment equivalent of `--disaggregation-require-bulk`; either setting being true requires bulk. Neither implicitly enables staging. |
| `SGLANG_DISAGG_STAGING_POOL_SIZE_MB` | `4096` | Staging-pool capacity in MiB; account for it in the worker memory budget. |
| `SGLANG_PLB_SCATTER_DEFER` | `0` | Optional decode-side scheduling of incoming KV scatters between forwards. |
| `SGLANG_PLB_STREAM_PRIORITY` | `0` | Optional decode stream priority relative to background scatters. |
| `SGLANG_PLB_SCATTER_CTAS` | `0` | Scatter kernel tuning: `0` original kernel, a positive value for persistent CTA count, `-1` one CTA per page. |
| `SGLANG_STAGING_USE_TORCH` | `0` | Diagnostic Torch staging implementation; normally leave off. |
| `SGLANG_MOONCAKE_MAX_TRANSFER_BATCH_INDICES` | `0` | Optional bound on indices in one synchronous all-layer transfer batch; `0` retains the existing single-batch behavior. |

No additional `SGLANG_DISAGGREGATION_ALL_CP_RANKS_TRANSFER=1` override is required for this implementation: participation follows the actual sharded KV/state ownership. Qwen3.8 group sharing has a separate TP-rank transfer contract; it must not be treated as replicated CP ranks with only rank zero transferring.

### Launch templates

These commands illustrate parameter combinations; they are not additional GPU test results or memory-capacity guarantees. Set the model and encoder variables to compatible deployed resources, install the required kernels/transfer dependencies, and size the request/chunk budgets for the available GPUs. The examples show P workers; E and D run as separate services with their own GPU allocation.

**Qwen3.5: CP4, FA3, language-only prefill.** For a compatible SM90 W4AFP8 MoE deployment, append the Mega options shown below.

```bash
python -m sglang.launch_server \
  --model-path "$QWEN35_MODEL" \
  --host 0.0.0.0 --port 30000 \
  --dtype bfloat16 \
  --tp-size 4 --attn-cp-size 4 \
  --enable-prefill-cp --cp-strategy interleave \
  --linear-attn-prefill-backend flashqla \
  --attention-backend fa3 \
  --attn-cp-transport nccl \
  --kv-cache-dtype bfloat16 --page-size 64 \
  --mamba-radix-cache-strategy extra_buffer \
  --mamba-track-interval 256 \
  --max-running-requests 8 --chunked-prefill-size 8192 \
  --disaggregation-mode prefill \
  --disaggregation-transfer-backend mooncake \
  --disaggregation-bootstrap-port 8998 \
  --language-only --encoder-urls "$ENCODER_URL"
```

Optional additions/replacements, each subject to its own compatibility conditions:

```text
Symm rings: replace --attn-cp-transport nccl with --attn-cp-transport symm_rma
SM90 W4AFP8 Mega: append --quantization w4afp8 --enable-mxfp4-mega --moe-a2a-backend megamoe --ep-size 4
Compatible bulk P/D: set SGLANG_DISAGG_STAGING_BUFFER=1 in both processes; optionally add --disaggregation-require-bulk to both
SM100 full attention: use --attention-backend trtllm_mha; omit the new --enable-mxfp4-mega path
```

**Qwen3.8: CP2, interleaved QSA, language-only prefill.** Keep checkpoint-compatible quantization/model attention selection; this CP template does not assume that an arbitrary native MXFP4 checkpoint has a working Mega loader. Keep staging/bulk disabled.

```bash
SGLANG_DISAGG_STAGING_BUFFER=0 SGLANG_DISAGG_REQUIRE_BULK=0 \
python -m sglang.launch_server \
  --model-path "$QWEN38_MODEL" \
  --host 0.0.0.0 --port 30000 \
  --dtype bfloat16 \
  --tp-size 2 --attn-cp-size 2 \
  --enable-prefill-cp --cp-strategy interleave \
  --linear-attn-prefill-backend flashqla \
  --attn-cp-transport nccl \
  --kv-cache-dtype bfloat16 --qsa-indexer-dtype bfloat16 \
  --page-size 64 --mamba-radix-cache-strategy extra_buffer \
  --mamba-track-interval 256 \
  --max-running-requests 8 --chunked-prefill-size 8192 \
  --disaggregation-mode prefill \
  --disaggregation-transfer-backend mooncake \
  --disaggregation-bootstrap-port 8998 \
  --language-only --encoder-urls "$ENCODER_URL"
```

For an already compatible SM90 Mega checkpoint/loader, the Qwen3.8 extension is `--enable-mxfp4-mega --moe-a2a-backend megamoe --ep-size 2`, with the checkpoint's supported quantization configuration. The prerequisites above still apply.

For profiling, the existing `--enable-layerwise-nvtx-marker` flag (default off) exposes the added ranges. Torch profiler ranges are emitted while the profiler is active. `--enable-profile-cuda-graph` does not enable profiling for this eager CP path.

Parameter definitions and validation: [parallel fields](https://github.com/wulinze/sglang/blob/main/python/sglang/srt/arg_groups/fields/parallel.py), [execution fields](https://github.com/wulinze/sglang/blob/main/python/sglang/srt/arg_groups/fields/exec_.py), [model fields](https://github.com/wulinze/sglang/blob/main/python/sglang/srt/arg_groups/fields/model.py), [capacity fields](https://github.com/wulinze/sglang/blob/main/python/sglang/srt/arg_groups/fields/schedule.py), [disaggregation fields](https://github.com/wulinze/sglang/blob/main/python/sglang/srt/arg_groups/fields/disagg.py), [CP validation](https://github.com/wulinze/sglang/blob/main/python/sglang/srt/arg_groups/parallel_hook.py), [CP/TP sharing](https://github.com/wulinze/sglang/blob/main/python/sglang/srt/arg_groups/cp_tp_group_sharing_hook.py), [Mega validation](https://github.com/wulinze/sglang/blob/main/python/sglang/srt/arg_groups/mega_moe_hook.py), and [environment defaults](https://github.com/wulinze/sglang/blob/main/python/sglang/srt/environ.py).

## Current limitations and dependencies

The following limits describe this hybrid CP implementation, rather than the capabilities of every existing SGLang backend:

| Area | Current implementation boundary |
| --- | --- |
| Parallel topology / serving | PD prefill with `mooncake` or `mooncake_tcp`; `CP = TP >= 2`; `DP = attention-DP = PP = MoE-DP = DCP = 1`. These row plans do not compose with attention head sharding or pipeline-stage boundaries. Decode-side CP and speculative execution are outside this path. |
| GDN hardware / dependencies | FlashQLA explicitly checks SM90, SM100 or SM103; `flash-qla==0.1.3` and `tilelang>=0.1.12` are required. Key/value head dimensions must both be 128 and model execution must be BF16. |
| Storage / bounds | Positive explicit `max-running-requests` and page size divisible by CP; ring/Mega workspaces use bounded startup geometry. QSA interleave warmup requires a positive prefill chunk size. |
| Attention | Sharded full attention uses FA3 or TRT-LLM MHA; Qwen3.8 retains the QSA backend. FA3's FP8 E5M2 restriction remains. The prefill worker does not use a separately selected decode attention backend. |
| Multimodal / execution | External encoder plus language-only prefill worker; inline vision encoding is unsupported. CUDA graph phases are disabled for this GDN CP path. |
| Unsupported combinations | LoRA, speculation, HiCache, unified memory, INT8 Mamba checkpoints, linear ReplaySSM and memory saver are rejected by the CP argument checks. |
| PD staging | Optional for compatible MHA/GQA pools; enable consistently on P/D. Qwen3.8 CP/TP group sharing rejects staging and must leave bulk enforcement off. |
| Symm | Peer RMA is single-host. GDN multicast has the additional 2/4/8-rank capability constraints above; its NCCL fallback does not imply automatic fallback for explicitly selected ring RMA. |
| New MXFP4 Mega path | SM90 only, single-node pure EP with `CP = EP = TP`, `MoE-TP = 1`; requires both opt-in flags, NVSHMEM and compatible FlashInfer SM90 APIs. Other weights are rejected unless they use the supported native-MXFP4 or explicit W4AFP8 conversion path. TBO/SBO are unsupported by the Mega dispatch/combine interface. |
| Validation | Development GPU, model-accuracy and complete EPD/PD tests are recorded below, together with regression checks. Matching non-CP/conversion comparisons, upstream dependency reproducibility and community CI remain outstanding. |

MoE TP and EP are resolved separately from CP; enabling CP alone does not enable Mega. FlashInfer SM90 staging, conversion and mapped-combine API/version compatibility still needs to be established for an upstream-reproducible environment. The optional SM90 path above does not extend the support matrix of other Mega implementations to this integration.

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

## Checklist

- [x] Add or extend regression/GPU tests and explain both model paths with diagrams.
- [x] Complete development GPU/model accuracy and end-to-end EPD/PD serving validation; record configurations and scoring boundaries.
- [x] Run regression and Python checks.
- [ ] Complete matched non-CP/conversion comparisons, upstream dependency reproducibility and community CI.
