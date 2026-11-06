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
