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
