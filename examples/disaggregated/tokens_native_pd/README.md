# Tokens Native P/D Deployment (MooncakeConnector)

Serve configs for the Tokens native disaggregated prefill/decode deployment of
`GLM-5.2-NVFP4-W4A4C8-MG40-BNT3`, expressed as `vllm serve --config` YAML files
(see `FlexibleArgumentParser` in `vllm/utils/argparse_utils.py`; precedence is
CLI > config file > defaults, so any value here can be overridden per launch).

## Files

- `prefill.yaml` — Prefill role, TP1 x PP4, `MooncakeConnector` as `kv_producer`,
  listens on `:8010`.
- `decode.yaml` — Decode role, TP1 x DP4 over Ray, `MooncakeConnector` as
  `kv_consumer`, listens on `:8020`.
- `standalone.yaml` — single-instance baseline (Prefill shape, no
  `kv-transfer-config`), serving directly on `:8000`.

## Usage

```bash
# Prefill (PD mode)
VLLM_PLUGINS="" VLLM_PP_LAYER_PARTITION=22,20,20,16 \
XCCL_TURBO_PREFILL_MLA_HEAD_PADDING=64 VLLM_MOONCAKE_BOOTSTRAP_PORT=8998 \
  vllm serve --config prefill.yaml

# Decode (PD mode)
VLLM_PLUGINS="" RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1 \
  vllm serve --config decode.yaml

# Standalone baseline
VLLM_PLUGINS="" VLLM_PP_LAYER_PARTITION=22,20,20,16 \
XCCL_TURBO_PREFILL_MLA_HEAD_PADDING=64 \
  vllm serve --config standalone.yaml
```

## What is intentionally not in these files

`--config` covers CLI/engine arguments only. The rest of the deployment stays
in the orchestration layer:

- Process env vars: `VLLM_PLUGINS`, `VLLM_PP_LAYER_PARTITION`,
  `XCCL_TURBO_PREFILL_MLA_HEAD_PADDING`, `VLLM_MOONCAKE_BOOTSTRAP_PORT`,
  `RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES` (listed above per role).
- Kubernetes Pod spec: namespace, node pinning, GPU limits, image,
  `runtimeClassName`, `hostNetwork`, `/dev/infiniband` and `/dev/shm` volumes.
- Decode-only MoERunner fix files (`vllm_native_moefix.py` /
  `zz_vllm_native_moefix.pth`) mounted into dist-packages.
- The Mooncake P/D proxy (`examples/disaggregated/mooncake_connector/mooncake_connector_proxy.py`),
  which is a plain Python process, not a `vllm serve` instance: it takes
  `--prefill http://127.0.0.1:8010 8998 --decode http://127.0.0.1:8020` and
  exposes the public `:8000`.

The same recipe is embedded as generated Pod specs in the tokens-worktable
pipeline scripts (`TokensService/tokens-worktable`,
`projects/pipeline/scripts/vllm-native-*.sh`); keep those and these files in
sync when tuning the deployment.

## Verifying the PD path

After the proxy is rebuilt onto ready Prefill/Decode pods, run a generation
request through `:8000` and check `/metrics` on the Decode pod: the
`external_kv_transfer` counters must be > 0, proving KV actually flowed over
Mooncake instead of being recomputed.
