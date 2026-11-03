# Tokens Native P/D Deployment (MooncakeConnector)

Deployment strategy registry for Tokens vLLM serving. All strategies live in
a single file, `model_arch.yaml`, keyed by `arch_name`.

## Naming rule

```text
<model>-<quant>-<platform>-<topology>
```

e.g. `GLM-5.3-NVFP4-NVIDIA-PP2-1P1D-6G` = GLM-5.3 model, NVFP4 quantization,
NVIDIA platform, 1 Prefill + 1 Decode instance. The topology suffix is `<n>P<m>D`
for disaggregated strategies, `<n>P` for standalone-only ones.

Currently registered:

- `GLM-5.3-W4A4C8-MG46-BNT3-2P1D-8G` — GLM-5.3 W4A4C8 (NVFP4 + FP8) on
  MG46-BNT3: 2 Prefill pods (TP1 x PP2, ports from :8010, assigned by the
  deploy layer) + 1 Decode pod (TP1 x DP4 over Ray, :8020), 8 GPUs in total,
  KV transfer over MooncakeConnector, plus a `standalone` single-instance
  baseline mode (:8000).
- `GLM-5.3-NVFP4-NVIDIA-*` — GLM-5.3 native NVFP4, several GPU budgets/topologies:
  `PP2-1P1D-6G`, `BNT3-1P1D-8G` (P TP1 x DP4), `BNT3-2P1D-8G` (two Prefills),
  `BNT3-1P1D-PP4-8G` (P TP1 x PP4).

## Strategy layout

Each strategy holds per-role sections (`prefill` / `decode` / `standalone`):

- `engine` — engine arguments compatible with `vllm serve --config`
  (precedence is CLI > config file > defaults, so any value can be
  overridden per launch);
- `env` — process environment variables, which `--config` does not cover.

## Usage

Extract one role with `render_serve_config.py`, which turns the role's
`engine` section into a `--config`-ready YAML and its `env` section into
shell exports:

```bash
# Prefill (PD mode)
eval "$(./render_serve_config.py GLM-5.3-NVFP4-NVIDIA-PP2-1P1D-6G prefill --env)"
vllm serve --config <(./render_serve_config.py GLM-5.3-NVFP4-NVIDIA-PP2-1P1D-6G prefill)

# Decode (PD mode)
eval "$(./render_serve_config.py GLM-5.3-NVFP4-NVIDIA-PP2-1P1D-6G decode --env)"
vllm serve --config <(./render_serve_config.py GLM-5.3-NVFP4-NVIDIA-PP2-1P1D-6G decode)

# Standalone baseline
eval "$(./render_serve_config.py GLM-5.3-NVFP4-NVIDIA-PP2-1P1D-6G standalone --env)"
vllm serve --config <(./render_serve_config.py GLM-5.3-NVFP4-NVIDIA-PP2-1P1D-6G standalone)
```

## What is intentionally not in model_arch.yaml

Only per-process engine args and env belong to a strategy. The rest of the
deployment stays in the orchestration layer:

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
`projects/pipeline/scripts/vllm-native-*.sh`); keep those and this registry
in sync when tuning the deployment.

## Verifying the PD path

After the proxy is rebuilt onto ready Prefill/Decode pods, run a generation
request through `:8000` and check `/metrics` on the Decode pod: the
`external_kv_transfer` counters must be > 0, proving KV actually flowed over
Mooncake instead of being recomputed.
