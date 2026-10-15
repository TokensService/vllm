# Tokens Native P/D Deployment (MooncakeConnector)

Deployment strategy registry for Tokens vLLM serving. All strategies live in
a single file, `model_arch.yaml`, keyed by `arch_name`.

## Naming rule

```text
<model>-<quant>-<platform>-<topology>
```

e.g. `GLM-5.2-W4A4C8-EMS-3P1D` = GLM-5.2 model, W4A4C8 quantization, EMS
platform, 3 Prefill + 1 Decode instances. The topology suffix is `<n>P<m>D`
for disaggregated strategies, `<n>P` for standalone-only ones.

Currently registered:

- `GLM-5.2-W4A4C8-MG40-BNT3-1P1D` — GLM-5.2 NVFP4 on MG40-BNT3: 1 Prefill
  pod (TP1 x PP4, :8010) + 1 Decode pod (TP1 x DP4 over Ray, :8020), KV
  transfer over MooncakeConnector, plus a `standalone` single-instance
  baseline mode (:8000).

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
eval "$(./render_serve_config.py GLM-5.2-W4A4C8-MG40-BNT3-1P1D prefill --env)"
vllm serve --config <(./render_serve_config.py GLM-5.2-W4A4C8-MG40-BNT3-1P1D prefill)

# Decode (PD mode)
eval "$(./render_serve_config.py GLM-5.2-W4A4C8-MG40-BNT3-1P1D decode --env)"
vllm serve --config <(./render_serve_config.py GLM-5.2-W4A4C8-MG40-BNT3-1P1D decode)

# Standalone baseline
eval "$(./render_serve_config.py GLM-5.2-W4A4C8-MG40-BNT3-1P1D standalone --env)"
vllm serve --config <(./render_serve_config.py GLM-5.2-W4A4C8-MG40-BNT3-1P1D standalone)
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
