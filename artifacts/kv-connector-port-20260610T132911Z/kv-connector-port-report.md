# KV Connector Port Report — kv-connector-port-20260610T132911Z

## Outcome

`implementation_branch_ready` — code is ported, committed on branch
`tdeshane/spyre-inference-kv-connector-port`, and all local compile, lint,
and registration-gated tests pass (vLLM-dependent tests skip locally and
are queued for the cluster image).

## Source and target

- Source baseline: `llm-d-on-spyre` commit `abca9a0`, plus the four fixes on
  `origin/pd-disagg-with-nixl-transfer` through `3db6d57` (connector
  registration timing, import-cycle fix, dynamic prefill IP via
  `SpyreRemoteMeta`/`kv_transfer_params`, metadata changes).
- Target: `spyre-inference` (torch-spyre stack), branch
  `tdeshane/spyre-inference-kv-connector-port` from `origin/main` `54fafa0`,
  push remote `tdeshane` (toddllm fork).

## Files changed

New `spyre_inference/distributed/kv_transfer/kv_connector/v1/`:
`inmemory_spyre_connector.py`, `metadata.py`, `heap_kv_accessor.py`,
`heap_kv_helper.py`, `heap_kv_helper_client.py`, `heap_kv_inprocess_client.py`,
`local_transport_store.py`, `persistent_kv_service.py`, package `__init__.py`s.
New `spyre_inference/v1/worker/spyre_kv_connector_bridge.py`.
Edited: `spyre_inference/envs.py` (16 `VLLM_SPYRE_*` KV vars, names kept for
pod-script compatibility), `spyre_inference/__init__.py`
(`register_kv_connector()`), `spyre_inference/platform.py`
(scheduler-side registration in `check_and_update_config`),
`spyre_inference/v1/worker/spyre_worker.py` (worker-side registration in
`init_device`), `pyproject.toml` (E501 ignore for ported tree),
`tests/test_kv_connector_registration.py` (new).

## Registration design (replaces vendored-vLLM fixes)

The source repo registered the connector by patching vLLM's `EngineCore`
(import-cycle workaround). Here, registration is adapter-level only:
`check_and_update_config` runs after `vllm.config` is fully loaded and
before the scheduler builds connectors; `init_device` covers worker
processes. No vendored vLLM files are modified.

## Tests run (exact commands)

- `python3 -m py_compile spyre_inference/.../*.py tests/test_kv_connector_registration.py` — PASS
- `uvx pytest tests/test_kv_connector_registration.py -q` — 1 skipped (no vLLM locally; module is import-gated)
- `uvx ruff check <all changed files>` — PASS; `uvx ruff format --check` — PASS

## Unresolved mappings / assumptions

- CLI `--nixl-remote-ip`: spyre-inference does not own CLI parsing; the env
  var `VLLM_SPYRE_NIXL_REMOTE_IP` plus per-request `kv_transfer_params`
  (routing-proxy path) replace it. If a CLI flag is required, the upstream
  surface is vLLM `entrypoints/openai/cli_args.py` + `engine/arg_utils.py`.
- Worker KV-cache registration relies on vLLM's standard
  `register_kv_caches` flow through `CPUWorker`/model runner; the ported
  `SpyreKVConnectorBridge` is available but not wired into
  `TorchSpyreWorker.execute_model` (current model runner is vLLM's CPU
  runner; the bridge is only needed if the standard flow proves
  insufficient on AIU).
- Heap-KV paths assume sendnn perfdsc dump layout identical to the source.
- No runtime verification on AIU cards in this pass.

## Smallest next cluster test

In a pod with vLLM + spyre-inference installed:
`python -m pytest tests/test_kv_connector_registration.py -v` — proves
factory registration, connector import, and no-NIXL import safety.
Then a two-pod smoke: prefill with `VLLM_SPYRE_KV_ROLE=kv_producer` and
decode with `VLLM_SPYRE_KV_ROLE=kv_consumer`,
`--kv-transfer-config '{"kv_connector":"InMemorySpyreConnector","kv_role":...}'`.
