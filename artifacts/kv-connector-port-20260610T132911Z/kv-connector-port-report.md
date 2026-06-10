# KV Connector Port Report — kv-connector-port-20260610T132911Z

## Outcome

`local_structural_ready_for_pod_smoke` — code is ported and committed on
branch `tdeshane/spyre-inference-kv-connector-port`, cross-checked against
vLLM 0.20.1 source (the version pinned by uv.lock), and all local
non-vLLM checks pass. vLLM is NOT installed locally, so the vLLM-backed
registration tests were skipped, not executed. The branch is not
runtime-verified until they pass in a pod image.

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

## vLLM 0.20.1 API review (against tag source, see command log)

- Factory `register_connector(name, module_path, class_name)` + lazy loader: matches.
- `KVConnectorBase_V1.__init__(vllm_config, role, kv_cache_config)`: matches.
- All 7 abstract methods implemented; `start_load_kv` accepts `**kwargs`;
  `get_finished`/`request_finished`/`register_kv_caches` signatures match.
- Optional methods (`handle_preemptions`, `clear_connector_metadata`,
  `get_kv_connector_kv_cache_events`) have concrete base defaults.
- Drift fixed: `handle_preemptions(kv_connector_metadata)` — the bridge
  previously passed preempted req ids (old API). Bridge updated.
- NIXL: connector imports `nixl._api` directly (guarded); it does not
  depend on vLLM's relocated `kv_connector/v1/nixl/` package.
- Timing: scheduler-side connector is created in `Scheduler.__init__`,
  after `check_and_update_config` runs registration; worker-side caches
  register via `kv_transfer_group.register_kv_caches` in the model runner,
  after `init_device` runs registration. Both seams hold for 0.20.1.

## HMA caveat for first smoke

vLLM 0.20.1 auto-disables the hybrid KV cache manager when
`--kv-transfer-config` is set unless the user explicitly enables it.
`InMemorySpyreConnector` does not subclass `SupportsHMA`, so the first pod
smoke must NOT pass `--no-disable-hybrid-kv-cache-manager`; leave HMA
flags unset and let vLLM disable it.

## Tests run (exact commands)

- `python3 -m py_compile spyre_inference/.../*.py tests/*.py` — PASS
- `uvx pytest tests/test_kv_connector_structure.py tests/test_kv_connector_registration.py -v`
  — 7 passed (structural, no vLLM needed), 1 skipped (registration module
  gated on vLLM, which is not installed locally — NOT executed)
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
