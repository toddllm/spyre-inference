# Spyre KV Connector Port Status

Ported the `InMemorySpyreConnector` surface (in-memory + NIXL KV transfer for
disaggregated prefill/decode) from the llm-d-on-spyre working baseline
(`abca9a0`, plus dynamic-prefill-IP and registration fixes through `3db6d57`)
into `spyre_inference`.

## What was ported

- `spyre_inference/distributed/kv_transfer/kv_connector/v1/` — connector,
  metadata/store backends, heap-KV accessors and clients, UDS transport,
  persistent KV service client. Mechanical rename `vllm_spyre` →
  `spyre_inference`; env var names keep the `VLLM_SPYRE_*` prefix so existing
  pod scripts work unchanged.
- `spyre_inference/envs.py` — 16 KV/NIXL/heap env vars.
- `spyre_inference.register_kv_connector()` — vLLM `KVConnectorFactory`
  registration, called from `TorchSpyrePlatform.check_and_update_config`
  (engine/scheduler process) and `TorchSpyreWorker.init_device` (workers).
  No vendored vLLM changes.
- `spyre_inference/v1/worker/spyre_kv_connector_bridge.py` — worker-side
  bridge; present but not wired (vLLM's standard connector flow expected to
  apply via CPUWorker).

## Status: local_structural_ready_for_pod_smoke

- Cross-checked against vLLM 0.20.1 tag source (uv.lock pin): factory,
  `KVConnectorBase_V1` abstract/optional methods, scheduler and worker
  registration timing, worker KV-cache registration path all match. One
  drift fixed (`handle_preemptions` now receives connector metadata).
- All modules compile (`py_compile`) and pass ruff check/format.
- `tests/test_kv_connector_structure.py` — 7 structural tests pass locally
  without vLLM (module paths, package files, env var consistency, no
  `vllm_spyre` leftovers, registration hooks).
- `tests/test_kv_connector_registration.py` is import-gated on vLLM and was
  SKIPPED locally (vLLM not installed). Registration is not yet verified.

## Needs pod testing

- First command: `python -m pytest tests/test_kv_connector_registration.py -v`
  in a pod image with vLLM installed — must pass before further work.
- Two-pod prefill/decode smoke with `--kv-transfer-config
  '{"kv_connector":"InMemorySpyreConnector",...}'` and
  `VLLM_SPYRE_KV_ROLE=kv_producer|kv_consumer`.
- HMA caveat: vLLM 0.20.1 auto-disables the hybrid KV cache manager when
  `kv_transfer_config` is set. Do NOT pass
  `--no-disable-hybrid-kv-cache-manager` on the first smoke; the connector
  does not implement `SupportsHMA`.
- Heap-KV / NIXL paths require AIU hardware; NIXL gracefully disables when
  `nixl` is not importable.

Details: `artifacts/kv-connector-port-20260610T132911Z/`.
