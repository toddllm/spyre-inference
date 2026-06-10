# KV Connector Pod Smoke Report

- **Run ID:** `kv-connector-pod-smoke-20260610T142708Z`
- **Date (UTC):** 2026-06-10
- **Outcome:** **`pod_registration_passed`**
- **Branch:** `tdeshane/spyre-inference-kv-connector-port`
- **Commit:** `ca24124916e7fa40bb15fcde5fe3bdcc4c04571f`
- **Branch tracking:** `tdeshane/tdeshane/spyre-inference-kv-connector-port`
  (remote `tdeshane` -> `https://github.com/toddllm/spyre-inference.git`)
- **Code changes on branch this pass:** none (validation-only)

## Pod / Image

| Field | Value |
|---|---|
| Cluster | `https://api.llm-d.spyre.res.ibm.com:6443` |
| Namespace | `aiu-vllm` |
| Pod | `kvconn-smoke-06101428` (Todd-owned, fresh, `restartPolicy: Never`) |
| Service account | `llm-d-sa` |
| Scheduler | `spyre-scheduler` |
| Node | `p1-worker-63` |
| Image | `image-registry.openshift-image-registry.svc:5000/aiu-vllm/tdeshane-spyre-inference-runtime:comms-fixed-20260610-043005` |
| Image digest | `sha256:3210427b53d72bc1ed761625893a26eed2c07089375ce5a64be2cd7f4efe53ce` |
| Resources | `ibm.com/spyre_pf: '2'` (req+lim) |
| `dev-shm` | 8Gi tmpfs |

Source of pod recipe: `~/spyre-inference-testing/docs/spyre-inference-runtime-candidate-usage-2026-06-09.md` ("Create Your Own Smoke Pod"), current fixed-image tag.

## Baseline image acceptance (runbook smoke)

```text
spyre-inference 0.1.dev79
vllm 0.20.1+cpu
torch 2.11.0+cpu
torch-spyre 0.0.1
privateuse1 spyre
hasattr torch.spyre True
device_count 2
```

This is the image's baked-in `spyre-inference 0.1.dev79`, before the branch tree was layered in.

## Branch staging into the pod

The fresh pod has a uv-managed `.venv` with no `pip` and a baked-in `spyre-inference 0.1.dev79`. Rather than fighting that, the branch tree at `ca24124` was staged via `git archive | oc exec` into `/home/senuser/work/spyre-inference-kvconn` and pytest was run from that cwd. Python's cwd-first `sys.path` resolves `import spyre_inference` to the branch tree (verified: `python -c 'import spyre_inference; print(spyre_inference.__file__)'` returns `/home/senuser/work/spyre-inference-kvconn/spyre_inference/__init__.py`), so the gate runs against the branch sources without a `pip install -e .` step. **No dependency rebuild, no `uv sync`, no torch-spyre rebuild was performed.**

## Required gate: `tests/test_kv_connector_registration.py` (and structure)

Command:

```bash
python -m pytest tests/test_kv_connector_structure.py \
                 tests/test_kv_connector_registration.py \
                 -v -p no:spyre_inference_test
```

`-p no:spyre_inference_test` disables the image-installed `spyre-testing-plugin`, which auto-injects the upstream-vLLM act-and-mul fixture set on every collection (~437 skipped + 38 passed + 2 unrelated FAILs blocked on the gated `meta-llama/Meta-Llama-3-8B` HuggingFace repo from the pod). With the plugin disabled, the gate result is clean:

```text
tests/test_kv_connector_structure.py::test_connector_module_path_resolves_to_file PASSED [ 10%]
tests/test_kv_connector_structure.py::test_registration_uses_existing_module_path PASSED [ 20%]
tests/test_kv_connector_structure.py::test_all_connector_packages_have_init_files PASSED [ 30%]
tests/test_kv_connector_structure.py::test_connector_defines_class_and_nixl_guard PASSED [ 40%]
tests/test_kv_connector_structure.py::test_no_vllm_spyre_references PASSED [ 50%]
tests/test_kv_connector_structure.py::test_kv_env_vars_declared_consistently PASSED [ 60%]
tests/test_kv_connector_structure.py::test_registration_hooks_present PASSED [ 70%]
tests/test_kv_connector_registration.py::test_register_kv_connector_with_factory PASSED [ 80%]
tests/test_kv_connector_registration.py::test_connector_module_imports PASSED [ 90%]
tests/test_kv_connector_registration.py::test_connector_imports_without_nixl PASSED [100%]

======================== 10 passed, 2 warnings in 5.26s ========================
```

**Result:** 10 passed, 0 failed, 0 skipped — 7 structure + 3 registration tests. Run time 5.26s.

For completeness, the run with the upstream plugin enabled produced `2 failed, 38 passed, 437 skipped`. The two failures are NOT in the KV connector code path; they are `tests/test_spyre_attn.py::test_causal_backend_correctness[1-meta-llama/Meta-Llama-3-8B-{single_decode,single_prefill}]`, both blocked on `OSError: You are trying to access a gated repo. ... meta-llama/Meta-Llama-3-8B`. The pod has no Hugging Face credential for that gated repo. This is unrelated to the KV connector port.

## Connector import / visibility smoke

In addition to the pytest gate, ran a freestanding smoke that exercises the three things the task asks for:

```text
=== before register ===
spyre_inference at: /home/senuser/work/spyre-inference-kvconn/spyre_inference/__init__.py
registry before: ['DecodeBenchConnector', 'ExampleConnector', 'ExampleHiddenStatesConnector',
                  'FlexKVConnectorV1', 'HF3FSKVConnector', 'LMCacheConnectorV1',
                  'LMCacheMPConnector', 'MoRIIOConnector', 'MooncakeConnector',
                  'MultiConnector', 'NixlConnector', 'OffloadingConnector',
                  'P2pNcclConnector', 'SimpleCPUOffloadConnector']

=== call register_kv_connector() ===
INFO ... [__init__.py:69] Registered InMemorySpyreConnector
registry after: [..., 'InMemorySpyreConnector', ...]
InMemorySpyreConnector registered: OK

=== idempotent? ===
second call OK

=== nixl-blocked import path (a clean fresh import) ===
connector imported under nixl block: OK
NIXL_AVAILABLE = False
class: <class 'spyre_inference.distributed.kv_transfer.kv_connector.v1.inmemory_spyre_connector.InMemorySpyreConnector'>

=== vLLM sees the registered name ===
registered factory: <function KVConnectorFactory.register_connector.<locals>.loader at 0x...>
```

Three required claims verified:
- `spyre_inference.register_kv_connector()` registers `InMemorySpyreConnector` in `vllm.distributed.kv_transfer.kv_connector.factory.KVConnectorFactory._registry` (and is idempotent).
- The connector module `spyre_inference.distributed.kv_transfer.kv_connector.v1.inmemory_spyre_connector` imports cleanly with `nixl` blocked at `__import__`; `NIXL_AVAILABLE == False` and the class is still exposed.
- vLLM resolves the registered name through its factory loader (post-registration the registry entry is a `KVConnectorFactory.register_connector.<locals>.loader` function).

## Code changes

None. The branch was structurally and runtime-validated against the runtime image as-is. No edits were applied to `spyre_inference/`, `tests/`, `pyproject.toml`, or anywhere else on the branch.

## Explicit non-coverage

This pass did **not** run:
- a two-pod prefill/decode smoke,
- `vllm serve` with `kv_transfer_config` set,
- any actual KV transfer between pods,
- any model-load or completion-generation,
- the runbook's two-rank `barrier` + `broadcast` smoke (not part of the KV connector gate; was already validated for this image in the runbook's acceptance run on 2026-06-10),
- a `pip install -e . --no-deps` (it failed because the venv has no `pip`; resolved by running pytest from the branch cwd, see "Branch staging" above).

The first registration gate passing does **not** imply prefill/decode works.

## Smallest next test

The smallest credible next gate is a two-pod prefill/decode smoke with the connector wired in. The required pre-step before that is to enumerate the inputs missing on this image, since the runbook's PD/NIXL Integration Requirements table flags several gaps:

1. `nixl` Python module — not installed on `comms-fixed-20260610-043005`. The connector imports under the `NIXL_AVAILABLE = False` guard, but a real PD smoke needs `nixl` runtime.
2. UCX userspace + NIXL system libraries (`libucp*`, `libucs*`, `libnixl*`) — none on this image.
3. A second Todd-owned pod on the same `aiu-vllm` namespace + `spyre-scheduler`, plus a service or ClusterIP that lets the two pods reach each other on the connector's wire ports.
4. A connector-config side: a vLLM `kv_transfer_config` referencing `InMemorySpyreConnector` by name on both prefill and decode sides.

A reasonable next command (single-pod, half of the path) once `nixl` and UCX/NIXL are added to the image:

```bash
oc -n aiu-vllm exec "$POD" -- bash -lc '
  source /home/senuser/spyre-inference/.venv/bin/activate
  cd /home/senuser/work/spyre-inference-kvconn
  python -c "
from vllm.config import KVTransferConfig
cfg = KVTransferConfig(kv_connector=\"InMemorySpyreConnector\", kv_role=\"kv_producer\")
print(\"KVTransferConfig built:\", cfg)
import spyre_inference
spyre_inference.register_kv_connector()
from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory
print(\"registered:\", \"InMemorySpyreConnector\" in KVConnectorFactory._registry)
"
'
```

This is config-only and doesn't yet stand up a server; it confirms `KVTransferConfig` accepts the registered name on this stack. A real two-pod smoke is a separate runbook.

## Pod cleanup

The pod `kvconn-smoke-06101428` is being **left running** until reviewer confirms the report. Cleanup command for after sign-off:

```bash
oc -n aiu-vllm delete pod kvconn-smoke-06101428 --ignore-not-found
```
