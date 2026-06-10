# Command Log — kv-connector-pod-smoke-20260610T142708Z

All paths and timestamps captured live during the run on 2026-06-10 (UTC).
The full pytest log of the gate run is appended at the end (truncated to head/tail).

Run id: `kv-connector-pod-smoke-20260610T142708Z`
Branch: `tdeshane/spyre-inference-kv-connector-port` @ `ca24124916e7fa40bb15fcde5fe3bdcc4c04571f`

## 1. Local branch hygiene

Found that `~/torch-spyre-open-work/spyre-inference` did not exist on this laptop and that no pre-existing checkout had a `tdeshane` remote pointing at the Todd fork. The `Todd-Deshane` namespace on github.com / github.ibm.com has no `spyre-inference` fork; the actual fork carrying `tdeshane/spyre-inference-kv-connector-port` is `toddllm/spyre-inference` on github.com.

```bash
mkdir -p "$HOME/torch-spyre-open-work"
cd "$HOME/torch-spyre-open-work"
git clone https://github.com/torch-spyre/spyre-inference.git spyre-inference
cd spyre-inference
git remote add tdeshane https://github.com/toddllm/spyre-inference.git
git fetch tdeshane --prune
git checkout -b tdeshane/spyre-inference-kv-connector-port \
    tdeshane/tdeshane/spyre-inference-kv-connector-port
git rev-parse --short HEAD     # → ca24124
git status --short              # → clean
```

`origin` -> `https://github.com/torch-spyre/spyre-inference.git` (upstream).
`tdeshane` -> `https://github.com/toddllm/spyre-inference.git` (Todd's fork; push target).

## 2. KUBECONFIG resolution

`KUBECONFIG` was set to a relative path `.kube/spyre-inference-testing.config`, so `oc whoami` failed from any cwd that was not `$HOME`. Resolved with absolute path:

```bash
export KUBECONFIG="$HOME/.kube/spyre-inference-testing.config"
oc whoami                       # → Todd.Deshane@ibm.com
oc whoami --show-server         # → https://api.llm-d.spyre.res.ibm.com:6443
oc -n aiu-vllm get sa llm-d-sa  # → present (187d old)
```

## 3. Run-id and artifact dir

```bash
RUN_ID="kv-connector-pod-smoke-$(date -u +%Y%m%dT%H%M%SZ)"   # → kv-connector-pod-smoke-20260610T142708Z
mkdir -p "artifacts/$RUN_ID"
touch "artifacts/$RUN_ID/command-log.md" "artifacts/$RUN_ID/pod-smoke-report.md"
```

## 4. Fresh smoke pod

Recipe from `~/spyre-inference-testing/docs/spyre-inference-runtime-candidate-usage-2026-06-09.md` ("Create Your Own Smoke Pod"), current image tag.

```bash
export NAMESPACE=aiu-vllm
export IMAGE_REF="image-registry.openshift-image-registry.svc:5000/aiu-vllm/tdeshane-spyre-inference-runtime:comms-fixed-20260610-043005"
export POD="kvconn-smoke-$(date -u +%m%d%H%M)"   # → kvconn-smoke-06101428
```

Pod manifest applied was the runbook template verbatim, with:
- `metadata.labels.purpose: spyre-inference-kvconn-smoke`
- `metadata.labels.owner: tdeshane`
- `command: ["sleep", "infinity"]`
- requests/limits `ibm.com/spyre_pf: '2'`
- `runAsUser: 1234`, `seccompProfile: RuntimeDefault`
- `serviceAccountName: llm-d-sa`, `schedulerName: spyre-scheduler`
- env: `HOME=/home/senuser`, `FLEX_COMPUTE=SENTIENT`, `FLEX_DEVICE=PF`, `AIU_SETUP_MULTI_AIU=1`
- `dev-shm` 8Gi tmpfs
- `restartPolicy: Never`

```text
pod/kvconn-smoke-06101428 created
pod/kvconn-smoke-06101428 condition met            (≈2s after apply)
NAME                    READY   STATUS    RESTARTS   AGE   IP            NODE           NOMINATED NODE   READINESS GATES
kvconn-smoke-06101428   1/1     Running   0          2s    10.130.3.78   p1-worker-63   <none>           <none>
```

## 5. Baseline import + device smoke

```bash
oc -n aiu-vllm exec "$POD" -- bash -lc '
set +u
export AIU_SETUP_MULTI_AIU=0
source /etc/profile.d/ibm-aiu-setup.sh
_setup_common_env
setup_runtime_paths 2>/dev/null || true
set -u
unset TORCH_DEVICE_BACKEND_AUTOLOAD
cd /home/senuser/spyre-inference
source .venv/bin/activate
python -c "
import importlib.metadata as md, sys
for n in [\"spyre-inference\",\"vllm\",\"torch\",\"torch-spyre\"]: print(n, md.version(n))
import torch, torch_spyre, spyre_inference, vllm
print(\"privateuse1\", torch._C._get_privateuse1_backend_name())
print(\"device_count\", torch.spyre.device_count())
"
'
```

Output:

```text
spyre-inference 0.1.dev79
vllm 0.20.1+cpu
torch 2.11.0+cpu
torch-spyre 0.0.1
privateuse1 spyre
device_count 2
```

## 6. Stage branch tree into pod (no pip / no uv)

The image's venv has no `pip` (it's uv-managed), so a literal `python -m pip install -e . --no-deps` fails with `No module named pip`. Instead, the branch tree at `ca24124` was streamed in via `git archive | oc exec`, and the gate was run from that cwd so Python's cwd-first `sys.path` resolves `import spyre_inference` to the branch tree.

```bash
cd "$HOME/torch-spyre-open-work/spyre-inference"
git archive --format=tar --prefix=spyre-inference-kvconn/ HEAD | \
  oc -n aiu-vllm exec -i "$POD" -- bash -lc \
    'mkdir -p /home/senuser/work && cd /home/senuser/work && tar xf -'
```

In-pod sanity check (cwd-first sys.path resolves to branch tree):

```bash
oc -n aiu-vllm exec "$POD" -- bash -lc '
source /home/senuser/spyre-inference/.venv/bin/activate
cd /home/senuser/work/spyre-inference-kvconn
python -c "import spyre_inference; print(spyre_inference.__file__)"
'
# → /home/senuser/work/spyre-inference-kvconn/spyre_inference/__init__.py
```

Note: `python -m pip install -e . --no-deps` was attempted and produced `/home/senuser/spyre-inference/.venv/bin/python: No module named pip` — confirmation that the venv is uv-managed. No `uv sync` / no rebuild was performed; running pytest from the branch cwd is sufficient and avoids modifying the image's installed package set.

## 7. Required gate (clean)

```bash
oc -n aiu-vllm exec "$POD" -- bash -lc '
set +u; source /etc/profile.d/ibm-aiu-setup.sh; _setup_common_env; setup_runtime_paths 2>/dev/null || true; set -u
unset TORCH_DEVICE_BACKEND_AUTOLOAD
source /home/senuser/spyre-inference/.venv/bin/activate
cd /home/senuser/work/spyre-inference-kvconn
python -m pytest tests/test_kv_connector_structure.py \
                 tests/test_kv_connector_registration.py \
                 -v -p no:spyre_inference_test
'
```

Result:

```text
collected 10 items

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
EXIT=0
```

The two warnings are SwigPyPacked / SwigPyObject `DeprecationWarning`s emitted by `<frozen importlib._bootstrap>` — they originate from the runtime image's torch / numactl bindings, not from connector code.

## 8. Same gate WITHOUT the plugin disable (for completeness)

The first invocation didn't pass `-p no:spyre_inference_test`. Result:

```text
================== 2 failed, 38 passed, 437 skipped in 14.42s ==================
EXIT=1
```

The 2 failures are NOT in the connector code path:

```text
FAILED tests/test_spyre_attn.py::test_causal_backend_correctness[1-meta-llama/Meta-Llama-3-8B-single_decode]
FAILED tests/test_spyre_attn.py::test_causal_backend_correctness[1-meta-llama/Meta-Llama-3-8B-single_prefill]
  OSError: You are trying to access a gated repo.
  Cannot access gated repo for url https://huggingface.co/meta-llama/Meta-Llama-3-8B/resolve/main/config.json.
```

Root cause: the image-installed pytest plugin `spyre-testing-plugin` (entry-point `spyre_inference_test` -> `spyre_testing_plugin.pytest_plugin`) auto-injects an upstream-vLLM act-and-mul fixture set and the `test_causal_backend_correctness` parametrizations on every collection, regardless of which test files I name on the command line. Disabling the plugin produces the clean `10 passed` result above. The 38 unrelated passes (act/mul) and 437 skips stay green underneath the gating issue, but they are out of scope for this gate.

## 9. Connector import / visibility smoke

```bash
oc -n aiu-vllm exec "$POD" -- bash -lc '
set +u; source /etc/profile.d/ibm-aiu-setup.sh; _setup_common_env; setup_runtime_paths 2>/dev/null || true; set -u
unset TORCH_DEVICE_BACKEND_AUTOLOAD
source /home/senuser/spyre-inference/.venv/bin/activate
cd /home/senuser/work/spyre-inference-kvconn
python <<PY
# (full script in pod-smoke-report.md "Connector import / visibility smoke" section)
PY
'
```

Key output lines:

```text
spyre_inference at: /home/senuser/work/spyre-inference-kvconn/spyre_inference/__init__.py
INFO ... [__init__.py:69] Registered InMemorySpyreConnector
InMemorySpyreConnector registered: OK
second call OK
connector imported under nixl block: OK
NIXL_AVAILABLE = False
class: <class 'spyre_inference.distributed.kv_transfer.kv_connector.v1.inmemory_spyre_connector.InMemorySpyreConnector'>
registered factory: <function KVConnectorFactory.register_connector.<locals>.loader at 0x...>
```

## 10. Branch state at end of run

```bash
git rev-parse HEAD                              # ca24124916e7fa40bb15fcde5fe3bdcc4c04571f
git status --short                              # only untracked: artifacts/<RUN_ID>/
```

## 11. Follow-up: KVTransferConfig + class resolution + create_connector

Same pod, same branch staging.

### 11.1 Inventory (nixl, UCX/NIXL libs, connector class shape, KVTransferConfig signature)

```bash
oc -n aiu-vllm exec "$POD" -- bash -lc '... <inventory script> ...'
```

Result:

```text
nixl python module:    spec: None / ModuleNotFoundError: No module named 'nixl'
libucp/libucs/libnixl/libucm in ldconfig: none

inmemory_spyre_connector.NIXL_AVAILABLE: False
class MRO: ['InMemorySpyreConnector', 'KVConnectorBase_V1', 'ABC', 'object']

vllm.config.KVTransferConfig params:
  ['kv_connector', 'engine_id', 'kv_buffer_device', 'kv_buffer_size', 'kv_role',
   'kv_rank', 'kv_parallel_size', 'kv_ip', 'kv_port', 'kv_connector_extra_config',
   'kv_connector_module_path', 'enable_permute_local_kv', 'kv_load_failure_policy']
KVTransferConfig src: vllm/config/kv_transfer.py
```

### 11.2 KVTransferConfig build smoke (six probes)

Script: written to `/tmp/kv_kvconfig_smoke.py` then `oc cp` into pod and run as `python /tmp/kv_kvconfig_smoke.py`. Six probes:
1. `register_kv_connector()` and verify registry contains `InMemorySpyreConnector`.
2. `KVTransferConfig(kv_connector="InMemorySpyreConnector", kv_role="kv_producer")`.
3. Same with `kv_role="kv_consumer"`.
4. Same with `kv_role="kv_both"`.
5. With explicit `kv_connector_module_path="spyre_inference.distributed.kv_transfer.kv_connector.v1.inmemory_spyre_connector"`.
6. **Negative control:** `kv_connector="DefinitelyNotARealConnectorXYZ"`.

**Important runtime observation about `oc exec` + python script files.**
The first attempt ran `python /tmp/kv_kvconfig_smoke.py` after a `cd /home/senuser/work/spyre-inference-kvconn` in the wrapper exec line. It failed with `AttributeError: module 'spyre_inference' has no attribute 'register_kv_connector'`. Cause: `python script.py` sets `sys.path[0]` to the **script's** directory (`/tmp`), not the cwd. So the branch tree is invisible and the image-installed `spyre-inference 0.1.dev79` (which lacks `register_kv_connector`) gets imported. Resolution: export `PYTHONPATH=/home/senuser/work/spyre-inference-kvconn:${PYTHONPATH:-}` before running. **The pytest gate didn't hit this because the branch's pyproject pins `pythonpath = ["."]` for pytest specifically.** Anyone running an ad-hoc `python` (or starting `vllm serve`) needs `PYTHONPATH` (or a real install).

Successful run output (with `PYTHONPATH` set), trimmed:

```text
=== 1. register connector ===
INFO ... [__init__.py:69] Registered InMemorySpyreConnector
InMemorySpyreConnector in registry: True

=== 2/3/4. KVTransferConfig by name (producer/consumer/both) ===
BUILT producer: KVTransferConfig(kv_connector='InMemorySpyreConnector', engine_id='62d16b...', kv_buffer_device='cpu', kv_buffer_size=1000000000.0, kv_role='kv_producer', kv_rank=None, kv_parallel_size=1, kv_ip='127.0.0.1', kv_port=14579, kv_connector_extra_config={}, kv_connector_module_path=None, enable_permute_local_kv=False, kv_load_failure_policy='fail')
BUILT consumer: ... kv_role='kv_consumer' ...
BUILT both:     ... kv_role='kv_both' ...

=== 5. with explicit kv_connector_module_path ===
BUILT module-path: KVTransferConfig(...
    kv_connector_module_path='spyre_inference.distributed.kv_transfer.kv_connector.v1.inmemory_spyre_connector', ...)

=== 6. bogus name ===
BUILT bogus (unexpected!): KVTransferConfig(kv_connector='DefinitelyNotARealConnectorXYZ', ...)

=== 8. KVConnectorFactory public surface ===
public methods: ['create_connector', 'get_connector_class', 'get_connector_class_by_name', 'register_connector']
```

Probe 6 confirms `KVTransferConfig.__init__` does not validate `kv_connector` against the registry; it accepts any string. The actual gate is at class resolution / instantiation (probes in 11.3 / 11.4).

### 11.3 Class-resolution probe (no instantiation)

Script: `/tmp/kv_resolver_probe.py`. Tests `KVConnectorFactory.get_connector_class_by_name(...)` and `KVConnectorFactory.get_connector_class(KVTransferConfig)` for both positive (registered) and negative (unregistered) cases.

```text
=== get_connector_class_by_name (no-instantiate resolver) ===
signature: (connector_name: str) -> type[vllm.distributed.kv_transfer.kv_connector.v1.base.KVConnectorBase_V1]
positive: OK class: <class 'spyre_inference.distributed.kv_transfer.kv_connector.v1.inmemory_spyre_connector.InMemorySpyreConnector'>
   MRO: ['InMemorySpyreConnector', 'KVConnectorBase_V1', 'ABC', 'object']
negative ("DefinitelyNotARealConnectorXYZ"): ValueError "Connector '...' is not registered."

=== get_connector_class via KVTransferConfig (production path) ===
positive: OK class via cfg: <class 'spyre_inference.distributed.kv_transfer.kv_connector.v1.inmemory_spyre_connector.InMemorySpyreConnector'>
   same as registry path: True
negative via cfg: ValueError "Unsupported connector type: DefinitelyNotARealConnectorXYZ"

registered loader: <function KVConnectorFactory.register_connector.<locals>.loader at 0x...>
loader source file: vllm/distributed/kv_transfer/kv_connector/factory.py
```

Both positive paths return the branch class file (verified by `__module__` matching `spyre_inference.distributed.kv_transfer.kv_connector.v1.inmemory_spyre_connector`). Both negative paths error cleanly.

### 11.4 `create_connector` instantiation — heap/host_memory bug found

Script: `/tmp/kv_instantiate_probe.py`. Built a minimal `VllmConfig` over `facebook/opt-125m` with `kv_transfer_config=KVTransferConfig(kv_connector="InMemorySpyreConnector", kv_role="kv_producer")`, then called `KVConnectorFactory.create_connector(vc, "kv_producer")`.

vLLM-side success markers (matching task expectations):

```text
INFO ... [vllm.py:1293] Turning off hybrid kv cache manager because `--kv-transfer-config` is set.
INFO ... [factory.py:64] Creating v1 connector with name: InMemorySpyreConnector and engine_id: ...
WARNING [base.py:189] Initializing KVConnectorBase_V1.
WARNING [base.py:201] KVConnectorBase_V1 initialized without kv_cache_config.
```

Then the connector raised:

```text
ValueError: Unknown Spyre KV store backend 'heap'.
Supported backends: host_memory, serialized_host_memory, serialized_shared_memory,
                    serialized_shared_memory_service, serialized_uds_process_store
```

Trace:

```text
factory.py:82  return connector_cls(config, role, kv_cache_config)
inmemory_spyre_connector.py:137  self._store = store if store is not None else get_global_store()
inmemory_spyre_connector.py:96   _GLOBAL_STORE = _build_configured_store(store_backend_name)
inmemory_spyre_connector.py:85   return build_spyre_kv_store_backend(backend_name, ...)
metadata.py:1326                  raise ValueError(...)
```

Root cause and fix options are detailed in pod-smoke-report.md "Layer 3: create_connector — a real bug found". Summary:

- `spyre_inference/envs.py:28,90` defaults `VLLM_SPYRE_KV_STORE_BACKEND` to `"heap"`.
- `spyre_inference/distributed/kv_transfer/kv_connector/v1/metadata.py:1306-1312` (`_STORE_BACKEND_TYPES`) does not include `"heap"`.
- Both files were introduced in the same commit `45aa658`.
- No tests / probes / examples / docs reference `"heap"` as a backend name; only `envs.py` does (default + comment + comment).

Fix not yet applied — awaiting choice between:
- (A) Add `"heap"` -> `HostMemoryKVStoreBackend` to `_STORE_BACKEND_TYPES`.
- (B) Change `envs.py` default from `"heap"` to `"host_memory"` and update the comment.

## 12. Cleanup (deferred)

```bash
oc -n aiu-vllm delete pod kvconn-smoke-06101428 --ignore-not-found
```

Pod was left running for reviewer follow-up.
