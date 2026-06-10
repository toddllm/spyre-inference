# Paged KV Connector Implementation Report

Run: 20260610T145727Z (UTC)

## Terminal outcome

`paged_connector_implemented_and_structurally_verified`

Local structural and unit verification passed in full. No AIU runtime was
executed; no AIU performance or runtime success is claimed.

## Target branch and commits

- Repo: `spyre-inference`, branch `tdeshane/spyre-inference-paged-kv-connector`
- Commit before: `ca24124` ("Tighten KV connector port for pod testing")
- Cherry-pick: `ed6fd10` (":recycle: type the kv cache", from origin/pr/242
  head `3907470a`, author Joe Runde)
- Commit after: see `git log` tail of this branch (paged connector commit on
  top of `ed6fd10`)

## Source refs inspected

| Ref | Commit | Use |
| --- | --- | --- |
| origin/pr/242 | 3907470a | Typed `SpyrePagedKVCache` — cherry-picked (3 trivial conflicts in imports/signature resolved to PR side) |
| origin/pr/246 | f8adabd7 | Opaque attention wrapping; no cache-shape impact, nothing ported |
| origin/pr/249 | 558e421 | `convert()` rework; no cache placement/dtype impact, nothing ported |
| origin/pr/200 | a1263123 | Stale paged varlen example; reference only |
| torch-spyre origin/main | bb47f1c8 | No indirect access on main; whole-page copies only |
| torch-spyre origin/pr/2178 | d3068f07 | Indirect gather path; not needed for whole-page connector |
| torch-spyre origin/pr/2608 | 23137c51 | Indirect detection only; forward-compat note |
| torch-spyre origin/pr/2363 | 61d95d8b | Flash direction; prefer 128-aligned blocks later |

No torch-spyre source was vendored.

## Files changed

- `spyre_inference/v1/attention/backends/spyre_attn.py` — `SpyrePagedKVCache`
  NamedTuple + typed `forward` (cherry-pick)
- `spyre_inference/v1/worker/spyre_model_runner.py` — allocate
  `SpyrePagedKVCache` (cherry-pick)
- `tests/test_spyre_attn.py` — typed cache in tests (cherry-pick)
- `spyre_inference/distributed/kv_transfer/kv_connector/v1/spyre_paged_kv_accessor.py`
  — new `SpyrePagedKVCacheAccessor` (torch-only)
- `spyre_inference/distributed/kv_transfer/kv_connector/v1/inmemory_spyre_connector.py`
  — paged detection in `register_kv_caches`; `_load_via_paged_accessor`;
  paged read in `_save_kv_bulk`; paged shape in the NIXL receive path;
  heap-client guard; `paged_kv_active`/`get_paged_kv_status`
- `tests/test_paged_kv_accessor.py` — new (11 tests)
- `docs/paged-kv-connector-status.md` — new

## Implementation summary

`register_kv_caches` previously assumed a 4-D staging tensor and crashed on
the real Spyre cache (a tuple of page lists). The connector now detects the
`SpyrePagedKVCache`/`(k_pages, v_pages)` layout, infers all geometry from the
cache objects (no hard-coded layer/page/dtype/device), loads stored blocks
straight into page tensors with whole-page `copy_` (never slicing device
tensors), and saves pages to the store in the heap block convention so both
paths share one store format.

## Heap fallback vs paged-aware

- Heap fallback (unchanged): `HeapKVAccessor`, in-process heap client,
  legacy staging-tensor load/save; used when registered caches are not
  paged. Heap request with paged caches logs and is bypassed (no silent
  slow path).
- Paged-aware: cache registration, load path, bulk save path, NIXL receive
  shape, status/diagnostics, `transfer_units()` page descriptors.
- Not implemented: per-page NIXL registration/transfer (per-layer
  registration loop over page lists, AIU-blocked).

## Commands run

| Command | Result |
| --- | --- |
| `python3 -m compileall spyre_inference tests` | pass |
| `uvx ruff format .` then `check` / `format --check` | pass (3 files reformatted) |
| `uvx pytest tests/test_kv_connector_structure.py tests/test_kv_connector_registration.py -v` | 7 pass, 1 skip (vLLM absent) |
| `uvx --with torch pytest tests/test_paged_kv_accessor.py ...` (all 3 files) | 18 pass, 1 skip |
| `rg` secret/path scan | clean |

## Smallest pod smoke test next

Two pods (prefill, decode) with `VLLM_SPYRE_ENABLE_KV_CONNECTOR_BRIDGE=1`,
shared store, heap env vars unset. Pass: producer logs
`Paged KV cache registered: {'layout': 'list_of_pages', ...}`, decode logs
nonzero `loaded_blocks` / zero `load_misses`, first-token logits match
single-pod. (`tests/probes` has scaffolding.)

## Runtime assumptions needing AIU/vLLM/NIXL validation

1. Full-page `copy_` host→Spyre and `page.to("cpu")` are reliable.
2. vLLM 0.20.x calls `register_kv_caches` with bound page lists in worker.
3. NIXL staging interop on consumer side; per-page NIXL untested.
4. fp16 only on Spyre; non-fp16 needs CPU detour (torch-spyre PR 249).
