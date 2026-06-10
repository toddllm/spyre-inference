# Paged KV Connector Status

Status of the bridge between the vLLM KV connector flow and the Spyre
list-of-pages KV cache. See
`artifacts/paged-kv-connector-20260610T145727Z/paged-kv-connector-report.md`
for the full implementation report.

## What is paged-cache aware now

- `spyre_inference/distributed/kv_transfer/kv_connector/v1/spyre_paged_kv_accessor.py`
  detects the per-layer `SpyrePagedKVCache`/`(k_pages, v_pages)` layout,
  infers geometry (layers, pages, kv heads, block size, head dim, dtype,
  device) from the registered cache objects, and reads/writes whole pages
  in the heap-accessor block convention
  (`[block_size, num_kv_heads, head_size]` on CPU).
- `InMemorySpyreConnector.register_kv_caches` recognizes page lists and
  derives geometry from them instead of assuming a monolithic 4-D tensor
  (which crashed before).
- `start_load_kv` loads blocks straight into the registered page tensors
  via the accessor; `_save_kv_bulk` reads pages through the accessor.
- The typed `SpyrePagedKVCache` (NamedTuple of `k_pages`/`v_pages`) from
  PR 242 is cherry-picked; the accessor accepts both the typed cache and
  the plain tuple.

## What is still heap fallback

- `HeapKVAccessor` and the in-process heap KV client remain unchanged and
  are used only when registered caches are NOT page lists (older or unknown
  layouts). When pages are registered, an explicit `warning_once` says the
  heap request is bypassed.

## Runtime assumptions still needing AIU/vLLM/NIXL validation

- `page.copy_(cpu_block.permute(...).contiguous().to(page.device))` must be
  a reliable full-page device write on Spyre (no slicing is used).
- `page.to("cpu")` must return a faithful host copy.
- NIXL transfer descriptors still describe staging-buffer tensors; per-page
  NIXL registration is described by `transfer_units()` but not implemented.
- Torch-Spyre indirect-gather (torch-spyre PR 2178/2608) is not required
  for this connector path, since pages move whole; flash-attention sizing
  (PR 2363) suggests preferring 128-aligned blocks once available.

## Two-process NIXL smoke

`examples/kv_connector/spyre_connector_nixl_smoke.py --nixl` runs split
prefill/decode processes through the connector NIXL path: the producer
saves with `VLLM_SPYRE_NIXL_BLOCKING_TRANSFER=0`, exposes the pending
transfer (`CONNECTOR_NIXL_READY`), and keeps serving LIST/PULL for
`--keepalive-s`; the consumer pulls via `_load_saved_requests_nixl()`
(`CONNECTOR_NIXL_PULL_DONE`), then loads pages with `start_load_kv` and
verifies checksums. The non-blocking `_save_request_nixl` path no longer
waits for a connected client; only blocking mode does.

## Smallest next pod smoke test

Two-pod prefill/decode with `VLLM_SPYRE_ENABLE_KV_CONNECTOR_BRIDGE=1`,
shared store backend, no heap env vars; assert producer logs
`Paged KV cache registered`, decode logs nonzero `loaded_blocks`, and
first-token logits match a single-pod run.
