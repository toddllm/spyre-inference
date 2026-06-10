# Copyright 2026 The Spyre-Inference Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Connector-level KV smoke test for InMemorySpyreConnector.

Drives the real connector path — not nixl._api directly:

    register_kv_caches -> bind_connector_metadata
        -> wait_for_save (prefill) / start_load_kv (decode)
        -> verify KV block contents

The synthetic KV cache uses the Spyre paged layout: each layer cache is
``(k_pages, v_pages)`` with per-page tensors ``[num_kv_heads, block_size,
head_dim]``, so ``SpyrePagedKVCacheAccessor`` is active. Set
``VLLM_SPYRE_ENABLE_NIXL_TRANSFER=1`` (plus producer/consumer roles) on a
pod to push the same path through NIXL; with it unset the shared in-memory
store carries the blocks, which is the in-process default.

Grep-friendly markers:
    CONNECTOR_PREFILL_READY / CONNECTOR_SAVE_DONE
    CONNECTOR_DECODE_START / CONNECTOR_LOAD_DONE
    CONNECTOR_CONTENT_MATCH true|false
    CONNECTOR_SMOKE_SUCCESS true|false

Run on one host (in-process roundtrip, no NIXL needed):
    python examples/kv_connector/spyre_connector_nixl_smoke.py --role both

This module's helpers (CLI parser, paged cache builder, checksums) import
without vLLM; the connector itself is imported lazily inside main().
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
from typing import Any

import torch

MARK_PREFILL_READY = "CONNECTOR_PREFILL_READY"
MARK_SAVE_DONE = "CONNECTOR_SAVE_DONE"
MARK_DECODE_START = "CONNECTOR_DECODE_START"
MARK_LOAD_DONE = "CONNECTOR_LOAD_DONE"
MARK_CONTENT_MATCH = "CONNECTOR_CONTENT_MATCH"
MARK_SUCCESS = "CONNECTOR_SMOKE_SUCCESS"

PREFILL_REQ_ID = "smoke-prefill-0"
DECODE_REQ_ID = "smoke-decode-0"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--role", choices=("prefill", "decode", "both"), required=True)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-kv-heads", type=int, default=2)
    parser.add_argument("--block-size", type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=8)
    parser.add_argument("--num-pages", type=int, default=4)
    parser.add_argument("--block-ids", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--device", default="cpu", help="page tensor device (cpu for smoke)")
    parser.add_argument(
        "--result-file",
        default="spyre_connector_smoke_result.json",
        help="where to write the JSON result",
    )
    return parser


def layer_names(num_layers: int) -> list[str]:
    return [f"model.layers.{i}.self_attn.attn" for i in range(num_layers)]


def deterministic_block(
    layer_idx: int,
    kv_kind: str,
    block_id: int,
    *,
    num_kv_heads: int,
    block_size: int,
    head_dim: int,
) -> torch.Tensor:
    """Known-value page [num_kv_heads, block_size, head_dim], fp16."""
    base = layer_idx * 1000.0 + block_id * 10.0 + (0.0 if kv_kind == "k" else 5.0)
    numel = num_kv_heads * block_size * head_dim
    ramp = torch.arange(numel, dtype=torch.float16) * 0.125
    return (base + ramp).reshape(num_kv_heads, block_size, head_dim).to(torch.float16)


def build_paged_kv_caches(
    *,
    num_layers: int,
    num_kv_heads: int,
    block_size: int,
    head_dim: int,
    num_pages: int,
    fill_block_ids: list[int] | None = None,
    device: str = "cpu",
) -> dict[str, tuple[list[torch.Tensor], list[torch.Tensor]]]:
    """Synthetic Spyre paged caches; pages in fill_block_ids get known values."""
    caches: dict[str, tuple[list[torch.Tensor], list[torch.Tensor]]] = {}
    for layer_idx, name in enumerate(layer_names(num_layers)):
        pages = {}
        for kind in ("k", "v"):
            pages[kind] = [
                torch.zeros(num_kv_heads, block_size, head_dim, dtype=torch.float16, device=device)
                for _ in range(num_pages)
            ]
            for block_id in fill_block_ids or []:
                pages[kind][block_id] = deterministic_block(
                    layer_idx,
                    kind,
                    block_id,
                    num_kv_heads=num_kv_heads,
                    block_size=block_size,
                    head_dim=head_dim,
                ).to(device)
        caches[name] = (pages["k"], pages["v"])
    return caches


def block_checksum(page: torch.Tensor) -> str:
    raw = page.to("cpu").contiguous().view(-1).view(torch.uint8)
    return hashlib.sha256(bytes(raw.tolist())).hexdigest()


def cache_checksums(
    caches: dict[str, tuple[list[torch.Tensor], list[torch.Tensor]]],
    block_ids: list[int],
) -> dict[str, str]:
    """Checksum per layer/kind/block: {'layer|k|1': sha256}."""
    sums: dict[str, str] = {}
    for name, (k_pages, v_pages) in sorted(caches.items()):
        for kind, pages in (("k", k_pages), ("v", v_pages)):
            for block_id in block_ids:
                sums[f"{name}|{kind}|{block_id}"] = block_checksum(pages[block_id])
    return sums


def checksums_match(expected: dict[str, str], actual: dict[str, str]) -> bool:
    return bool(expected) and expected == actual


def expected_checksums(args: argparse.Namespace) -> dict[str, str]:
    filled = build_paged_kv_caches(
        num_layers=args.num_layers,
        num_kv_heads=args.num_kv_heads,
        block_size=args.block_size,
        head_dim=args.head_dim,
        num_pages=args.num_pages,
        fill_block_ids=list(args.block_ids),
    )
    return cache_checksums(filled, list(args.block_ids))


def _make_connector(role_name: str):
    """Build an InMemorySpyreConnector without a full engine."""
    from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole

    from spyre_inference.distributed.kv_transfer.kv_connector.v1.inmemory_spyre_connector import (
        InMemorySpyreConnector,
    )

    role = KVConnectorRole.WORKER
    try:
        from vllm.config import VllmConfig

        vllm_config = VllmConfig()
    except Exception:
        # Minimal duck-typed config: the connector only reads
        # cache_config.block_size at construction time.
        from types import SimpleNamespace

        vllm_config = SimpleNamespace(
            cache_config=SimpleNamespace(block_size=0), kv_transfer_config=None
        )
    return InMemorySpyreConnector(vllm_config, role)


def _make_meta(args: argparse.Namespace, *, is_store: bool):
    from spyre_inference.distributed.kv_transfer.kv_connector.v1.metadata import (
        SpyreConnectorMeta,
    )

    meta = SpyreConnectorMeta(
        layer_names=layer_names(args.num_layers),
        block_size=args.block_size,
        dtype="torch.float16",
        num_layers=args.num_layers,
        num_kv_heads=args.num_kv_heads,
        head_dim=args.head_dim,
    )
    block_ids = list(args.block_ids)
    token_count = len(block_ids) * args.block_size
    if is_store:
        meta.add_store_request(PREFILL_REQ_ID, block_ids, token_count=token_count)
    else:
        meta.add_load_request(
            DECODE_REQ_ID,
            block_ids,
            source_req_id=PREFILL_REQ_ID,
            token_count=token_count,
            block_mapping=[(b, b) for b in block_ids],
        )
    return meta


def run_prefill(args: argparse.Namespace, result: dict[str, Any]) -> None:
    connector = _make_connector("prefill")
    caches = build_paged_kv_caches(
        num_layers=args.num_layers,
        num_kv_heads=args.num_kv_heads,
        block_size=args.block_size,
        head_dim=args.head_dim,
        num_pages=args.num_pages,
        fill_block_ids=list(args.block_ids),
        device=args.device,
    )
    connector.register_kv_caches(caches)
    if not connector.paged_kv_active():
        raise RuntimeError("paged accessor not active for synthetic page lists")
    print(MARK_PREFILL_READY, flush=True)

    connector.bind_connector_metadata(_make_meta(args, is_store=True))
    connector.wait_for_save()
    connector.clear_connector_metadata()
    result["connector_stats"]["prefill"] = connector.get_cumulative_metrics()
    print(MARK_SAVE_DONE, flush=True)


def run_decode(args: argparse.Namespace, result: dict[str, Any]) -> tuple[bool, bool]:
    print(MARK_DECODE_START, flush=True)
    connector = _make_connector("decode")
    caches = build_paged_kv_caches(
        num_layers=args.num_layers,
        num_kv_heads=args.num_kv_heads,
        block_size=args.block_size,
        head_dim=args.head_dim,
        num_pages=args.num_pages,
        fill_block_ids=None,  # decode starts empty
        device=args.device,
    )
    connector.register_kv_caches(caches)
    if not connector.paged_kv_active():
        raise RuntimeError("paged accessor not active for synthetic page lists")

    connector.bind_connector_metadata(_make_meta(args, is_store=False))
    connector.start_load_kv(None)
    load_errors = connector.get_block_ids_with_load_errors()
    connector.clear_connector_metadata()
    result["connector_stats"]["decode"] = connector.get_cumulative_metrics()
    result["load_error_block_ids"] = sorted(load_errors)
    print(MARK_LOAD_DONE, flush=True)

    actual = cache_checksums(caches, list(args.block_ids))
    result["actual_checksums"] = actual
    match = checksums_match(result["expected_checksums"], actual) and not load_errors
    return match, bool(load_errors)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result: dict[str, Any] = {
        "role": args.role,
        "layout": "list_of_pages",
        "request_ids": {"prefill": PREFILL_REQ_ID, "decode": DECODE_REQ_ID},
        "block_ids": list(args.block_ids),
        "connector_stats": {},
        "expected_checksums": expected_checksums(args),
        "actual_checksums": {},
        "content_match": None,
        "success": False,
        "error": None,
    }

    try:
        if max(args.block_ids) >= args.num_pages or min(args.block_ids) < 0:
            raise ValueError("--block-ids must be within [0, --num-pages)")
        if args.role in ("prefill", "both"):
            run_prefill(args, result)
        if args.role in ("decode", "both"):
            match, _ = run_decode(args, result)
            result["content_match"] = match
            print(f"{MARK_CONTENT_MATCH} {str(match).lower()}", flush=True)
            result["success"] = match
        else:
            result["success"] = True  # prefill-only success = save completed
    except Exception as exc:  # noqa: BLE001 — smoke must always emit a verdict
        result["error"] = f"{type(exc).__name__}: {exc}"

    print(f"{MARK_SUCCESS} {str(result['success']).lower()}", flush=True)
    pathlib.Path(args.result_file).write_text(json.dumps(result, indent=2))
    return 0 if result["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
