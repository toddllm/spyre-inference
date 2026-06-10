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

"""Tests for the connector-level NIXL smoke script.

Helper coverage (CLI parser, paged cache builder, checksums) runs with
torch only. The full in-process roundtrip needs vLLM and skips cleanly
when it is absent. No NIXL or AIU hardware is required anywhere.
"""

import importlib.util
import json
import pathlib

import pytest

torch = pytest.importorskip("torch", reason="torch required for smoke helper tests")

REPO = pathlib.Path(__file__).resolve().parents[1]
CONNECTOR_DIR = REPO / "spyre_inference" / "distributed" / "kv_transfer" / "kv_connector" / "v1"


def _load_by_path(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# Loading by file path keeps these imports vLLM-free: the smoke script's
# helpers are torch-only, and spyre_inference/__init__.py would pull vLLM in.
smoke = _load_by_path(
    "spyre_connector_nixl_smoke",
    REPO / "examples" / "kv_connector" / "spyre_connector_nixl_smoke.py",
)
accessor_mod = _load_by_path(
    "spyre_paged_kv_accessor", CONNECTOR_DIR / "spyre_paged_kv_accessor.py"
)


def test_parser_roles_and_defaults():
    args = smoke.build_parser().parse_args(["--role", "both"])
    assert args.role == "both"
    assert args.num_layers == 2
    assert args.num_kv_heads == 2
    assert args.block_size == 4
    assert args.head_dim == 8
    assert args.num_pages == 4
    assert args.block_ids == [1, 2]
    assert args.device == "cpu"

    args = smoke.build_parser().parse_args(
        ["--role", "decode", "--block-ids", "0", "3", "--num-pages", "6"]
    )
    assert args.role == "decode"
    assert args.block_ids == [0, 3]
    assert args.num_pages == 6


def test_parser_requires_valid_role():
    with pytest.raises(SystemExit):
        smoke.build_parser().parse_args([])
    with pytest.raises(SystemExit):
        smoke.build_parser().parse_args(["--role", "scheduler"])


def test_paged_cache_builder_activates_accessor():
    caches = smoke.build_paged_kv_caches(
        num_layers=2, num_kv_heads=2, block_size=4, head_dim=8, num_pages=4
    )
    accessor = accessor_mod.SpyrePagedKVCacheAccessor.try_from_kv_caches(caches)
    assert accessor is not None
    assert accessor.num_layers == 2
    assert accessor.num_pages == 4
    assert accessor.page_shape == (2, 4, 8)
    assert accessor.dtype == torch.float16


def test_deterministic_blocks_distinct_and_reproducible():
    kw = {"num_kv_heads": 2, "block_size": 4, "head_dim": 8}
    a = smoke.deterministic_block(0, "k", 1, **kw)
    assert a.shape == (2, 4, 8)
    assert a.dtype == torch.float16
    assert torch.equal(a, smoke.deterministic_block(0, "k", 1, **kw))
    assert not torch.equal(a, smoke.deterministic_block(0, "v", 1, **kw))
    assert not torch.equal(a, smoke.deterministic_block(1, "k", 1, **kw))
    assert not torch.equal(a, smoke.deterministic_block(0, "k", 2, **kw))


def test_checksum_match_helper():
    filled = smoke.build_paged_kv_caches(
        num_layers=2, num_kv_heads=2, block_size=4, head_dim=8, num_pages=4, fill_block_ids=[1, 2]
    )
    empty = smoke.build_paged_kv_caches(
        num_layers=2, num_kv_heads=2, block_size=4, head_dim=8, num_pages=4
    )
    expected = smoke.cache_checksums(filled, [1, 2])
    assert len(expected) == 2 * 2 * 2  # layers * kinds * blocks
    assert smoke.checksums_match(expected, smoke.cache_checksums(filled, [1, 2]))
    assert not smoke.checksums_match(expected, smoke.cache_checksums(empty, [1, 2]))
    assert not smoke.checksums_match({}, {})


def test_expected_checksums_match_builder():
    args = smoke.build_parser().parse_args(["--role", "both"])
    assert smoke.expected_checksums(args) == smoke.cache_checksums(
        smoke.build_paged_kv_caches(
            num_layers=2,
            num_kv_heads=2,
            block_size=4,
            head_dim=8,
            num_pages=4,
            fill_block_ids=[1, 2],
        ),
        [1, 2],
    )


def test_inprocess_roundtrip_without_nixl(tmp_path, capsys, monkeypatch):
    """Full prefill -> store -> decode roundtrip through the real connector."""
    pytest.importorskip("vllm", reason="vLLM required for connector roundtrip")
    monkeypatch.delenv("VLLM_SPYRE_ENABLE_NIXL_TRANSFER", raising=False)
    monkeypatch.delenv("VLLM_SPYRE_EXPERIMENTAL_HEAP_KV_ENABLE", raising=False)

    from spyre_inference.distributed.kv_transfer.kv_connector.v1.inmemory_spyre_connector import (
        reset_global_store,
    )

    reset_global_store()
    result_file = tmp_path / "result.json"
    rc = smoke.main(["--role", "both", "--result-file", str(result_file)])

    out = capsys.readouterr().out
    for marker in (
        smoke.MARK_PREFILL_READY,
        smoke.MARK_SAVE_DONE,
        smoke.MARK_DECODE_START,
        smoke.MARK_LOAD_DONE,
        f"{smoke.MARK_CONTENT_MATCH} true",
        f"{smoke.MARK_SUCCESS} true",
    ):
        assert marker in out

    result = json.loads(result_file.read_text())
    assert rc == 0
    assert result["success"] is True
    assert result["content_match"] is True
    assert result["error"] is None
    assert result["layout"] == "list_of_pages"
    assert result["expected_checksums"] == result["actual_checksums"]
    assert result["load_error_block_ids"] == []
