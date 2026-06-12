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

"""Lifecycle/readiness tests for the real vllm serve prefill/decode path.

The serve lifecycle is inherited from vLLM v0.20.1: the scheduler builds
the connector when --kv-transfer-config is set, the worker initializes the
transfer group, GPUModelRunner.initialize_kv_cache registers the paged
caches, and execute_model drives bind/start_load/wait_for_save/get_finished.
These tests pin the repo-side contracts that inheritance relies on without
needing vLLM, NIXL, or AIU; vLLM-gated tests skip cleanly when absent.
"""

import pathlib

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
PKG = REPO / "spyre_inference"
CONNECTOR = (
    PKG / "distributed" / "kv_transfer" / "kv_connector" / "v1" / "inmemory_spyre_connector.py"
)


def test_bridge_module_removed_everywhere():
    """The worker bridge would double-invoke vLLM 0.20.x's inherited
    connector lifecycle (bind/start_load/wait_for_save run from
    GPUModelRunner.execute_model). It must stay deleted."""
    assert not (PKG / "v1" / "worker" / "spyre_kv_connector_bridge.py").exists()
    offenders = [
        p
        for p in (PKG.rglob("*.py"))
        if "SpyreKVConnectorBridge" in p.read_text() or "KV_CONNECTOR_BRIDGE" in p.read_text()
    ]
    assert not offenders, f"bridge references remain: {offenders}"


def test_worker_registers_connector_before_transfer_group_init():
    """TorchSpyreWorker.init_device registers the connector factory name;
    worker-side ensure_kv_transfer_initialized is inherited afterwards."""
    src = (PKG / "v1" / "worker" / "spyre_worker.py").read_text()
    assert "register_kv_connector()" in src
    assert "execute_model" not in src  # lifecycle stays inherited
    runner = (PKG / "v1" / "worker" / "spyre_model_runner.py").read_text()
    assert "def execute_model" not in runner  # inherited from GPUModelRunner
    assert "def initialize_kv_cache_tensors" in runner  # paged caches feed inheritance


def test_active_kv_path_reporting_present():
    src = CONNECTOR.read_text()
    assert "def get_active_kv_path" in src
    # Path selection is delegated to the pure select_kv_path helper, which
    # owns the three path-name literals.
    assert "select_kv_path(" in src
    cfg = (CONNECTOR.parent / "connector_config.py").read_text()
    assert '"paged"' in cfg and '"heap"' in cfg and '"staging"' in cfg
    # The legacy fallback announces itself.
    assert "falling back to legacy" in src


def test_dynamic_endpoint_selection_is_per_batch_not_global():
    """Decode endpoint selection must use the pure per-batch helper and not
    re-introduce the old loop that mutated _nixl_remote_ip from the first
    remote it saw without flagging conflicts."""
    src = CONNECTOR.read_text()
    body = src.split("def start_load_kv", 1)[1].split("def _load_layer", 1)[0]
    assert "select_remote_endpoint(" in body
    # Both host and port are applied from the selected endpoint (the old code
    # captured remote_port but never used it).
    assert "self._nixl_remote_ip = endpoint.host" in body
    assert "self._nixl_port = endpoint.port" in body
    # Multi-endpoint batches are surfaced, documenting the one-remote-per-worker
    # concurrency limitation instead of silently using the first host.
    assert "distinct_endpoints > 1" in body
    assert "not supported in this prototype" in body


def test_nixl_activation_diagnostic_wired_at_construction():
    """The connector must surface the NIXL_PLUGIN_DIR activation hint at
    construction without importing NIXL."""
    src = CONNECTOR.read_text()
    assert "nixl_activation_diagnostic(" in src
    assert "NIXL_PLUGIN_DIR_ENV" in src


def test_decode_list_request_retry_present():
    """First LIST_REQUEST can be consumed before the producer registers the
    client agent; decode must re-send rather than rely on a settle delay."""
    src = CONNECTOR.read_text()
    body = src.split("def _load_saved_requests_nixl", 1)[1].split("def _save_request_nixl", 1)[0]
    assert body.count('send_notif("server", b"LIST_REQUEST")') >= 2
    assert "list_resend_iters" in body


def test_nixl_absent_disables_cleanly():
    src = CONNECTOR.read_text()
    body = src.split("def _init_nixl_agent", 1)[1].split("def _cleanup_nixl", 1)[0]
    assert "if not NIXL_AVAILABLE or nixl_agent is None or nixl_agent_config is None:" in body
    assert "self._use_nixl = False" in body


def test_heap_fallback_explicit_and_bounded():
    """Heap stays a legacy fallback: never silently used when pages exist."""
    src = CONNECTOR.read_text()
    assert "Heap KV requested but paged KV cache" in src
    heap = src.split("def _ensure_heap_kv_client", 1)[1].split("def bind_connector_metadata", 1)[0]
    assert "if self._paged_accessor is not None:" in heap


@pytest.mark.parametrize(
    "role_name,kv_role", [("prefill", "kv_producer"), ("decode", "kv_consumer")]
)
def test_connector_roles_construct(role_name, kv_role, monkeypatch):
    """Producer/consumer construction with kv_transfer_config; needs vLLM."""
    pytest.importorskip("vllm", reason="vLLM required for connector construction")
    monkeypatch.delenv("VLLM_SPYRE_ENABLE_NIXL_TRANSFER", raising=False)
    monkeypatch.setenv("VLLM_SPYRE_KV_ROLE", kv_role)

    from vllm.config import KVTransferConfig, VllmConfig
    from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole

    from spyre_inference.distributed.kv_transfer.kv_connector.v1.inmemory_spyre_connector import (
        InMemorySpyreConnector,
    )

    kv_cfg = KVTransferConfig(kv_connector="InMemorySpyreConnector", kv_role=kv_role)
    connector = InMemorySpyreConnector(
        VllmConfig(kv_transfer_config=kv_cfg), KVConnectorRole.WORKER
    )
    assert connector._kv_role == kv_role
    assert connector._use_nixl is False  # NIXL disabled in env
    assert connector.get_active_kv_path() == "staging"  # nothing registered yet
