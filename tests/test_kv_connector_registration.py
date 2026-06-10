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

"""Smoke tests for the Spyre KV connector registration and import path."""

import builtins
import importlib
import sys

import pytest

vllm = pytest.importorskip("vllm", reason="vLLM required for connector tests")

CONNECTOR_MODULE = (
    "spyre_inference.distributed.kv_transfer.kv_connector.v1.inmemory_spyre_connector"
)


def test_register_kv_connector_with_factory():
    """register_kv_connector registers InMemorySpyreConnector by name."""
    from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory

    import spyre_inference

    spyre_inference.register_kv_connector()
    assert "InMemorySpyreConnector" in KVConnectorFactory._registry

    # Idempotent on second call.
    spyre_inference.register_kv_connector()


def test_connector_module_imports():
    """The ported connector module imports and exposes the connector class."""
    mod = importlib.import_module(CONNECTOR_MODULE)
    assert hasattr(mod, "InMemorySpyreConnector")


def test_connector_imports_without_nixl(monkeypatch):
    """The connector module must not crash when nixl is not importable."""
    real_import = builtins.__import__

    def block_nixl(name, *args, **kwargs):
        if name == "nixl" or name.startswith("nixl."):
            raise ImportError(f"nixl blocked for test: {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", block_nixl)
    for key in [k for k in sys.modules if k == "nixl" or k.startswith("nixl.")]:
        monkeypatch.delitem(sys.modules, key)
    monkeypatch.delitem(sys.modules, CONNECTOR_MODULE, raising=False)

    mod = importlib.import_module(CONNECTOR_MODULE)
    assert mod.NIXL_AVAILABLE is False
    assert hasattr(mod, "InMemorySpyreConnector")
