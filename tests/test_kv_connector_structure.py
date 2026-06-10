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

"""Structural checks for the ported KV connector that run without vLLM.

These catch obvious port regressions (bad module paths, missing package
files, stale env var declarations, leftover vllm_spyre references) on any
machine, including ones without vLLM installed.
"""

import ast
import pathlib
import re

REPO = pathlib.Path(__file__).resolve().parents[1]
PKG = REPO / "spyre_inference"
CONNECTOR_DIR = PKG / "distributed" / "kv_transfer" / "kv_connector" / "v1"
CONNECTOR_MODULE = (
    "spyre_inference.distributed.kv_transfer.kv_connector.v1.inmemory_spyre_connector"
)


def test_connector_module_path_resolves_to_file():
    rel = pathlib.Path(*CONNECTOR_MODULE.split(".")).with_suffix(".py")
    assert (REPO / rel).is_file()


def test_registration_uses_existing_module_path():
    init_src = (PKG / "__init__.py").read_text()
    assert f'"{CONNECTOR_MODULE}"' in init_src
    assert '"InMemorySpyreConnector"' in init_src


def test_all_connector_packages_have_init_files():
    parts = PKG / "distributed"
    for sub in ["", "kv_transfer", "kv_transfer/kv_connector", "kv_transfer/kv_connector/v1"]:
        assert (parts / sub / "__init__.py").is_file(), f"missing __init__.py under {sub or '.'}"


def test_connector_defines_class_and_nixl_guard():
    src = (CONNECTOR_DIR / "inmemory_spyre_connector.py").read_text()
    tree = ast.parse(src)
    classes = {n.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef)}
    assert "InMemorySpyreConnector" in classes
    assert "NIXL_AVAILABLE = False" in src, "nixl import must be guarded"


def test_no_vllm_spyre_references():
    offenders = [p for p in PKG.rglob("*.py") if "vllm_spyre" in p.read_text()]
    assert not offenders, f"leftover vllm_spyre references: {offenders}"


def test_kv_env_vars_declared_consistently():
    src = (PKG / "envs.py").read_text()
    declared = set(re.findall(r'"(VLLM_SPYRE_[A-Z_]+)":', src))
    type_checked = set(re.findall(r"(VLLM_SPYRE_[A-Z_]+):", src))
    assert declared, "expected ported KV env vars in envs.py"
    assert type_checked == declared, f"TYPE_CHECKING/dict drift: {type_checked ^ declared}"
    for required in [
        "VLLM_SPYRE_ENABLE_NIXL_TRANSFER",
        "VLLM_SPYRE_NIXL_REMOTE_IP",
        "VLLM_SPYRE_KV_ROLE",
        "VLLM_SPYRE_KV_STORE_BACKEND",
    ]:
        assert required in declared


def test_registration_hooks_present():
    assert "register_kv_connector()" in (PKG / "platform.py").read_text()
    assert "register_kv_connector()" in (PKG / "v1" / "worker" / "spyre_worker.py").read_text()
