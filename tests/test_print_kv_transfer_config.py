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

"""Tests for the local --kv-transfer-config printer helper.

The helper is pure stdlib (argparse, json), so everything here runs without
vLLM, NIXL, or AIU, and the script must never call ``oc`` or touch a cluster.
"""

import importlib.util
import json
import pathlib

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]


def _load_by_path(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


printer = _load_by_path(
    "print_kv_transfer_config",
    REPO / "examples" / "kv_connector" / "print_kv_transfer_config.py",
)


def test_parser_defaults():
    args = printer.build_parser().parse_args(["--remote-ip", "10.0.0.1"])
    assert args.role == "both"
    assert args.remote_ip == "10.0.0.1"
    assert args.nixl_port == printer.DEFAULT_NIXL_PORT
    assert args.connector == printer.CONNECTOR_NAME
    assert args.no_nixl is False
    assert args.as_serve_cmd is False
    assert args.compact is False


def test_parser_rejects_unknown_role():
    with pytest.raises(SystemExit):
        printer.build_parser().parse_args(["--role", "scheduler"])


def test_producer_config_shape():
    cfg = printer.producer_config(printer.CONNECTOR_NAME, True, 9100)
    assert cfg["kv_role"] == "kv_producer"
    assert cfg["kv_connector"] == printer.CONNECTOR_NAME
    assert cfg["kv_connector_module_path"] == printer.CONNECTOR_MODULE_PATH
    assert cfg["kv_connector_extra_config"] == {"use_nixl": True, "nixl_port": 9100}
    # Producer only listens; it carries no remote IP.
    assert "nixl_remote_ip" not in cfg["kv_connector_extra_config"]


def test_consumer_config_shape():
    cfg = printer.consumer_config(printer.CONNECTOR_NAME, True, "10.130.2.89", 9100)
    assert cfg["kv_role"] == "kv_consumer"
    assert cfg["kv_connector_module_path"] == printer.CONNECTOR_MODULE_PATH
    assert cfg["kv_connector_extra_config"] == {
        "use_nixl": True,
        "nixl_remote_ip": "10.130.2.89",
        "nixl_port": 9100,
    }


def test_no_nixl_flag_disables_use_nixl():
    args = printer.build_parser().parse_args(["--role", "both", "--remote-ip", "x", "--no-nixl"])
    configs = printer.build_configs(args)
    assert configs["producer"]["kv_connector_extra_config"]["use_nixl"] is False
    assert configs["consumer"]["kv_connector_extra_config"]["use_nixl"] is False


def test_build_configs_role_selection():
    parse = printer.build_parser().parse_args
    assert set(printer.build_configs(parse(["--role", "producer"]))) == {"producer"}
    assert set(printer.build_configs(parse(["--role", "consumer", "--remote-ip", "x"]))) == {
        "consumer"
    }
    assert set(printer.build_configs(parse(["--role", "both", "--remote-ip", "x"]))) == {
        "producer",
        "consumer",
    }


def test_render_json_compact_is_single_line_and_valid():
    cfg = printer.consumer_config(printer.CONNECTOR_NAME, True, "10.0.0.1", 9100)
    compact = printer.render_json(cfg, compact=True)
    assert "\n" not in compact
    assert json.loads(compact) == cfg


def test_render_serve_cmd_wraps_compact_json():
    cfg = printer.producer_config(printer.CONNECTOR_NAME, True, 9100)
    line = printer.render_serve_cmd("/models/granite", cfg)
    assert line.startswith("vllm serve /models/granite --kv-transfer-config '")
    assert line.endswith("'")
    # The single-quoted payload is one round-trippable JSON object.
    payload = line.split("--kv-transfer-config '", 1)[1][:-1]
    assert json.loads(payload) == cfg


def test_main_requires_remote_ip_for_consumer(capsys):
    rc = printer.main(["--role", "consumer"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "--remote-ip is required" in err


def test_main_both_emits_markers_and_valid_json(capsys):
    rc = printer.main(["--role", "both", "--remote-ip", "10.130.2.89"])
    assert rc == 0
    out = capsys.readouterr().out
    assert printer.MARK_PRODUCER in out
    assert printer.MARK_CONSUMER in out

    # Each marker is followed by a parseable JSON block.
    for marker, role in (
        (printer.MARK_PRODUCER, "kv_producer"),
        (printer.MARK_CONSUMER, "kv_consumer"),
    ):
        block = out.split(marker, 1)[1].lstrip()
        decoded, _ = json.JSONDecoder().raw_decode(block)
        assert decoded["kv_role"] == role


def test_main_producer_only_omits_consumer(capsys):
    rc = printer.main(["--role", "producer"])
    assert rc == 0
    out = capsys.readouterr().out
    assert printer.MARK_PRODUCER in out
    assert printer.MARK_CONSUMER not in out


def test_script_has_no_cluster_dependencies():
    """A local helper must not shell out (so it cannot run oc/kubectl/create
    pods) nor import vLLM. Checks executable patterns, not prose mentions."""
    src = (REPO / "examples" / "kv_connector" / "print_kv_transfer_config.py").read_text()
    for forbidden in (
        "import subprocess",
        "subprocess.",
        "os.system(",
        "os.popen(",
        "import vllm",
        "from vllm",
    ):
        assert forbidden not in src
