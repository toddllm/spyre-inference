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

"""Print ``--kv-transfer-config`` JSON for InMemorySpyreConnector.

A local-only convenience for the prefill/decode (PD) disaggregation pod
harness: given a remote prefill endpoint, it emits the exact
``vllm serve --kv-transfer-config`` JSON for the producer (prefill) and/or
consumer (decode) side, matching the precedence documented in
``docs/user_guide/kv_connector_config.md``.

It does **not** call ``oc``, create pods, read kubeconfigs, or touch the IBM
cluster — it only formats JSON on stdout, so it can sit next to the pod
harness without duplicating it. Pure stdlib (argparse, json), so it imports
and unit-tests without vLLM, NIXL, or AIU.

Examples::

    # Decode/consumer config pointing at a prefill pod:
    python examples/kv_connector/print_kv_transfer_config.py \
        --role consumer --remote-ip 10.130.2.89 --nixl-port 9100

    # Both sides at once, as ready-to-paste `vllm serve` lines:
    python examples/kv_connector/print_kv_transfer_config.py \
        --role both --remote-ip 10.130.2.89 --as-serve-cmd --model /models/granite

Grep-friendly markers: ``KV_TRANSFER_CONFIG_PRODUCER`` and
``KV_TRANSFER_CONFIG_CONSUMER`` precede each emitted block.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

CONNECTOR_NAME = "InMemorySpyreConnector"
CONNECTOR_MODULE_PATH = (
    "spyre_inference.distributed.kv_transfer.kv_connector.v1.inmemory_spyre_connector"
)
DEFAULT_NIXL_PORT = 9100
DEFAULT_MODEL = "<model>"

MARK_PRODUCER = "KV_TRANSFER_CONFIG_PRODUCER"
MARK_CONSUMER = "KV_TRANSFER_CONFIG_CONSUMER"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Print --kv-transfer-config JSON for InMemorySpyreConnector."
    )
    parser.add_argument(
        "--role",
        choices=("producer", "consumer", "both"),
        default="both",
        help="Which side(s) to emit (producer=prefill, consumer=decode).",
    )
    parser.add_argument(
        "--remote-ip",
        default=None,
        help="Prefill host the consumer pulls KV from (required for consumer/both).",
    )
    parser.add_argument(
        "--nixl-port",
        type=int,
        default=DEFAULT_NIXL_PORT,
        help=f"NIXL listen/connect port (default {DEFAULT_NIXL_PORT}).",
    )
    parser.add_argument(
        "--connector",
        default=CONNECTOR_NAME,
        help=f"kv_connector name (default {CONNECTOR_NAME}).",
    )
    parser.add_argument(
        "--no-nixl",
        action="store_true",
        help="Emit use_nixl=false (in-process/local store path instead of NIXL).",
    )
    parser.add_argument(
        "--as-serve-cmd",
        action="store_true",
        help="Wrap each config in a copy-paste `vllm serve` command line.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Model placeholder used by --as-serve-cmd output.",
    )
    parser.add_argument(
        "--compact",
        action="store_true",
        help="Emit single-line JSON instead of indented JSON.",
    )
    return parser


def producer_config(connector: str, use_nixl: bool, nixl_port: int) -> dict[str, Any]:
    """Prefill-side config. The producer only listens, so it carries no
    remote IP; it advertises its NIXL port for consumers to connect to."""
    return {
        "kv_connector": connector,
        "kv_role": "kv_producer",
        "kv_connector_module_path": CONNECTOR_MODULE_PATH,
        "kv_connector_extra_config": {"use_nixl": use_nixl, "nixl_port": nixl_port},
    }


def consumer_config(
    connector: str, use_nixl: bool, remote_ip: str, nixl_port: int
) -> dict[str, Any]:
    """Decode-side config. The consumer points at the producer entirely
    through ``kv_connector_extra_config``, so no env var is required."""
    return {
        "kv_connector": connector,
        "kv_role": "kv_consumer",
        "kv_connector_module_path": CONNECTOR_MODULE_PATH,
        "kv_connector_extra_config": {
            "use_nixl": use_nixl,
            "nixl_remote_ip": remote_ip,
            "nixl_port": nixl_port,
        },
    }


def build_configs(args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    """Return ``{role: config}`` for the requested side(s)."""
    use_nixl = not args.no_nixl
    configs: dict[str, dict[str, Any]] = {}
    if args.role in ("producer", "both"):
        configs["producer"] = producer_config(args.connector, use_nixl, args.nixl_port)
    if args.role in ("consumer", "both"):
        configs["consumer"] = consumer_config(
            args.connector, use_nixl, args.remote_ip, args.nixl_port
        )
    return configs


def render_json(config: dict[str, Any], compact: bool) -> str:
    if compact:
        return json.dumps(config, separators=(",", ":"))
    return json.dumps(config, indent=2)


def render_serve_cmd(model: str, config: dict[str, Any]) -> str:
    """A copy-paste `vllm serve` line; the config is always compact here so
    it stays a single shell-quotable argument."""
    return f"vllm serve {model} --kv-transfer-config '{render_json(config, compact=True)}'"


def render(args: argparse.Namespace, configs: dict[str, dict[str, Any]]) -> str:
    marks = {"producer": MARK_PRODUCER, "consumer": MARK_CONSUMER}
    lines: list[str] = []
    for role in ("producer", "consumer"):
        if role not in configs:
            continue
        lines.append(marks[role])
        if args.as_serve_cmd:
            lines.append(render_serve_cmd(args.model, configs[role]))
        else:
            lines.append(render_json(configs[role], args.compact))
        lines.append("")
    return "\n".join(lines).rstrip("\n")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.role in ("consumer", "both") and not args.remote_ip:
        print(
            f"error: --remote-ip is required for the consumer side (role {args.role!r})",
            file=sys.stderr,
        )
        return 2
    configs = build_configs(args)
    print(render(args, configs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
