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

"""Pure resolution of InMemorySpyreConnector runtime settings.

These helpers derive role, NIXL enablement, and the NIXL remote endpoint
from the two sources the connector accepts:

1. Environment variables (the baked-image/manual deployment path). An
   explicitly set env var always wins, for backwards compatibility.
2. vLLM ``kv_transfer_config`` (the ``vllm serve --kv-transfer-config``
   path): ``kv_role`` for the role, ``kv_connector_extra_config`` for
   NIXL knobs (``use_nixl``, ``nixl_remote_ip``, ``nixl_port``), and
   ``kv_ip`` as a remote-IP fallback.

The functions take plain values (no vLLM types) so they unit-test without
vLLM, NIXL, or AIU installed.
"""

from __future__ import annotations

from typing import Any, NamedTuple

DEFAULT_NIXL_REMOTE_IP = "10.130.2.89"

# Env var that activates the source-controlled UCX/NIXL image layer: it must
# point at the directory holding the built NIXL plugins. We only ever check
# whether it is *set*, never echo its value, so deployment-local paths stay
# out of logs.
NIXL_PLUGIN_DIR_ENV = "NIXL_PLUGIN_DIR"


def _parse_env_bool(value: str) -> bool:
    """Match the existing ``bool(int(...))`` env parsing (1/0)."""
    return bool(int(value))


def resolve_kv_role(env_role: str | None, config_role: str | None) -> str:
    """Non-empty ``VLLM_SPYRE_KV_ROLE`` wins, else config role, else ''."""
    if env_role:
        return env_role
    return config_role or ""


def resolve_use_nixl(env_value: str | None, extra_config: dict[str, Any] | None) -> bool:
    """Explicitly set env wins, else ``extra_config['use_nixl']``, else False.

    ``env_value`` is the raw ``os.environ.get`` result: ``None`` means the
    env var was not set, so config is consulted.
    """
    if env_value is not None:
        return _parse_env_bool(env_value)
    if extra_config and "use_nixl" in extra_config:
        return bool(extra_config["use_nixl"])
    return False


def resolve_nixl_remote_ip(
    env_value: str | None,
    extra_config: dict[str, Any] | None,
    config_ip: str | None,
    default: str = DEFAULT_NIXL_REMOTE_IP,
) -> str:
    """Env override wins, else ``extra_config['nixl_remote_ip']``, else
    the connector ``kv_ip``, else the legacy default."""
    if env_value is not None:
        return env_value
    if extra_config and extra_config.get("nixl_remote_ip"):
        return str(extra_config["nixl_remote_ip"])
    if config_ip:
        return str(config_ip)
    return default


def resolve_nixl_port(extra_config: dict[str, Any] | None, default: int) -> int:
    """``extra_config['nixl_port']`` if set, else the module default.

    ``kv_port`` is intentionally not used as a fallback: vLLM auto-assigns
    it (e.g. 14579), which would silently break the established listen-port
    contract. Operators wanting a non-default port set ``nixl_port``.
    """
    if extra_config and extra_config.get("nixl_port"):
        return int(extra_config["nixl_port"])
    return default


# --- decode-side dynamic endpoint routing ----------------------------------


class RemoteEndpoint(NamedTuple):
    """An explicit ``host:port`` a decode worker should pull KV from.

    A NamedTuple (not a dataclass) so this pure helper module stays loadable
    by file path in tests without registering it in ``sys.modules`` — the
    way the connector test-suite imports it to avoid pulling vLLM in.
    """

    host: str
    port: int


def select_remote_endpoint(
    candidates: list[tuple[str | None, int | None]],
    fallback_host: str,
    fallback_port: int,
) -> tuple[RemoteEndpoint, int]:
    """Pick the decode-side remote NIXL endpoint for one load batch.

    ``candidates`` is the list of ``(host, port)`` pairs carried by the
    batch's load requests (populated from each request's
    ``kv_transfer_params`` routing metadata: ``remote_host``/``remote_port``).
    Returns ``(endpoint, distinct_count)`` where ``distinct_count`` is the
    number of *distinct* explicit endpoints seen:

    * ``0``  -> no routing metadata in the batch; ``fallback_*`` (the env/
      config endpoint) is returned unchanged.
    * ``1``  -> exactly one explicit endpoint; it is returned.
    * ``>1`` -> several distinct endpoints; the first is returned and the
      count lets the caller warn. This connector binds a single NIXL client
      agent to one server for the worker's lifetime, so a batch (or a later
      batch) naming several prefill hosts cannot be served concurrently.

    The function is pure and vLLM-free: it takes plain tuples rather than
    request dataclasses, so it unit-tests without vLLM, NIXL, or AIU, and it
    never mutates connector state — the caller decides what to apply.
    """
    distinct: list[RemoteEndpoint] = []
    for host, port in candidates:
        if host and port:
            endpoint = RemoteEndpoint(str(host), int(port))
            if endpoint not in distinct:
                distinct.append(endpoint)
    if not distinct:
        return RemoteEndpoint(fallback_host, fallback_port), 0
    return distinct[0], len(distinct)


# --- KV access path selection ----------------------------------------------


def select_kv_path(paged_active: bool, use_heap_kv: bool) -> str:
    """Resolve which KV access path is active.

    Paged is the primary Spyre path and always wins when real paged caches
    are registered. Heap is a legacy fallback that is used only when it was
    explicitly requested *and* no paged cache is present. Otherwise the
    monolithic staging tensors path is used.
    """
    if paged_active:
        return "paged"
    if use_heap_kv:
        return "heap"
    return "staging"


# --- NIXL activation diagnostics -------------------------------------------


def nixl_activation_diagnostic(nixl_available: bool, plugin_dir_set: bool) -> str | None:
    """Explain why NIXL transfer cannot fully activate, or ``None`` if it can.

    Takes plain booleans — whether the ``nixl`` package imported, and whether
    ``NIXL_PLUGIN_DIR`` is set — so it unit-tests without importing NIXL or
    reading the real environment. The message names the activation
    requirement but never echoes path *values*, keeping deployment-local
    paths out of logs.
    """
    if nixl_available and plugin_dir_set:
        return None
    reasons: list[str] = []
    if not nixl_available:
        reasons.append("the NIXL Python package is not importable")
    if not plugin_dir_set:
        reasons.append(f"{NIXL_PLUGIN_DIR_ENV} is not set, so NIXL plugins will not load")
    return (
        "NIXL transfer requested but not fully activated: "
        + "; ".join(reasons)
        + f". Activate the UCX/NIXL image layer and set {NIXL_PLUGIN_DIR_ENV} "
        + "to the NIXL plugins directory."
    )
