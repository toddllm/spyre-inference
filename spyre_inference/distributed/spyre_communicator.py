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

"""DeviceCommunicator override for IBM Spyre devices.

libspyre_comms natively implements: `barrier`, `broadcast`, `send`/`recv`,
list-form `allgather`, and `gather`. The `allreduce` and `reduce` overrides
on `SpyreCommsContext` are still throw-stubs. The `_allgather_base` entry
point on the torch-spyre spyreccl backend side is also still stubbed,
so `dist.all_gather_into_tensor` does not work.

This class supplies:
  - A hand-rolled `all_reduce` for TP=2 using send/recv + broadcast (native
    allreduce is not yet available).
  - An `all_gather` that uses native list-form `dist.all_gather` because
    the base class routes through `dist.all_gather_into_tensor` (blocked
    by the `_allgather_base` stub).

This class supplies a hand-rolled fallback for `all_reduce` that works for
TP=2 by using send/recv + broadcast (all of which ARE implemented), so the
TP forward path can run end-to-end on two ranks without waiting for the
upstream comms-side implementations to land. Other collectives raise a
clear NotImplementedError describing what's needed to unblock them.

Remaining per-op blockers (REPLACE-WITH-NATIVE markers below):
  - all_reduce : libspyre_comms native allreduce.
  - all_gather : torch-spyre spyreccl `_allgather_base` (so the base class
                 can use `dist.all_gather_into_tensor` directly).
  - reduce     : libspyre_comms native reduce (not on the TP forward path).

The companion test file `tests/test_spyre_comms_native_probes.py` runs
each native collective on a real spyreccl device_group and is xfail-strict.
When a comms RPM lands an impl, the corresponding probe flips to passing,
the strict-xfail fails CI, and that's the signal to delete the override
here.
"""

from __future__ import annotations

import torch
import torch.distributed as dist

from vllm.distributed.device_communicators.base_device_communicator import (
    DeviceCommunicatorBase,
)


class _SpyreAllReduceWork:
    """Work-like handle returned by ``all_reduce(async_op=True)``.

    Mirrors ``dist.broadcast(async_op=True)``'s Work API:
    ``work.wait()`` blocks until the two underlying broadcasts complete,
    applies the in-place add, and returns the reduced tensor.
    """

    def __init__(self, *, input_, peer, w_self, w_peer):
        # input_ holds this rank's contribution; receives the sum in wait().
        # peer is the buffer for the peer's contribution.
        self._input = input_
        self._peer = peer
        self._w_self = w_self
        self._w_peer = w_peer
        self._done = False

    def wait(self) -> torch.Tensor:
        if self._done:
            return self._input
        self._w_self.wait()
        self._w_peer.wait()
        self._input.add_(self._peer)
        self._done = True
        return self._input


# libspyre_comms enforces a per-message minimum size (128 bytes). Every
# TP all_reduce we expect to see in inference is on a hidden-state shard
# that comfortably exceeds this threshold (hidden_size=1024 float16 =
# 2048 bytes, well above the 128-byte threshold), so we don't pad here.
# If something smaller hits this path, the fallback will surface the
# comms-layer error verbatim and we add padding then.


def _spyre_collective_unsupported_message(
    op_name: str, world_size: int, blocker: str | None = None
) -> str:
    parts = [
        f"SpyreCommunicator: {op_name} is not natively available in the "
        "installed libspyre_comms (the corresponding "
        f"SpyreCommsContext::{op_name} method throws). ",
    ]
    if blocker is not None:
        parts.append(f"Blocked on: {blocker}. ")
    parts.append(
        f"No fallback is implemented for world_size={world_size}. Either "
        "wait for the upstream implementation to land + a comms RPM rebuild, "
        "or extend SpyreCommunicator with an additional manual fallback."
    )
    return "".join(parts)


class SpyreCommunicator(DeviceCommunicatorBase):
    """Spyre-specific DeviceCommunicator with manual fallbacks.

    See the module docstring for the full picture. In short:
      - `all_reduce` is overridden with a TP=2-only manual reduce-to-root
        + broadcast that uses send/recv. TP>2 raises.
      - All other broken collectives raise NotImplementedError describing
        what's needed to unblock them.
    """

    def all_reduce(
        self,
        input_: torch.Tensor,
        async_op: bool = False,
    ):
        """TP=2 all_reduce via two broadcasts.

        Each rank broadcasts its own ``input_`` and receives the peer's
        broadcast into a scratch buffer. ``dist.broadcast`` on the
        sender's side is a no-op for the data (``input_`` is unchanged
        on the sender), so it still holds this rank's contribution and
        we apply ``input_.add_(peer)`` to produce the sum.

        With ``async_op=False`` (default), behaves like a normal sync
        allreduce and returns the reduced tensor in place.

        With ``async_op=True``, returns a :class:`_SpyreAllReduceWork`
        whose ``.wait()`` drains the two underlying broadcasts and
        applies the add. Multiple ``async_op=True`` calls can be in
        flight simultaneously; the underlying spyreccl broadcasts
        pipeline through the per-WorkSchedule fence.

        TP>2 raises until libspyre_comms gains a native allreduce.
        """
        if self.world_size == 1:
            return input_

        if self.world_size != 2:
            # REPLACE-WITH-NATIVE: once libspyre_comms supports allreduce
            # natively, delete this entire `all_reduce` override and let
            # the base class call `dist.all_reduce(input_, group=self.device_group)`.
            raise NotImplementedError(
                _spyre_collective_unsupported_message(
                    "allreduce",
                    self.world_size,
                    blocker="libspyre_comms native allreduce impl",
                )
            )

        # dist.broadcast requires contiguous storage.
        assert input_.is_contiguous(), (
            "SpyreCommunicator.all_reduce requires a contiguous input tensor; "
            f"got shape={tuple(input_.shape)} stride={input_.stride()}"
        )

        # The Spyre comms message matcher pairs broadcasts in issue order
        # across ranks, so both ranks must agree on issue order:
        #   1. broadcast with src=ranks[0]   (rank 0 sends, rank 1 recvs)
        #   2. broadcast with src=ranks[1]   (rank 1 sends, rank 0 recvs)
        # Replacing the old send/recv + single broadcast pattern with two
        # broadcasts removes the data dependency between recv and the
        # following broadcast on rank 0, which previously prevented
        # multiple in-flight allreduces from pipelining.
        peer = torch.empty_like(input_)
        if self.rank_in_group == 0:
            w_self = dist.broadcast(
                input_, src=self.ranks[0],
                group=self.device_group, async_op=async_op,
            )
            w_peer = dist.broadcast(
                peer, src=self.ranks[1],
                group=self.device_group, async_op=async_op,
            )
        else:
            w_peer = dist.broadcast(
                peer, src=self.ranks[0],
                group=self.device_group, async_op=async_op,
            )
            w_self = dist.broadcast(
                input_, src=self.ranks[1],
                group=self.device_group, async_op=async_op,
            )

        if async_op:
            return _SpyreAllReduceWork(
                input_=input_, peer=peer, w_self=w_self, w_peer=w_peer,
            )

        # Sync path: dist.broadcast already returned, peer is filled.
        input_.add_(peer)
        return input_

    def all_gather(self, input_: torch.Tensor, dim: int = -1) -> torch.Tensor:
        # The base class uses dist.all_gather_into_tensor which needs
        # _allgather_base in spyreccl is still stubbed. Use list-form
        # dist.all_gather instead (natively supported).
        # REPLACE-WITH-NATIVE: when torch-spyre wires up _allgather_base,
        # delete this override and let the base class handle it.
        if self.world_size == 1:
            return input_
        if input_.device.type == "cpu":
            return super().all_gather(input_, dim)
        output_list = [torch.empty_like(input_) for _ in range(self.world_size)]
        dist.all_gather(output_list, input_, group=self.device_group)
        return torch.cat(output_list, dim=dim)

    def reduce_scatter(self, input_: torch.Tensor, dim: int = -1) -> torch.Tensor:
        # Not on the standard TP path; raise loudly if anything tries it.
        if self.world_size == 1:
            return input_
        raise NotImplementedError(
            _spyre_collective_unsupported_message("reduce_scatter", self.world_size)
        )

    # `broadcast`, `send`, `recv` from DeviceCommunicatorBase route through
    # ops that are implemented in libspyre_comms, so we leave them alone.
