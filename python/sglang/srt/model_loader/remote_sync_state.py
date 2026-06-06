# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Infrastructure to synchronize model state that lives *outside* of
``named_parameters()`` across remote instances.

When a destination ("client") instance loads weights from a seed instance, it
only constructs the model via ``_initialize_model`` and then copies the seed's
parameters. However, model loading also produces derived state that is **not**
captured by ``named_parameters()``:

  * derived tensors bound as plain module attributes (e.g. DeepSeek MLA's
    ``self_attn.w_kc`` / ``w_vc`` / ``w_scale_k`` / ``w_scale_v``), which may be
    non-contiguous (transposed) views over a dedicated storage;
  * Python scalars / flags on modules (e.g. ``w_scale=1.0``,
    ``use_deep_gemm_bmm=True``);
  * custom Python attributes attached to *parameter tensor objects*
    (e.g. ``weight_scale_inv.format_ue8m0=True``).

This module provides an opt-in protocol (:class:`RemoteSyncStateMixin`) for
modules to declare such state, plus collection/serialization/apply helpers so
the remote-instance loaders can transfer it faithfully instead of re-deriving it
locally (which is order-sensitive and fragile, see ``post_load_weights`` vs
``process_weights_after_loading`` ordering).
"""

from __future__ import annotations

import logging
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# Bump when the wire format of the payload changes.
REMOTE_SYNC_VERSION = 1

# Reserved key under which the transfer-engine weight-info dict carries the
# sync-state manifest + metadata (so it travels alongside the per-parameter
# RDMA descriptors over HTTP/JSON without colliding with parameter names).
REMOTE_SYNC_RESERVED_KEY = "__remote_sync__"

# Custom attributes attached to *parameter tensor objects* (not modules) that
# must be carried alongside the weight data. They are addressed by the owning
# parameter's state_dict FQN.
REMOTE_SYNC_TENSOR_FLAGS: Tuple[str, ...] = ("format_ue8m0",)


@dataclass
class RemoteSyncState:
    """A module's extra state, split by transport medium.

    * ``tensors``: go through the pointer/byte channel (NCCL broadcast / RDMA),
      keyed by *local* attribute name.
    * ``metadata``: go through the object channel (pickle), keyed by *local*
      attribute name. Values must be picklable, non-tensor scalars.
    """

    tensors: Dict[str, torch.Tensor] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class RemoteSyncStateMixin:
    """Mixin for modules that own state outside ``named_parameters()``.

    Subclasses only need to declare ``_REMOTE_SYNC_ATTRS`` with the attribute
    names to synchronize. Routing between the tensor channel and the metadata
    channel happens at runtime based on the actual value type, so attributes
    with a union type (e.g. ``w_scale`` may be ``float`` or ``Tensor``) are
    handled correctly.
    """

    # A single declarative list; values are routed by runtime type.
    _REMOTE_SYNC_ATTRS: Tuple[str, ...] = ()

    def get_remote_sync_state(self) -> RemoteSyncState:
        state = RemoteSyncState()
        for name in self._REMOTE_SYNC_ATTRS:
            value = getattr(self, name, None)
            if value is None:
                # Not materialized on this path (e.g. feature disabled).
                continue
            if torch.is_tensor(value):
                state.tensors[name] = value
            else:
                state.metadata[name] = value
        return state

    def set_remote_sync_state(self, state: RemoteSyncState) -> None:
        for name, tensor in state.tensors.items():
            current = getattr(self, name, None)
            if (
                torch.is_tensor(current)
                and current.shape == tensor.shape
                and current.dtype == tensor.dtype
                and current.stride() == tensor.stride()
            ):
                # Reuse existing storage (e.g. a second sync during RL).
                current.copy_(tensor)
            else:
                setattr(self, name, tensor)
        for name, value in state.metadata.items():
            setattr(self, name, value)


# ----------------------------------------------------------------------------
# dtype helpers
# ----------------------------------------------------------------------------
def _dtype_to_str(dtype: torch.dtype) -> str:
    return str(dtype)


def _str_to_dtype(s: str) -> torch.dtype:
    dtype = getattr(torch, s.split(".")[-1])
    assert isinstance(dtype, torch.dtype), f"Invalid dtype string: {s}"
    return dtype


# ----------------------------------------------------------------------------
# storage (byte-level) views — faithfully handle non-contiguous tensors
# ----------------------------------------------------------------------------
def storage_byte_view(t: torch.Tensor) -> torch.Tensor:
    """Return a 1-D contiguous ``uint8`` view over the *entire* untyped storage
    of ``t``. Broadcasting / RDMA-reading this view transfers the exact bytes,
    so any stride / storage_offset / non-contiguity is preserved on rebuild."""
    storage = t.untyped_storage()
    view = torch.empty(0, dtype=torch.uint8, device=t.device)
    view.set_(storage, 0, (storage.nbytes(),), (1,))
    return view


def build_manifest_entry(name: str, t: torch.Tensor) -> Dict[str, Any]:
    return {
        "name": name,
        "dtype": _dtype_to_str(t.dtype),
        "shape": list(t.shape),
        "stride": list(t.stride()),
        "storage_offset": t.storage_offset(),
        "storage_nbytes": t.untyped_storage().nbytes(),
    }


def rebuild_tensor(byte_buf: torch.Tensor, entry: Dict[str, Any]) -> torch.Tensor:
    """Rebuild a (possibly non-contiguous) tensor from received storage bytes."""
    dtype = _str_to_dtype(entry["dtype"])
    t = torch.empty(0, dtype=dtype, device=byte_buf.device)
    t.set_(
        byte_buf.untyped_storage(),
        entry["storage_offset"],
        tuple(entry["shape"]),
        tuple(entry["stride"]),
    )
    return t


# ----------------------------------------------------------------------------
# collection (seed side)
# ----------------------------------------------------------------------------
def has_sync_providers(model: nn.Module) -> bool:
    for _, module in model.named_modules():
        if hasattr(module, "get_remote_sync_state"):
            return True
    return False


def collect_remote_sync_payload(
    model: nn.Module,
) -> Tuple["OrderedDict[str, torch.Tensor]", Dict[str, Any]]:
    """Collect everything that must be synced beyond ``named_parameters()``.

    Returns ``(extra_tensors, payload)`` where:
      * ``extra_tensors`` maps FQN -> tensor object (transient, for sending).
      * ``payload`` is a picklable dict: ``{"manifest": [...], "metadata": {...}}``.
    """
    extra_tensors: "OrderedDict[str, torch.Tensor]" = OrderedDict()
    module_attrs: Dict[str, Dict[str, Any]] = {}

    for module_name, module in model.named_modules():
        getter = getattr(module, "get_remote_sync_state", None)
        if getter is None:
            continue
        state = getter()
        for local_name, tensor in state.tensors.items():
            if tensor is None:
                continue
            extra_tensors[f"{module_name}.{local_name}"] = tensor
        if state.metadata:
            module_attrs[module_name] = dict(state.metadata)

    # Tensor-object flags (e.g. format_ue8m0), addressed by parameter FQN.
    # NOTE: keep_vars=True is required so we see the *actual* Parameter objects;
    # the default state_dict() detaches tensors, dropping custom python attrs.
    tensor_flags: Dict[str, Dict[str, Any]] = {}
    for name, tensor in model.state_dict(keep_vars=True).items():
        if not torch.is_tensor(tensor):
            continue
        present = {
            flag: getattr(tensor, flag)
            for flag in REMOTE_SYNC_TENSOR_FLAGS
            if hasattr(tensor, flag)
        }
        if present:
            tensor_flags[name] = present

    manifest = [build_manifest_entry(name, t) for name, t in extra_tensors.items()]
    payload = {
        "manifest": manifest,
        "metadata": {
            "module_attrs": module_attrs,
            "tensor_flags": tensor_flags,
            "header": {
                "version": REMOTE_SYNC_VERSION,
                "uses_sync_providers": has_sync_providers(model),
            },
        },
    }
    return extra_tensors, payload


# ----------------------------------------------------------------------------
# apply (client side)
# ----------------------------------------------------------------------------
def apply_tensor_flags(
    model: nn.Module, tensor_flags: Dict[str, Dict[str, Any]]
) -> None:
    if not tensor_flags:
        return
    # keep_vars=True so flags are set on the live Parameter objects, not on
    # detached copies that state_dict() would otherwise return.
    state_dict = model.state_dict(keep_vars=True)
    for name, flags in tensor_flags.items():
        tensor = state_dict.get(name, None)
        if tensor is None:
            logger.warning(
                "remote_sync: tensor flag target %s not found in state_dict", name
            )
            continue
        for flag, value in flags.items():
            setattr(tensor, flag, value)


def apply_remote_sync_state(
    model: nn.Module,
    manifest: List[Dict[str, Any]],
    recv_buffers: Dict[str, torch.Tensor],
    metadata: Dict[str, Any],
) -> None:
    """Rebuild extra tensors and route every piece of extra state back to the
    module / tensor it belongs to."""
    module_attrs: Dict[str, Dict[str, Any]] = metadata.get("module_attrs", {})
    tensor_flags: Dict[str, Dict[str, Any]] = metadata.get("tensor_flags", {})

    # 1) Group rebuilt extra tensors by owning module FQN.
    tensors_by_module: Dict[str, Dict[str, torch.Tensor]] = defaultdict(dict)
    for entry in manifest:
        fqn = entry["name"]
        module_fqn, _, attr = fqn.rpartition(".")
        tensors_by_module[module_fqn][attr] = rebuild_tensor(recv_buffers[fqn], entry)

    # 2) Dispatch tensors + module-level scalars to each module's protocol.
    for module_fqn in set(tensors_by_module) | set(module_attrs):
        try:
            module = model.get_submodule(module_fqn)
        except AttributeError:
            logger.warning("remote_sync: submodule %s not found", module_fqn)
            continue
        setter = getattr(module, "set_remote_sync_state", None)
        if setter is None:
            logger.warning(
                "remote_sync: module %s has no set_remote_sync_state", module_fqn
            )
            continue
        setter(
            RemoteSyncState(
                tensors=tensors_by_module.get(module_fqn, {}),
                metadata=module_attrs.get(module_fqn, {}),
            )
        )

    # 3) Tensor-object flags (addressed by parameter FQN).
    apply_tensor_flags(model, tensor_flags)


def payload_uses_sync_providers(payload: Dict[str, Any]) -> bool:
    return bool(
        payload.get("metadata", {}).get("header", {}).get("uses_sync_providers")
    )


# ----------------------------------------------------------------------------
# NCCL transport helpers
# ----------------------------------------------------------------------------
def nccl_send_remote_sync_state(model: nn.Module, group, device=None) -> None:
    """Seed side: broadcast the payload, then the storage bytes of each extra
    tensor in manifest order."""
    extra_tensors, payload = collect_remote_sync_payload(model)
    torch.distributed.broadcast_object_list(
        [payload], src=0, group=group, device=device
    )
    for entry in payload["manifest"]:
        view = storage_byte_view(extra_tensors[entry["name"]])
        torch.distributed.broadcast(view, src=0, group=group)


def nccl_recv_remote_sync_state(
    model: nn.Module, group, device
) -> Tuple[Dict[str, Any], Dict[str, torch.Tensor]]:
    """Client side: receive the payload and the storage bytes. Returns the
    payload and the per-tensor receive buffers (not yet applied)."""
    holder: List[Optional[Dict[str, Any]]] = [None]
    torch.distributed.broadcast_object_list(holder, src=0, group=group, device=device)
    payload = holder[0]
    assert payload is not None, "remote_sync: failed to receive payload"

    recv_buffers: Dict[str, torch.Tensor] = {}
    for entry in payload["manifest"]:
        recv_buffers[entry["name"]] = torch.empty(
            entry["storage_nbytes"], dtype=torch.uint8, device=device
        )
    for entry in payload["manifest"]:
        torch.distributed.broadcast(recv_buffers[entry["name"]], src=0, group=group)
    return payload, recv_buffers


def maybe_apply_remote_sync_state(
    model: nn.Module,
    payload: Dict[str, Any],
    recv_buffers: Dict[str, torch.Tensor],
) -> bool:
    """Apply received sync state if the seed declared sync providers.

    Returns True if applied (caller should *skip* the local ``post_load_weights``
    fixup), False otherwise (caller should fall back to ``_post_load_weights``).
    """
    if not payload_uses_sync_providers(payload):
        return False
    apply_remote_sync_state(
        model, payload["manifest"], recv_buffers, payload["metadata"]
    )
    return True
