# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Benchmark integration for unmodified Factory entrypoints and native Warp storage."""
import dataclasses
import importlib
import inspect
import json
from functools import lru_cache

import torch
import warp as wp

_LAYOUT_FIELDS = (
    "articulation_body_offsets", "articulation_block_row_offsets", "articulation_block_cols",
    "articulation_block_col_offsets", "articulation_block_col_rows", "articulation_block_col_slots",
    "articulation_diag_slots", "levels",
)


@lru_cache(maxsize=2)
def _entry(kind):
    module = importlib.import_module(f".{kind}_winner", __package__)
    return module.run, list(inspect.signature(module.run).parameters)


@lru_cache(maxsize=16)
def _torch_stream(device_ordinal, cuda_stream):
    return torch.cuda.ExternalStream(cuda_stream, device=device_ordinal)


def _serialize(value, name, arrays):
    if isinstance(value, wp.array):
        arrays[name] = value
        return {"array": name, "warp_type": str(value.dtype), "ndim": value.ndim}
    if value is None or isinstance(value, (int, float, bool, str)):
        return value
    if isinstance(value, (tuple, list)):
        return list(value)
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "_cls"):
        return {"warp_struct": value._cls.key, "fields": {
            key: _serialize(getattr(value, key), f"{name}_{key}", arrays)
            for key in value._cls.vars
        }}
    if dataclasses.is_dataclass(value):
        return {"fields": {
            field.name: _serialize(getattr(value, field.name), f"{name}_{field.name}", arrays)
            for field in dataclasses.fields(value)
        }}
    raise TypeError((name, type(value)))


def _dispatch(kind, arrays, outputs, device, metadata):
    run, names = _entry(kind)
    stream = wp.get_stream(device)
    torch_stream = _torch_stream(device.ordinal, stream.cuda_stream)
    scratch_outputs = {}
    tensors = {}
    with torch.cuda.stream(torch_stream):
        for name in names:
            if name == "dispatch":
                tensors[name] = metadata
            elif name.startswith("out_"):
                output = outputs[name[4:]]
                temp = torch.empty_like(wp.to_torch(output))
                scratch_outputs[name] = temp
                tensors[name] = temp
            else:
                tensors[name] = wp.to_torch(arrays[name])
        run(*(tensors[name] for name in names))
        for name, temp in scratch_outputs.items():
            native = outputs[name[4:]]
            wp.copy(native, wp.from_torch(temp, dtype=native.dtype), stream=stream)


def generated_assembly(stock_kernel, joint_inputs, values, rhs, device, dim, block_dim):
    names = list(inspect.signature(stock_kernel.func).parameters)
    params = [*joint_inputs, values, rhs]
    arrays = {}
    spec = {name: _serialize(value, name, arrays) for name, value in zip(names, params, strict=True)}
    metadata = json.dumps({"params": spec, "options": {"dim": int(dim), "block_dim": int(block_dim)}},
                          separators=(",", ":"))
    _dispatch("assembly", arrays, {"values_scalar": values, "rhs_scalar": rhs}, device, metadata)


def generated_levels(plan, layout, values, rhs, delta, device, pivot_clamped):
    arrays = {}
    params = {name: _serialize(value, name, arrays) for name, value in (
        ("plan", plan), ("values", values), ("rhs", rhs), ("delta", delta),
        ("pivot_clamped", pivot_clamped),
    )}
    params["layout"] = {"fields": {
        key: _serialize(getattr(layout, key), "layout_" + key, arrays) for key in _LAYOUT_FIELDS
    }}
    params["plan"]["kind"] = "RigidArticulationLevelLaunchPlan"
    for phase in ("update", "forward", "backward"):
        params["plan"]["fields"][phase]["kind"] = "LevelLaunchPhaseTasks"
    metadata = json.dumps({"params": params, "options": {"block_dim": 128}}, separators=(",", ":"))
    _dispatch("levels", arrays, {"values": values, "delta": delta, "pivot_clamped": pivot_clamped},
              device, metadata)
