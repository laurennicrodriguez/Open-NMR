"""Processing history: every processing step is recorded and replayable.

Each public processing function is registered with :func:`processing_step`. Calling it
returns a new dataset whose ``history`` is the input's history plus one record::

    {"step": "apodize", "params": {"kind": "em", "lb": 0.3, ...}}

All parameters, including defaults, are recorded, so a recipe replays identically even
if defaults change later. Steps that make data-dependent decisions (automatic phasing)
also record a ``result``; on exact replay, a step with a ``replay`` rule is re-run from
that result (e.g. ``autophase`` becomes ``phase`` with the stored angles).

A recipe is the list of records, saved as JSON::

    {"nmrsuite_recipe": 1, "steps": [...]}
"""

from __future__ import annotations

import functools
import inspect
import json
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from nmrsuite.data import NMRData

RECIPE_VERSION = 1

ReplayRule = Callable[[dict[str, Any], dict[str, Any]], tuple[str, dict[str, Any]]]


@dataclass(frozen=True)
class StepSpec:
    name: str
    func: Callable[..., NMRData]
    replay: ReplayRule | None = None


_REGISTRY: dict[str, StepSpec] = {}


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, tuple | list):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, Path):
        return str(value)
    return value


def processing_step(name: str, replay: ReplayRule | None = None) -> Callable:
    """Register a processing function and record each call in the dataset's history.

    The function takes the dataset as its only positional argument and keyword
    parameters, and returns a new dataset, or ``(dataset, result_dict)``.
    """

    def decorate(func: Callable[..., Any]) -> Callable[..., NMRData]:
        signature = inspect.signature(func)

        @functools.wraps(func)
        def wrapper(nmr: NMRData, /, **params: Any) -> NMRData:
            bound = signature.bind(nmr, **params)
            bound.apply_defaults()
            recorded = {k: v for k, v in bound.arguments.items() if k != "nmr"}
            out = func(nmr, **params)
            result = None
            if isinstance(out, tuple):
                out, result = out
            record: dict[str, Any] = {"step": name, "params": _jsonable(recorded)}
            if result:
                record["result"] = _jsonable(result)
            out.history = [*nmr.history, record]
            return out

        if name in _REGISTRY:
            raise ValueError(f"processing step {name!r} registered twice")
        _REGISTRY[name] = StepSpec(name, wrapper, replay)
        wrapper.step_name = name  # type: ignore[attr-defined]
        return wrapper

    return decorate


def registered_steps() -> dict[str, StepSpec]:
    _load_builtin_steps()
    return dict(_REGISTRY)


def _load_builtin_steps() -> None:
    import nmrsuite.processing  # noqa: F401  (registers the built-in steps)


def apply_step(nmr: NMRData, name: str, params: dict[str, Any] | None = None) -> NMRData:
    _load_builtin_steps()
    try:
        spec = _REGISTRY[name]
    except KeyError:
        raise KeyError(f"unknown processing step {name!r}") from None
    return spec.func(nmr, **(params or {}))


def replay(nmr: NMRData, steps: Iterable[dict[str, Any]], *, exact: bool = True) -> NMRData:
    """Apply recorded steps to (raw) data.

    With ``exact=True`` steps that recorded a result and have a replay rule are applied
    from that result (bit-for-bit the same decision); otherwise every step is re-run.
    """
    _load_builtin_steps()
    for record in steps:
        name, params = record["step"], dict(record.get("params", {}))
        spec = _REGISTRY.get(name)
        if spec is None:
            raise KeyError(f"unknown processing step {name!r}")
        if exact and spec.replay is not None and record.get("result"):
            name, params = spec.replay(params, record["result"])
        nmr = apply_step(nmr, name, params)
    return nmr


def recipe_to_json(steps: Iterable[dict[str, Any]], indent: int | None = 2) -> str:
    return json.dumps({"nmrsuite_recipe": RECIPE_VERSION, "steps": list(steps)}, indent=indent)


def recipe_from_json(text: str) -> list[dict[str, Any]]:
    doc = json.loads(text)
    if isinstance(doc, list):
        return doc
    version = doc.get("nmrsuite_recipe")
    if version != RECIPE_VERSION:
        raise ValueError(f"unsupported recipe version {version!r}")
    return list(doc["steps"])


def save_recipe(steps: Iterable[dict[str, Any]], path: str | Path) -> None:
    Path(path).write_text(recipe_to_json(steps) + "\n", encoding="utf-8")


def load_recipe(path: str | Path) -> list[dict[str, Any]]:
    return recipe_from_json(Path(path).read_text(encoding="utf-8"))
