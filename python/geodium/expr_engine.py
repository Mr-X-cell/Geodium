# expr_engine.py
from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import numpy as np
from . import geodium
from .lazy import LazyNode, LazyBand

try:
    import spyndex
    HAS_SPYNDEX = True
except ImportError:
    HAS_SPYNDEX = False

def _require_spyndex():
    if not HAS_SPYNDEX:
        raise ImportError(
            "The 'spyndex' library is required to use the formula compiler. "
            "Install it via: pip install spyndex"
        )

# -------------------------------------------------------------------------
# AST Walker Logic
# -------------------------------------------------------------------------

def _ast_to_bytecode(
    node: ast.expr, 
    state: dict, 
    band_mapping: Dict[str, Union[LazyBand, int]], 
    scalar_consts: Dict[str, float]
) -> None:
    """
    Recursively walks the Python AST and emits bytecode instructions 
    into the 'state' dictionary used by the lazy/pipeline engine.
    """
    if isinstance(node, ast.BinOp):
        _ast_to_bytecode(node.left, state, band_mapping, scalar_consts)
        _ast_to_bytecode(node.right, state, band_mapping, scalar_consts)
        op_map = {
            ast.Add: "add", ast.Sub: "sub", ast.Mult: "mul",
            ast.Div: "div", ast.Pow: "pow",
        }
        if type(node.op) not in op_map:
            raise NotImplementedError(f"Unsupported operator: {type(node.op).__name__}")
        state["instructions"].append({"type": op_map[type(node.op)]})

    elif isinstance(node, ast.UnaryOp):
        if not isinstance(node.op, ast.USub):
            raise NotImplementedError("Only unary minus (-) is supported.")
        _ast_to_bytecode(node.operand, state, band_mapping, scalar_consts)
        state["instructions"].append({"type": "neg"})

    elif isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in ("abs", "sqrt"):
            raise NotImplementedError("Only abs() and sqrt() are supported.")
        _ast_to_bytecode(node.args[0], state, band_mapping, scalar_consts)
        state["instructions"].append({"type": node.func.id})

    elif isinstance(node, ast.Name):
        name = node.id
        if name in scalar_consts:
            state["instructions"].append({"type": "push_scalar", "value": scalar_consts[name]})
        elif name in band_mapping:
            target = band_mapping[name]
            if isinstance(target, LazyNode):
                # If it's a LazyBand, let it register itself in the global state
                target.compile_graph(state)
            else:
                # Legacy support: if it's just an index (int), register it
                # this path is used by the non-lazy compile_index
                if name not in state.get("_legacy_names", []):
                    if "_legacy_names" not in state: state["_legacy_names"] = []
                    state["_legacy_names"].append(name)
                idx = state["_legacy_names"].index(name)
                state["instructions"].append({"type": "push_band", "index": idx})
        else:
            raise ValueError(f"Formula requires band/constant '{name}' but it was not provided.")

    elif isinstance(node, ast.Constant):
        if not isinstance(node.value, (int, float)):
            raise NotImplementedError("Only numeric constants supported.")
        state["instructions"].append({"type": "push_scalar", "value": float(node.value)})
    else:
        raise NotImplementedError(f"Unsupported AST node: {type(node).__name__}")


def _get_scalar_defaults(index_name: str, band_map: dict) -> dict[str, float]:
    """Retrieves default constant values (like L, C1, G) from Spyndex."""
    index = spyndex.indices[index_name]
    known_bands = set(spyndex.bands.keys())
    scalars: dict[str, float] = {}

    for token in index.bands:
        if token in band_map or token in known_bands:
            continue
        if token in spyndex.constants:
            default = getattr(spyndex.constants[token], "default", None)
            if default is not None:
                try:
                    scalars[token] = float(default)
                except (TypeError, ValueError):
                    pass
    return scalars

# -------------------------------------------------------------------------
# Lazy Implementation (The New Pipeline)
# -------------------------------------------------------------------------

class LazyIndex(LazyNode):
    """
    A LazyNode that wraps a Spyndex formula.
    Allows formulas to be treated as part of the Lazy DAG and executed 
    via the concurrent I/O pipeline.
    """
    def __init__(self, index_name: str, band_mapping: Dict[str, LazyBand]):
        _require_spyndex()
        if index_name not in spyndex.indices:
            raise KeyError(f"{index_name!r} is not a known Spyndex index.")
        
        self.index_name = index_name
        self.band_mapping = band_mapping

    def compile_graph(self, state: dict):
        formula = spyndex.indices[self.index_name].formula
        scalar_consts = _get_scalar_defaults(self.index_name, self.band_mapping)
        
        tree = ast.parse(formula, mode="eval")
        _ast_to_bytecode(tree.body, state, self.band_mapping, scalar_consts)

# -------------------------------------------------------------------------
# Immediate Execution (Legacy / In-Memory support)
# -------------------------------------------------------------------------

@dataclass
class CompiledIndex:
    rust_expr: Any
    band_names: list[str]
    index_name: str

def compile_index(index_name: str, band_map: dict[str, np.ndarray]) -> CompiledIndex:
    """Compiles a Spyndex formula for immediate use with in-memory numpy arrays."""
    _require_spyndex()
    formula = spyndex.indices[index_name].formula
    scalar_consts = _get_scalar_defaults(index_name, band_map)

    # Use a dummy state for the walker
    state = {"instructions": [], "_legacy_names": []}
    tree = ast.parse(formula, mode="eval")
    
    # In legacy mode, band_mapping values are just markers
    _ast_to_bytecode(tree.body, state, {k: i for i, k in enumerate(band_map.keys())}, scalar_consts)

    band_names = state["_legacy_names"]
    rust_expr = geodium.compile_expr(state["instructions"], len(band_names))

    return CompiledIndex(rust_expr, band_names, index_name)

def execute_index(compiled: CompiledIndex, band_map: dict[str, np.ndarray], out_buffer: np.ndarray) -> np.ndarray:
    """Executes a previously compiled index on numpy arrays."""
    ordered_bands = [band_map[name] for name in compiled.band_names]
    geodium.execute_expr_inplace(compiled.rust_expr, ordered_bands, out_buffer)
    return out_buffer

def compute_index(index_name: str, band_map: dict[str, np.ndarray], out_buffer: np.ndarray) -> np.ndarray:
    """One-shot in-memory computation."""
    compiled = compile_index(index_name, band_map)
    return execute_index(compiled, band_map, out_buffer)