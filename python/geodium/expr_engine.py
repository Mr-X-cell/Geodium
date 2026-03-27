from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from . import geodium

try:
    import spyndex
    HAS_SPYNDEX = True
except ImportError:
    HAS_SPYNDEX = False


def _require_spyndex():
    if not HAS_SPYNDEX:
        raise ImportError(
            "The 'spyndex' library is required to use the formula compiler. "
            "Install it via: pip install spyndex (or pip install geodium[formulas])"
        )


@dataclass
class _CompileState:
    instructions: list[dict]       = field(default_factory=list)
    band_names:   list[str]        = field(default_factory=list)
    scalar_consts: dict[str, float] = field(default_factory=dict)

    def band_index(self, name: str) -> int:
        if name not in self.band_names:
            self.band_names.append(name)
        return self.band_names.index(name)

    def emit(self, instr: dict) -> None:
        self.instructions.append(instr)


def _ast_to_bytecode(node: ast.expr, state: _CompileState) -> None:
    if isinstance(node, ast.BinOp):
        _ast_to_bytecode(node.left,  state)
        _ast_to_bytecode(node.right, state)
        op_map = {
            ast.Add: "add", ast.Sub: "sub", ast.Mult: "mul",
            ast.Div: "div", ast.Pow: "pow",
        }
        op_type = type(node.op)
        if op_type not in op_map:
            raise NotImplementedError(f"Unsupported operator: {op_type.__name__!r}")
        state.emit({"type": op_map[op_type]})

    elif isinstance(node, ast.UnaryOp):
        if not isinstance(node.op, ast.USub):
            raise NotImplementedError("Only unary minus (-) is supported.")
        _ast_to_bytecode(node.operand, state)
        state.emit({"type": "neg"})

    elif isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in ("abs", "sqrt"):
            raise NotImplementedError("Only abs() and sqrt() are supported.")
        _ast_to_bytecode(node.args[0], state)
        state.emit({"type": node.func.id})

    elif isinstance(node, ast.Name):
        name = node.id
        if name in state.scalar_consts:
            state.emit({"type": "push_scalar", "value": state.scalar_consts[name]})
        else:
            state.emit({"type": "push_band", "index": state.band_index(name)})

    elif isinstance(node, ast.Constant):
        if not isinstance(node.value, (int, float)):
            raise NotImplementedError("Only numeric constants are supported.")
        state.emit({"type": "push_scalar", "value": float(node.value)})
    else:
        raise NotImplementedError(f"Unsupported AST node: {type(node).__name__!r}")


def _get_scalar_defaults(index_name: str, band_map: dict[str, Any]) -> dict[str, float]:
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


@dataclass
class CompiledIndex:
    rust_expr:  Any
    band_names: list[str]
    index_name: str


def compile_index(index_name: str, band_map: dict[str, np.ndarray]) -> CompiledIndex:
    _require_spyndex()
    if index_name not in spyndex.indices:
        raise KeyError(f"{index_name!r} is not a known Spyndex index.")

    formula = spyndex.indices[index_name].formula
    scalar_consts = _get_scalar_defaults(index_name, band_map)

    tree = ast.parse(formula, mode="eval")
    state = _CompileState(scalar_consts=scalar_consts)
    _ast_to_bytecode(tree.body, state)

    missing = [n for n in state.band_names if n not in band_map]
    if missing:
        raise ValueError(f"Missing bands required by formula: {missing}")

    rust_expr = geodium.compile_expr(state.instructions, len(state.band_names))

    return CompiledIndex(
        rust_expr  = rust_expr,
        band_names = state.band_names,
        index_name = index_name,
    )


def execute_index(
    compiled: CompiledIndex,
    band_map: dict[str, np.ndarray],
    out_buffer: np.ndarray,
) -> np.ndarray:
    ordered_bands = [band_map[name] for name in compiled.band_names]
    geodium.execute_expr_inplace(compiled.rust_expr, ordered_bands, out_buffer)
    return out_buffer


def compute_index(
    index_name: str,
    band_map: dict[str, np.ndarray],
    out_buffer: np.ndarray,
) -> np.ndarray:
    compiled = compile_index(index_name, band_map)
    return execute_index(compiled, band_map, out_buffer)