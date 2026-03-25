"""
expr_engine.py
--------------
Bridges Spyndex's formula registry and the Rust stack-machine evaluator.

Pipeline
~~~~~~~~
1. `compile_index(index_name, band_map)`
   Looks up the Spyndex formula, parses it with Python's `ast` module into a
   post-order bytecode list, validates it, and calls `ggs_package.compile_expr`
   to produce a `CompiledExpr` handle.  This step runs once per index name.

2. `execute_index(compiled, band_map, out_buffer)`
   Unpacks the band arrays in the order the compiler expects and calls
   `ggs_package.execute_expr_inplace`.  This is the per-tile hot path.

3. `compute_index(index_name, band_map, out_buffer)`
   Convenience wrapper that compiles and executes in one call.  Use this for
   one-off calculations; use compile + execute separately inside tile loops.

Supported AST nodes
~~~~~~~~~~~~~~~~~~~
Binary operators : + - * / ** (mapped to add/sub/mul/div/pow)
Unary operators  : - (neg)
Built-in calls   : abs(), sqrt()
Names            : any Spyndex band role (N, R, G, B, S1, S2, L, C1, C2, ...)
Constants        : any numeric literal

Anything else raises NotImplementedError with the offending node type, so
adding new built-ins later is straightforward.

Formula constants (L, C1, C2, g, ...)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Spyndex exposes per-index default values for scalar parameters via
`index.parameters`.  This module substitutes those defaults automatically so
callers only need to supply actual band arrays.  If you want non-default scalar
values, pass them explicitly in `band_map` as plain Python floats — they will
be treated as per-call scalars rather than compile-time constants.

Example
~~~~~~~
    from expr_engine import compile_index, execute_index
    import numpy as np

    NIR  = np.random.randint(0, 10000, (4096, 4096), dtype=np.uint16)
    RED  = np.random.randint(0, 10000, (4096, 4096), dtype=np.uint16)
    BLUE = np.random.randint(0, 10000, (4096, 4096), dtype=np.uint16)
    out  = np.zeros((4096, 4096), dtype=np.float32)

    # NDVI — two bands
    compiled = compile_index("NDVI", {"N": NIR, "R": RED})
    execute_index(compiled, {"N": NIR, "R": RED}, out)

    # EVI — three bands + scalar defaults substituted automatically
    compiled = compile_index("EVI", {"N": NIR, "R": RED, "B": BLUE})
    execute_index(compiled, {"N": NIR, "R": RED, "B": BLUE}, out)
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import spyndex

import ggs_package


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

@dataclass
class _CompileState:
    """Accumulates bytecode instructions and tracks the band-name → index map."""
    instructions: list[dict]       = field(default_factory=list)
    band_names:   list[str]        = field(default_factory=list)   # insertion order
    scalar_consts: dict[str, float] = field(default_factory=dict)  # Spyndex defaults

    def band_index(self, name: str) -> int:
        """Return (or register) the positional index for a named band."""
        if name not in self.band_names:
            self.band_names.append(name)
        return self.band_names.index(name)

    def emit(self, instr: dict) -> None:
        self.instructions.append(instr)


def _ast_to_bytecode(node: ast.expr, state: _CompileState) -> None:
    """
    Recursively walk an AST node and emit post-order stack-machine instructions
    into `state`.  Raises NotImplementedError for unsupported constructs.
    """
    if isinstance(node, ast.BinOp):
        _ast_to_bytecode(node.left,  state)
        _ast_to_bytecode(node.right, state)
        op_map = {
            ast.Add:  "add",
            ast.Sub:  "sub",
            ast.Mult: "mul",
            ast.Div:  "div",
            ast.Pow:  "pow",
        }
        op_type = type(node.op)
        if op_type not in op_map:
            raise NotImplementedError(
                f"Unsupported binary operator {op_type.__name__!r} in formula. "
                f"Supported: +, -, *, /, **"
            )
        state.emit({"type": op_map[op_type]})

    elif isinstance(node, ast.UnaryOp):
        if not isinstance(node.op, ast.USub):
            raise NotImplementedError(
                f"Unsupported unary operator {type(node.op).__name__!r}. "
                f"Only unary minus (-) is supported."
            )
        _ast_to_bytecode(node.operand, state)
        state.emit({"type": "neg"})

    elif isinstance(node, ast.Call):
        # Only abs() and sqrt() are currently supported
        if not isinstance(node.func, ast.Name):
            raise NotImplementedError(
                f"Only simple built-in calls are supported (abs, sqrt). "
                f"Got: {ast.dump(node.func)}"
            )
        func_name = node.func.id
        if func_name not in ("abs", "sqrt"):
            raise NotImplementedError(
                f"Unsupported function call {func_name!r}. "
                f"Supported: abs(), sqrt()"
            )
        if len(node.args) != 1 or node.keywords:
            raise NotImplementedError(
                f"{func_name}() must have exactly one positional argument."
            )
        _ast_to_bytecode(node.args[0], state)
        state.emit({"type": func_name})

    elif isinstance(node, ast.Name):
        name = node.id
        # Check if this is a Spyndex scalar constant (L, C1, C2, g, etc.)
        if name in state.scalar_consts:
            state.emit({"type": "push_scalar", "value": state.scalar_consts[name]})
        else:
            # Treat as a band array
            idx = state.band_index(name)
            state.emit({"type": "push_band", "index": idx})

    elif isinstance(node, ast.Constant):
        if not isinstance(node.value, (int, float)):
            raise NotImplementedError(
                f"Only numeric constants are supported. Got {type(node.value).__name__!r}."
            )
        state.emit({"type": "push_scalar", "value": float(node.value)})

    else:
        raise NotImplementedError(
            f"Unsupported AST node type {type(node).__name__!r}. "
            f"Formula may contain constructs this engine does not support."
        )


def _get_scalar_defaults(index_name: str, band_map: dict[str, Any]) -> dict[str, float]:
    """
    Extract Spyndex scalar constant defaults for an index, excluding any
    names the caller has already supplied in band_map.

    Scalar constants (L, C1, C2, g, ...) are looked up in spyndex.constants.
    The index's `bands` attribute lists ALL parameter tokens in the formula —
    band roles AND scalar tokens.  We identify scalars as tokens that appear
    in spyndex.constants but NOT in spyndex.bands.
    """
    index       = spyndex.indices[index_name]
    known_bands = set(spyndex.bands.keys())
    scalars: dict[str, float] = {}

    for token in index.bands:
        # Skip tokens the caller already supplied
        if token in band_map:
            continue
        # Skip genuine band roles — they must be supplied by the caller
        if token in known_bands:
            continue
        # Look up in spyndex.constants for the default value
        if token in spyndex.constants:
            constant = spyndex.constants[token]
            default  = getattr(constant, "default", None)
            if default is not None:
                try:
                    scalars[token] = float(default)
                except (TypeError, ValueError):
                    pass  # non-numeric default — will error at parse time if used

    return scalars


# ---------------------------------------------------------------------------
# Compiled index handle
# ---------------------------------------------------------------------------

@dataclass
class CompiledIndex:
    """
    Opaque handle returned by `compile_index`.
    Holds the Rust CompiledExpr and the ordered band name list so
    `execute_index` can assemble the band array list in the right order.
    """
    rust_expr:  Any        # ggs_package.CompiledExpr
    band_names: list[str]  # ordered list matching the Rust band index slots
    index_name: str        # for error messages


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compile_index(index_name: str, band_map: dict[str, np.ndarray]) -> CompiledIndex:
    """
    Parse the Spyndex formula for `index_name` and compile it to a Rust
    `CompiledExpr`.

    Parameters
    ----------
    index_name : str
        Any index name recognised by Spyndex (e.g. "NDVI", "EVI", "SAVI").
    band_map : dict[str, np.ndarray]
        Mapping of Spyndex band role names to uint16 numpy arrays.
        Scalar parameters (L, C1, C2, g, ...) are substituted automatically
        from Spyndex defaults; pass them in band_map as plain floats to
        override defaults.

    Returns
    -------
    CompiledIndex
        An opaque handle suitable for repeated calls to `execute_index`.

    Raises
    ------
    KeyError
        If `index_name` is not found in the Spyndex registry.
    NotImplementedError
        If the formula contains AST constructs this engine does not support.
    """
    if index_name not in spyndex.indices:
        raise KeyError(
            f"{index_name!r} is not a known Spyndex index. "
            f"Check spelling or consult spyndex.indices."
        )

    index   = spyndex.indices[index_name]
    formula = index.formula

    # Substitute Spyndex scalar defaults so the AST walker sees them as
    # compile-time constants rather than band names.
    scalar_consts = _get_scalar_defaults(index_name, band_map)

    # Parse the formula string into a Python AST
    try:
        tree = ast.parse(formula, mode="eval")
    except SyntaxError as exc:
        raise ValueError(
            f"Failed to parse Spyndex formula for {index_name!r}: {formula!r}\n"
            f"Syntax error: {exc}"
        ) from exc

    state = _CompileState(scalar_consts=scalar_consts)
    _ast_to_bytecode(tree.body, state)

    # Validate that every band referenced in the bytecode was supplied
    missing = [name for name in state.band_names if name not in band_map]
    if missing:
        raise ValueError(
            f"Formula for {index_name!r} references band(s) {missing} "
            f"that are not present in band_map. "
            f"Supplied keys: {list(band_map.keys())}"
        )

    rust_expr = ggs_package.compile_expr(state.instructions, len(state.band_names))

    return CompiledIndex(
        rust_expr  = rust_expr,
        band_names = state.band_names,
        index_name = index_name,
    )


def execute_index(
    compiled:   CompiledIndex,
    band_map:   dict[str, np.ndarray],
    out_buffer: np.ndarray,
) -> np.ndarray:
    """
    Execute a `CompiledIndex` against a set of band arrays, writing results
    into `out_buffer`.

    Parameters
    ----------
    compiled : CompiledIndex
        Handle returned by `compile_index`.
    band_map : dict[str, np.ndarray]
        Same band arrays as passed to `compile_index`.  Must contain all band
        names referenced during compilation.
    out_buffer : np.ndarray
        Pre-allocated float32 C-contiguous output array matching band shape.

    Returns
    -------
    np.ndarray
        `out_buffer`, filled with the index values.
    """
    # Assemble bands in the order the Rust compiler assigned indices
    try:
        ordered_bands = [band_map[name] for name in compiled.band_names]
    except KeyError as exc:
        raise ValueError(
            f"Band {exc} required by compiled index {compiled.index_name!r} "
            f"is missing from band_map."
        ) from exc

    ggs_package.execute_expr_inplace(compiled.rust_expr, ordered_bands, out_buffer)
    return out_buffer


def compute_index(
    index_name: str,
    band_map:   dict[str, np.ndarray],
    out_buffer: np.ndarray,
) -> np.ndarray:
    """
    Compile and execute a Spyndex index in one call.

    Convenience wrapper for one-off calculations.  For tile loops, call
    `compile_index` once outside the loop and `execute_index` inside it.

    Parameters
    ----------
    index_name : str
        Any Spyndex index name (e.g. "NDVI", "EVI", "NBR").
    band_map : dict[str, np.ndarray]
        uint16 band arrays keyed by Spyndex band role names.
    out_buffer : np.ndarray
        Pre-allocated float32 output array.

    Returns
    -------
    np.ndarray
        `out_buffer`, filled with the computed index values.
    """
    compiled = compile_index(index_name, band_map)
    return execute_index(compiled, band_map, out_buffer)
