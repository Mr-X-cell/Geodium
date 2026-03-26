"""
geodium
-------
High-performance Rust-backed geospatial math engine.
"""

# ALWAYS load the core Rust math functions. 
from .geodium import (
    calculate_normalized_difference,
    calculate_normalized_difference_inplace,
    calculate_normalized_difference_lut_inplace,
    compile_expr,
    execute_expr_inplace,
    CompiledExpr
)

__all__ = [
    "calculate_normalized_difference",
    "calculate_normalized_difference_inplace",
    "calculate_normalized_difference_lut_inplace",
    "compile_expr",
    "execute_expr_inplace",
    "CompiledExpr",
]

# LAZY LOAD the Python wrappers.
def __getattr__(name: str):
    # Route for lazy.py
    if name in ("LazyBand", "LazyNode", "LazyScalar"):
        from .lazy import LazyBand, LazyNode, LazyScalar
        if name == "LazyBand": return LazyBand
        if name == "LazyNode": return LazyNode
        if name == "LazyScalar": return LazyScalar

    # Route for geospatial_image.py
    if name in ("GeospatialImage", "Collection"):
        from .geospatial_image import GeospatialImage, Collection
        return GeospatialImage if name == "GeospatialImage" else Collection

    # Route for expr_engine.py
    if name in ("compile_index", "execute_index", "compute_index"):
        from .expr_engine import compile_index, execute_index, compute_index
        if name == "compile_index": return compile_index
        if name == "execute_index": return execute_index
        if name == "compute_index": return compute_index

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")