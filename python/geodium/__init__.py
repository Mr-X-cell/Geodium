"""
geodium
-------
High-performance Rust-backed geospatial math engine.
"""

from __future__ import annotations
import importlib

# 1. CORE RUST ATTACHMENTS
# These are lightweight and always loaded to ensure the binary is linked correctly.
from .geodium import (
    calculate_normalized_difference,
    calculate_normalized_difference_inplace,
    compile_expr,
    execute_expr_inplace,
    CompiledExpr
)

# 2. COMPONENT MAPPING
# Map public names to the internal modules they live in for lazy loading.
_LAZY_EXPORTS = {
    # lazy.py
    "LazyBand": ".lazy",
    "LazyNode": ".lazy",
    "LazyScalar": ".lazy",
    
    # geospatial_image.py
    "GeospatialImage": ".geospatial_image",
    "Collection": ".geospatial_image",
    
    # expr_engine.py
    "LazyIndex": ".expr_engine",
    "compile_index": ".expr_engine",
    "execute_index": ".expr_engine",
    "compute_index": ".expr_engine",
}

# 3. PUBLIC API DEFINITION
__all__ = [
    # Rust Core
    "calculate_normalized_difference",
    "calculate_normalized_difference_inplace",
    "calculate_normalized_difference_lut_inplace",
    "compile_expr",
    "execute_expr_inplace",
    "CompiledExpr",
    # Lazy/DAG Engine
    "LazyBand",
    "LazyNode",
    "LazyScalar",
    "LazyIndex",
    # Imagery API
    "GeospatialImage",
    "Collection",
    # Formula Engine
    "compile_index",
    "compute_index",
]

# 4. LAZY LOAD IMPLEMENTATION
def __getattr__(name: str):
    """
    Dynamically imports components only when they are accessed.
    This keeps the initial 'import geodium' extremely fast.
    """
    if name in _LAZY_EXPORTS:
        module_path = _LAZY_EXPORTS[name]
        # import_module uses the relative path (e.g., .lazy) 
        # relative to this package (geodium)
        module = importlib.import_module(module_path, __package__)
        export = getattr(module, name)
        # Cache the result in the module's globals to avoid re-importing
        globals()[name] = export
        return export

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")