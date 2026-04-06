import numpy as np
from . import geodium
from .pipeline import run_concurrent_pipeline

def _wrap(val):
    if isinstance(val, (int, float)): return LazyScalar(float(val))
    if isinstance(val, LazyNode): return val
    raise TypeError(f"Unsupported type: {type(val)}")

class LazyNode:
    def __add__(self, other): return LazyBinOp("add", self, _wrap(other))
    def __sub__(self, other): return LazyBinOp("sub", self, _wrap(other))
    def __mul__(self, other): return LazyBinOp("mul", self, _wrap(other))
    def __truediv__(self, other): return LazyBinOp("div", self, _wrap(other))
    def __pow__(self, other): return LazyBinOp("pow", self, _wrap(other))
    def __radd__(self, other): return LazyBinOp("add", _wrap(other), self)
    def __rsub__(self, other): return LazyBinOp("sub", _wrap(other), self)
    def __rmul__(self, other): return LazyBinOp("mul", _wrap(other), self)
    def __rtruediv__(self, other): return LazyBinOp("div", _wrap(other), self)
    
    def __neg__(self): return LazyUnaryOp("neg", self)

    def compile_graph(self, state: dict):
        raise NotImplementedError

    def save(self, output_path: str, tile_size: int = 4096):
        state = {"instructions": [], "unique_bands": [], "band_sources": []}
        self.compile_graph(state)
        
        # Compile instructions to Rust Bytecode
        compiled_expr = geodium.compile_expr(state["instructions"], len(state["unique_bands"]))
        
        # Dispatch to the unified pipeline
        run_concurrent_pipeline(
            output_path=output_path,
            band_sources=state["band_sources"],
            compiled_expr=compiled_expr,
            tile_size=tile_size
        )

class LazyBand(LazyNode):
    def __init__(self, filepath: str, band_idx: int):
        self.filepath, self.band_idx = str(filepath), band_idx
        self.key = f"{self.filepath}:{self.band_idx}"

    def compile_graph(self, state: dict):
        if self.key not in state["unique_bands"]:
            state["unique_bands"].append(self.key)
            state["band_sources"].append((self.filepath, self.band_idx))
        idx = state["unique_bands"].index(self.key)
        state["instructions"].append({"type": "push_band", "index": idx})

class LazyScalar(LazyNode):
    def __init__(self, value: float): self.value = value
    def compile_graph(self, state: dict):
        state["instructions"].append({"type": "push_scalar", "value": self.value})

class LazyBinOp(LazyNode):
    def __init__(self, op, left, right): self.op, self.left, self.right = op, left, right
    def compile_graph(self, state: dict):
        self.left.compile_graph(state); self.right.compile_graph(state)
        state["instructions"].append({"type": self.op})

class LazyUnaryOp(LazyNode):
    def __init__(self, op, operand): self.op, self.operand = op, operand
    def compile_graph(self, state: dict):
        self.operand.compile_graph(state)
        state["instructions"].append({"type": self.op})