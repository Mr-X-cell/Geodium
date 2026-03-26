import numpy as np
import geodium
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

# -------------------------------------------------------------------------
# Optional Dependency Guard
# -------------------------------------------------------------------------
try:
    import rasterio
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

def _require_rasterio():
    if not HAS_RASTERIO:
        raise ImportError(
            "The 'rasterio' library is required to execute lazy graphs to disk. "
            "Install it via: pip install rasterio (or pip install geodium[io])"
        )
# -------------------------------------------------------------------------

# Helper to ensure numbers become LazyScalars
def _wrap(val):
    if isinstance(val, (int, float)):
        return LazyScalar(float(val))
    if isinstance(val, LazyNode):
        return val
    raise TypeError(f"Unsupported type for lazy evaluation: {type(val)}")

class LazyNode:
    """Base class that overloads Python's math operators to build an execution graph."""
    def __add__(self, other): return LazyBinOp("add", self, _wrap(other))
    def __radd__(self, other): return LazyBinOp("add", _wrap(other), self)
    def __sub__(self, other): return LazyBinOp("sub", self, _wrap(other))
    def __rsub__(self, other): return LazyBinOp("sub", _wrap(other), self)
    def __mul__(self, other): return LazyBinOp("mul", self, _wrap(other))
    def __rmul__(self, other): return LazyBinOp("mul", _wrap(other), self)
    def __truediv__(self, other): return LazyBinOp("div", self, _wrap(other))
    def __rtruediv__(self, other): return LazyBinOp("div", _wrap(other), self)
    def __pow__(self, other): return LazyBinOp("pow", self, _wrap(other))
    def __neg__(self): return LazyUnaryOp("neg", self)
    
    def abs(self): return LazyUnaryOp("abs", self)
    def sqrt(self): return LazyUnaryOp("sqrt", self)

    def compile_graph(self, state: dict):
        raise NotImplementedError

    def save(self, output_path: str):
        """Compiles the graph to Rust bytecode and executes the concurrent I/O pipeline."""
        execute_lazy_graph(self, output_path)

class LazyBand(LazyNode):
    """Represents a specific band inside a GeoTIFF on the hard drive."""
    def __init__(self, filepath: str, band_idx: int = 1):
        self.filepath = str(filepath)
        self.band_idx = band_idx
        self.key = f"{self.filepath}:{self.band_idx}"

    def compile_graph(self, state: dict):
        if self.key not in state["unique_bands"]:
            state["unique_bands"].append(self.key)
            state["band_sources"].append((self.filepath, self.band_idx))
        
        idx = state["unique_bands"].index(self.key)
        state["instructions"].append({"type": "push_band", "index": idx})

class LazyScalar(LazyNode):
    def __init__(self, value: float):
        self.value = value

    def compile_graph(self, state: dict):
        state["instructions"].append({"type": "push_scalar", "value": self.value})

class LazyBinOp(LazyNode):
    def __init__(self, op: str, left: LazyNode, right: LazyNode):
        self.op = op
        self.left = left
        self.right = right

    def compile_graph(self, state: dict):
        self.left.compile_graph(state)
        self.right.compile_graph(state)
        state["instructions"].append({"type": self.op})

class LazyUnaryOp(LazyNode):
    def __init__(self, op: str, operand: LazyNode):
        self.op = op
        self.operand = operand

    def compile_graph(self, state: dict):
        self.operand.compile_graph(state)
        state["instructions"].append({"type": self.op})

def execute_lazy_graph(root_node: LazyNode, output_path: str, tile_size: int = 4096):
    _require_rasterio() # Guard against missing optional dependencies!
    
    state = {
        "instructions":[],
        "unique_bands": [],
        "band_sources":[] 
    }
    root_node.compile_graph(state)
    
    num_bands = len(state["band_sources"])
    print(f"Compiled Lazy Graph: {len(state['instructions'])} instructions, reading {num_bands} unique bands.")
    
    compiled_expr = geodium.compile_expr(state["instructions"], num_bands)
    
    srcs = [rasterio.open(fp) for fp, _ in state["band_sources"]]
    band_indices = [b_idx for _, b_idx in state["band_sources"]]
    
    base_shape = srcs[0].shape
    for src in srcs:
        if src.shape != base_shape:
            for s in srcs: s.close()
            raise ValueError("All input GeoTIFFs must have identical dimensions.")

    profile = srcs[0].profile.copy()
    profile.update(dtype='float32', count=1, compress='lzw', predictor=3, tiled=True, blockxsize=256, blockysize=256)
    windows = [w for _, w in srcs[0].block_windows(1)]

    buffers = [
        np.zeros((tile_size, tile_size), dtype=np.float32, order='C'),
        np.zeros((tile_size, tile_size), dtype=np.float32, order='C')
    ]

    def read_fn(window):
        return [src.read(b_idx, window=window, out_dtype='uint16') for src, b_idx in zip(srcs, band_indices)], window

    with rasterio.open(output_path, 'w', **profile) as dst:
        with ThreadPoolExecutor(max_workers=num_bands + 1) as io_pool:
            future_read = io_pool.submit(read_fn, windows[0])
            future_write = None
            buffer_idx = 0

            for i in range(len(windows)):
                tiles, current_window = future_read.result()

                if i + 1 < len(windows):
                    future_read = io_pool.submit(read_fn, windows[i + 1])

                th, tw = tiles[0].shape
                active_buf = buffers[buffer_idx][:th, :tw]

                geodium.execute_expr_inplace(compiled_expr, tiles, active_buf)

                if future_write is not None:
                    future_write.result()

                future_write = io_pool.submit(dst.write, active_buf, 1, window=current_window)
                buffer_idx = 1 - buffer_idx

            if future_write is not None:
                future_write.result()

    for src in srcs:
        src.close()
    print(f"Lazy Evaluation Complete. Saved to: {output_path}")