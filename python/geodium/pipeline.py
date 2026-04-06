import numpy as np
import rasterio
from typing import Any, List, Tuple # <--- Added Any, List, Tuple
from concurrent.futures import ThreadPoolExecutor
from . import geodium

def run_concurrent_pipeline(
    output_path: str,
    band_sources: List[Tuple[str, int]],  # List of (filepath, 1-based index)
    compiled_expr: Any,
    tile_size: int = 4096
):
    """The unified execution engine for all Geodium operations."""
    srcs = [rasterio.open(fp) for fp, _ in band_sources]
    band_indices = [idx for _, idx in band_sources]
    
    try:
        # Build profile from the first source
        profile = srcs[0].profile.copy()
        profile.update(
            dtype='float32', count=1, compress='lzw', 
            predictor=3, tiled=True, blockxsize=256, blockysize=256
        )
        
        windows = [w for _, w in srcs[0].block_windows(1)]
        buffers = [
            np.zeros((tile_size, tile_size), dtype=np.float32, order='C'),
            np.zeros((tile_size, tile_size), dtype=np.float32, order='C')
        ]

        def read_fn(window):
            return [src.read(idx, window=window, out_dtype='uint16') 
                    for src, idx in zip(srcs, band_indices)], window

        with rasterio.open(output_path, 'w', **profile) as dst:
            with ThreadPoolExecutor(max_workers=len(srcs) + 1) as pool:
                future_read = pool.submit(read_fn, windows[0])
                future_write = None
                buf_idx = 0

                for i in range(len(windows)):
                    tiles, current_win = future_read.result()
                    
                    if i + 1 < len(windows):
                        future_read = pool.submit(read_fn, windows[i + 1])

                    th, tw = tiles[0].shape
                    active_buf = buffers[buf_idx][:th, :tw]
                    
                    # Execute Rust Math Kernel
                    geodium.execute_expr_inplace(compiled_expr, tiles, active_buf)

                    if future_write: 
                        future_write.result()
                    
                    future_write = pool.submit(dst.write, active_buf, 1, window=current_win)
                    buf_idx = 1 - buf_idx

                if future_write: 
                    future_write.result()
    finally:
        for s in srcs: 
            s.close()