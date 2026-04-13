import numpy as np
import rasterio
from rasterio.env import Env
from typing import Any, List, Tuple
from concurrent.futures import ThreadPoolExecutor
from . import geodium

def run_concurrent_pipeline(
    output_path: str,
    band_sources: List[Tuple[str, int]],
    compiled_expr: Any,
):
    # 1. Group requests by file path so we only open each file once
    file_map = {}
    for pipeline_idx, (fpath, band_idx) in enumerate(band_sources):
        if fpath not in file_map:
            file_map[fpath] = {
                'src': rasterio.open(fpath),
                'bands': [],
                'dest_indices':[] # Maps back to the expected order for the Rust kernel
            }
        file_map[fpath]['bands'].append(band_idx)
        file_map[fpath]['dest_indices'].append(pipeline_idx)

    # Limit cache to prevent memory ballooning
    with Env(GDAL_CACHEMAX=1024 * 1024):
        try:
            # Grab profile and windows from the first available source
            first_src = list(file_map.values())[0]['src']
            profile = first_src.profile.copy()
            profile.update(
                dtype='float32', count=1, compress='lzw', 
                predictor=3, tiled=True, blockxsize=256, blockysize=256
            )
            
            windows =[w for _, w in first_src.block_windows(1)]
            max_h = max(w.height for w in windows)
            max_w = max(w.width for w in windows)
            
            out_buffers =[
                np.zeros((max_h, max_w), dtype=np.float32, order='C'),
                np.zeros((max_h, max_w), dtype=np.float32, order='C')
            ]
            
            # 2. Pre-allocate Grouped Input Buffers
            # Shape is (number_of_bands_requested_from_this_file, height, width)
            grouped_in_buffers =[
                {fpath: np.zeros((len(info['bands']), max_h, max_w), dtype=np.uint16) 
                 for fpath, info in file_map.items()},
                {fpath: np.zeros((len(info['bands']), max_h, max_w), dtype=np.uint16) 
                 for fpath, info in file_map.items()}
            ]

            def read_fn(window, buf_idx):
                h, w = window.height, window.width
                # This list will hold the views passed to the Rust kernel
                tiles = [None] * len(band_sources) 
                
                for fpath, info in file_map.items():
                    # Get the pre-allocated multi-band buffer for this file
                    target_buf = grouped_in_buffers[buf_idx][fpath][:, :h, :w]
                    
                    # 3. Read ALL required bands from this file in a single disk operation!
                    info['src'].read(info['bands'], window=window, out=target_buf)
                    
                    # Map the results back to the correct index for the Rust Expression
                    for i, pipeline_idx in enumerate(info['dest_indices']):
                        tiles[pipeline_idx] = target_buf[i]
                        
                return tiles, window

            with rasterio.open(output_path, 'w', **profile) as dst:
                with ThreadPoolExecutor(max_workers=2) as pool:
                    in_flight_buf_idx = 0
                    future_read = pool.submit(read_fn, windows[0], in_flight_buf_idx)
                    future_write = None
                    out_buf_idx = 0

                    for i in range(len(windows)):
                        tiles, current_win = future_read.result()
                        
                        if i + 1 < len(windows):
                            in_flight_buf_idx = 1 - in_flight_buf_idx
                            future_read = pool.submit(read_fn, windows[i + 1], in_flight_buf_idx)

                        h, w = tiles[0].shape
                        active_out_buf = out_buffers[out_buf_idx][:h, :w]
                        
                        geodium.execute_expr_inplace(compiled_expr, tiles, active_out_buf)

                        if future_write: 
                            future_write.result()
                        
                        future_write = pool.submit(dst.write, active_out_buf, 1, window=current_win)
                        out_buf_idx = 1 - out_buf_idx

                    if future_write: 
                        future_write.result()
        finally:
            for info in file_map.values(): 
                info['src'].close()