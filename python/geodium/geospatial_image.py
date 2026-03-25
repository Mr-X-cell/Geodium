import numpy as np
import rasterio
import geodium
import psutil
import expr_engine 

from enum import Enum
from typing import Optional, Union, Callable
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

class Collection(Enum):
    """Standardized Band Mappings (1-based indices)."""
    LANDSAT_8_9 = (2, 3, 4, 5, 6, 7)
    SENTINEL_2  = (2, 3, 4, 8, 11, 12) 
    NAIP        = (3, 2, 1, 4, None, None)
    PLANET      = (1, 2, 3, 4, None, None)

class GeospatialImage:
    def __init__(self, file_path: Optional[Union[str, Path]] = None, tile_size: int = 4096):
        self.path = Path(file_path) if file_path else None
        self.tile_size = tile_size
        
        self._buffers = [
            np.zeros((tile_size, tile_size), dtype=np.float32, order='C'),
            np.zeros((tile_size, tile_size), dtype=np.float32, order='C')
        ]

    def _get_strategy(self, width: int, height: int, num_bands: int) -> str:
        """Dynamically calculates RAM requirements based on the number of input bands."""
        # Memory = (uint16_bytes * num_bands) + (float32_bytes)
        estimated_mem = (width * height * 2 * num_bands) + (width * height * 4)
        return "IN_MEMORY" if estimated_mem < (psutil.virtual_memory().available * 0.6) else "TILED"

    @staticmethod
    def _build_profile(src_profile: dict) -> dict:
        """Ensures standard, compressed, tiled float32 output."""
        profile = src_profile.copy()
        profile.update(dtype='float32', count=1, compress='lzw', predictor=3, tiled=True)
        return profile

    def _run_pipeline(
        self, 
        output_path: str, 
        strategy: str,
        read_fn: Callable, 
        compute_fn: Callable, 
        windows: list, 
        profile: dict, 
        label: str
    ):
        """
        The universal core execution pipeline. 
        Decouples the thread pooling / buffering from the specific math being executed.
        """
        if not windows:
            return

        print(f"Executing [{strategy}] pipeline for: {label}")

        with rasterio.open(output_path, 'w', **profile) as dst:
            
            # --- FAST PATH: If the image fits entirely in RAM, skip threading overhead ---
            if strategy == "IN_MEMORY":
                tiles, _ = read_fn(None)  # None tells read_fn to read the whole array
                th, tw = tiles[0].shape
                out_buf = np.zeros((th, tw), dtype=np.float32, order="C")
                compute_fn(tiles, out_buf)
                dst.write(out_buf, 1)
                return

            # --- STANDARD PATH: Double-Buffered Concurrent I/O ---
            with ThreadPoolExecutor(max_workers=2) as io_pool:
                future_read = io_pool.submit(read_fn, windows[0])
                future_write = None
                buffer_idx = 0

                for i in range(len(windows)):
                    tiles, current_window = future_read.result()

                    if i + 1 < len(windows):
                        future_read = io_pool.submit(read_fn, windows[i + 1])
                    
                    th, tw = tiles[0].shape
                    if th > self.tile_size or tw > self.tile_size:
                        raise RuntimeError(f"Tile {th}x{tw} exceeds pre-allocated {self.tile_size} buffer.")

                    active_buf = self._buffers[buffer_idx][:th, :tw]
                    
                    # Execute Math (LUT or Stack VM)
                    compute_fn(tiles, active_buf)
                    
                    if future_write is not None:
                        future_write.result()

                    future_write = io_pool.submit(dst.write, active_buf, 1, window=current_window)
                    buffer_idx = 1 - buffer_idx
                    
                if future_write is not None:
                    future_write.result()

        print(f"Result saved to: {output_path}")

    # -------------------------------------------------------------------------
    # 2-Band Hand-Coded Fast Paths (Uses LUT Engine)
    # -------------------------------------------------------------------------

    def process_separate_bands(self, output_path: str, band_a_path: str, band_b_path: str):
        with rasterio.open(band_a_path) as src_a, rasterio.open(band_b_path) as src_b:
            if src_a.shape != src_b.shape:
                raise ValueError(f"Dimensions mismatch: {src_a.shape} vs {src_b.shape}")

            strategy = self._get_strategy(src_a.width, src_a.height, num_bands=2)
            profile = self._build_profile(src_a.profile)
            windows = [w for _, w in src_a.block_windows(1)]

            def read_fn(window):
                return [
                    src_a.read(1, window=window, out_dtype='uint16'),
                    src_b.read(1, window=window, out_dtype='uint16')
                ], window

            def compute_fn(tiles, out_buf):
                geodium.calculate_normalized_difference_lut_inplace(tiles[0], tiles[1], out_buf)

            self._run_pipeline(output_path, strategy, read_fn, compute_fn, windows, profile, Path(band_a_path).name)

    def _process_single_stack(self, output_path: str, band_a_idx: int, band_b_idx: int):
        if not self.path:
            raise ValueError("No input file provided during GeospatialImage initialization.")

        with rasterio.open(self.path) as src:
            strategy = self._get_strategy(src.width, src.height, num_bands=2)
            profile = self._build_profile(src.profile)
            windows = [w for _, w in src.block_windows(1)]

            def read_fn(window):
                return [
                    src.read(band_a_idx, window=window, out_dtype='uint16'),
                    src.read(band_b_idx, window=window, out_dtype='uint16')
                ], window

            def compute_fn(tiles, out_buf):
                geodium.calculate_normalized_difference_lut_inplace(tiles[0], tiles[1], out_buf)

            self._run_pipeline(output_path, strategy, read_fn, compute_fn, windows, profile, self.path.name)

    # -------------------------------------------------------------------------
    # N-Band Stack Machine Path (Uses expr_engine VM)
    # -------------------------------------------------------------------------

    def process_index_stack(self, output_path: str, index_name: str, band_map: dict[str, int]):
        """
        Evaluates an N-band Spyndex formula natively via the Rust VM.
        band_map example: {"N": 4, "R": 3, "B": 2}  (Spyndex token -> TIFF band index)
        """
        if not self.path:
            raise ValueError("No input file provided.")

        # Compile the index to get the exact memory layout the Rust VM expects
        dummy_map = {k: np.array([], dtype=np.uint16) for k in band_map.keys()}
        compiled = expr_engine.compile_index(index_name, dummy_map)
        expected_bands = compiled.band_names

        with rasterio.open(self.path) as src:
            strategy = self._get_strategy(src.width, src.height, num_bands=len(expected_bands))
            profile = self._build_profile(src.profile)
            windows = [w for _, w in src.block_windows(1)]

            def read_fn(window):
                tiles = [src.read(band_map[name], window=window, out_dtype='uint16') for name in expected_bands]
                return tiles, window

            def compute_fn(tiles, out_buf):
                geodium.execute_expr_inplace(compiled.rust_expr, tiles, out_buf)

            self._run_pipeline(output_path, strategy, read_fn, compute_fn, windows, profile, f"{index_name} on {self.path.name}")

    # -------------------------------------------------------------------------
    # Convenience Methods
    # -------------------------------------------------------------------------

    def ndvi(self, out: str, col: Collection):
        _, _, r, nir, _, _ = col.value
        self._process_single_stack(out, nir, r)

# Example logic
if __name__ == "__main__":
    proc = GeospatialImage()
    # proc.process_separate_bands("ndvi.tif", "red_band.tif", "nir_band.tif")

    # Example evaluating EVI through the VM engine:
    # proc = GeospatialImage("planet_stack.tif")
    # proc.process_index_stack("EVI", {"N": 4, "R": 3, "B": 1})