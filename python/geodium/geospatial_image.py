from __future__ import annotations

import psutil
import numpy as np
from . import geodium
from . import expr_engine

from enum import Enum
from pathlib import Path
from typing import Callable, Optional, Union
from concurrent.futures import ThreadPoolExecutor

try:
    import rasterio
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

try:
    from tqdm import tqdm
    _TQDM_AVAILABLE = True
except ImportError:
    _TQDM_AVAILABLE = False


class Collection(Enum):
    LANDSAT_8_9 = (2, 3, 4, 5,    6,    7)
    SENTINEL_2  = (2, 3, 4, 8,    11,   12)
    NAIP        = (3, 2, 1, 4,    None, None)
    PLANET      = (1, 2, 3, 4,    None, None)


class GeospatialImage:
    _MEM_THRESHOLD = 0.6

    def __init__(self, file_path: Optional[Union[str, Path]] = None, tile_size: int = 4096) -> None:
        if not HAS_RASTERIO:
            raise ImportError("GeospatialImage requires 'rasterio'. Install it via: pip install rasterio")

        self.path      = Path(file_path) if file_path else None
        self.tile_size = tile_size

        self._buffers = [
            np.zeros((tile_size, tile_size), dtype=np.float32, order="C"),
            np.zeros((tile_size, tile_size), dtype=np.float32, order="C"),
        ]

    def _get_strategy(self, width: int, height: int, num_bands: int) -> str:
        estimated_bytes = (width * height * 2 * num_bands) + (width * height * 4)
        available = psutil.virtual_memory().available
        return "IN_MEMORY" if estimated_bytes < available * self._MEM_THRESHOLD else "TILED"

    @staticmethod
    def _build_output_profile(src_profile: dict) -> dict:
        profile = src_profile.copy()
        profile.update(
            dtype="float32", count=1, compress="lzw", predictor=3,
            tiled=True, blockxsize=256, blockysize=256,
        )
        return profile

    def _run_pipeline(
        self,
        strategy: str,
        output_path: str,
        read_fn: Callable,
        compute_fn: Callable,
        windows: list,
        profile: dict,
        label: str = "",
    ) -> None:
        if not windows:
            return

        with rasterio.open(output_path, "w", **profile) as dst:
            if strategy == "IN_MEMORY":
                print(f"Executing IN_MEMORY pipeline for {label}")
                band_tiles, _ = read_fn(None)
                th, tw = band_tiles[0].shape
                
                out_buf = np.zeros((th, tw), dtype=np.float32, order="C")
                compute_fn(band_tiles, out_buf)
                dst.write(out_buf, 1)
                print(f"Saved: {output_path}")
                return

            print(f"Executing TILED concurrent pipeline for {label}")
            window_iter = (
                tqdm(range(len(windows)), desc=label or "Processing", unit="tile")
                if _TQDM_AVAILABLE else range(len(windows))
            )

            with ThreadPoolExecutor(max_workers=2) as io_pool:
                future_read  = io_pool.submit(read_fn, windows[0])
                future_write = None
                buffer_idx   = 0

                for i in window_iter:
                    band_tiles, current_window = future_read.result()
                    th, tw = band_tiles[0].shape
                    
                    if th > self.tile_size or tw > self.tile_size:
                        raise RuntimeError(f"Tile {th}x{tw} exceeds buffer {self.tile_size}x{self.tile_size}.")

                    if i + 1 < len(windows):
                        future_read = io_pool.submit(read_fn, windows[i + 1])

                    active_buf = self._buffers[buffer_idx][:th, :tw]
                    compute_fn(band_tiles, active_buf)

                    if future_write is not None:
                        future_write.result()

                    future_write = io_pool.submit(dst.write, active_buf, 1, window=current_window)
                    buffer_idx = 1 - buffer_idx

                if future_write is not None:
                    future_write.result()

        print(f"Saved: {output_path}")

    def process_separate_bands(self, output_path: str, band_a_path: str, band_b_path: str) -> None:
        with rasterio.open(band_a_path) as src_a, rasterio.open(band_b_path) as src_b:
            if src_a.shape != src_b.shape:
                raise ValueError("Dimensions mismatch")

            strategy = self._get_strategy(src_a.width, src_a.height, num_bands=2)
            profile = self._build_output_profile(src_a.profile)
            windows = [win for _, win in src_a.block_windows(1)]

            def read_fn(window):
                return [
                    src_a.read(1, window=window, out_dtype="uint16"),
                    src_b.read(1, window=window, out_dtype="uint16")
                ], window

            def compute_fn(tiles, out_buf):
                geodium.calculate_normalized_difference_lut_inplace(tiles[0], tiles[1], out_buf)

            self._run_pipeline(strategy, output_path, read_fn, compute_fn, windows, profile, Path(band_a_path).name)

    def _process_single_stack(self, output_path: str, band_a_idx: int, band_b_idx: int) -> None:
        if not self.path:
            raise ValueError("No input file provided.")

        with rasterio.open(self.path) as src:
            strategy = self._get_strategy(src.width, src.height, num_bands=2)
            profile = self._build_output_profile(src.profile)
            windows = [win for _, win in src.block_windows(1)]

            def read_fn(window):
                return [
                    src.read(band_a_idx, window=window, out_dtype="uint16"),
                    src.read(band_b_idx, window=window, out_dtype="uint16")
                ], window

            def compute_fn(tiles, out_buf):
                geodium.calculate_normalized_difference_lut_inplace(tiles[0], tiles[1], out_buf)

            self._run_pipeline(strategy, output_path, read_fn, compute_fn, windows, profile, self.path.name)

    def process_index_stack(self, output_path: str, index_name: str, band_map: dict[str, int]) -> None:
        if not self.path:
            raise ValueError("No input file provided.")

        dummy_map = {k: np.array([], dtype=np.uint16) for k in band_map.keys()}
        compiled = expr_engine.compile_index(index_name, dummy_map)
        rust_expected_order = compiled.band_names

        with rasterio.open(self.path) as src:
            strategy = self._get_strategy(src.width, src.height, num_bands=len(rust_expected_order))
            profile = self._build_output_profile(src.profile)
            windows = [win for _, win in src.block_windows(1)]

            def read_fn(window):
                tiles = [src.read(band_map[name], window=window, out_dtype="uint16") for name in rust_expected_order]
                return tiles, window

            def compute_fn(tiles, out_buf):
                geodium.execute_expr_inplace(compiled.rust_expr, tiles, out_buf)

            self._run_pipeline(strategy, output_path, read_fn, compute_fn, windows, profile, f"{index_name} on {self.path.name}")

    def ndvi(self, out: str, col: Collection) -> None:
        _, _, r, nir, _, _ = col.value
        self._process_single_stack(out, nir, r)

    def ndwi(self, out: str, col: Collection) -> None:
        _, g, _, nir, _, _ = col.value
        self._process_single_stack(out, g, nir)