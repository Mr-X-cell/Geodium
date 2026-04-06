from __future__ import annotations # <--- Helps with type hint evaluation
from enum import Enum
from pathlib import Path
from typing import Optional, Union

from .lazy import LazyBand
from .expr_engine import LazyIndex

# Define Collection BEFORE the class so it is available for method signatures
class Collection(Enum):
    LANDSAT_8_9 = (2, 3, 4, 5,    6,    7)
    SENTINEL_2  = (2, 3, 4, 8,    11,   12)
    NAIP        = (3, 2, 1, 4,    None, None)
    PLANET      = (1, 2, 3, 4,    None, None)

class GeospatialImage:
    def __init__(self, file_path: Optional[Union[str, Path]] = None, tile_size: int = 4096) -> None:
        self.path      = Path(file_path) if file_path else None
        self.tile_size = tile_size

    def _get_band(self, idx: int) -> LazyBand:
        if not self.path:
            raise ValueError("No input file provided to GeospatialImage.")
        return LazyBand(str(self.path), idx)

    def ndvi(self, out: str, col: Collection) -> None:
        # Index 2=Red, 3=NIR in the Enum tuple (Landsat/Sentinel/Planet logic varies)
        # Based on your previous code: col.value = (B, G, R, NIR, SWIR1, SWIR2)
        _, _, r_idx, nir_idx, _, _ = col.value
        
        nir = self._get_band(nir_idx)
        red = self._get_band(r_idx)
        
        # Build DAG and Save
        expr = (nir - red) / (nir + red)
        expr.save(out, tile_size=self.tile_size)

    def ndwi(self, out: str, col: Collection) -> None:
        # Based on your previous code: col.value = (B, G, R, NIR, ...)
        _, g_idx, _, nir_idx, _, _ = col.value
        
        green = self._get_band(g_idx)
        nir   = self._get_band(nir_idx)
        
        expr = (green - nir) / (green + nir)
        expr.save(out, tile_size=self.tile_size)

    def process_index(self, output_path: str, index_name: str, band_map: dict[str, int]) -> None:
        """Process any Spyndex index using the unified pipeline."""
        lazy_mapping = {
            name: self._get_band(idx) 
            for name, idx in band_map.items()
        }
        
        # Create the Spyndex bridge node and execute
        LazyIndex(index_name, lazy_mapping).save(output_path, tile_size=self.tile_size)