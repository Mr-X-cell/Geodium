from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Optional, Union, Tuple

from .lazy import LazyBand
from .expr_engine import LazyIndex

try:
    import rasterio
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

class Collection(Enum):
    # Mapping: (Blue, Green, Red, NIR, SWIR1, SWIR2)
    LANDSAT_8_9        = (2, 3, 4, 5, 6, 7)
    SENTINEL_2         = (2, 3, 4, 8, 11, 12)
    SENTINEL_2_CLIPPED = (1, 2, 3, 4, None, None)
    NAIP               = (1, 2, 3, 4, None, None)
    PLANET             = (1, 2, 3, 4, None, None)

    @staticmethod
    def discover(path: Path) -> Optional[Collection]:
        """Heuristically detect satellite source based on Resolution and Band Count."""
        if not HAS_RASTERIO: return None
        
        with rasterio.open(path) as src:
            res = round(abs(src.res[0]), 2)
            count = src.count
            
            if 25.0 <= res <= 31.0:
                return Collection.LANDSAT_8_9
            if 9.0 <= res <= 21.0:
                return Collection.SENTINEL_2 if count >= 12 else Collection.SENTINEL_2_CLIPPED
            if 3.0 <= res <= 5.0:
                return Collection.PLANET
            if res <= 1.5:
                return Collection.NAIP
        return None

class GeospatialImage:
    def __init__(self, file_path: Optional[Union[str, Path]] = None, 
                 collection: Optional[Collection] = None) -> None:
        if not HAS_RASTERIO:
            raise ImportError("GeospatialImage requires 'rasterio'.")

        self.path = Path(file_path) if file_path else None
        self.collection = collection or Collection.discover(self.path)
        
        if self.collection:
            print(f"Geodium: Using {self.collection.name} band mapping.")
        else:
            print("Geodium Warning: Could not auto-detect collection.")

    def _get_band(self, idx: int) -> LazyBand:
        if not self.path:
            raise ValueError("No input file provided to GeospatialImage.")
        return LazyBand(str(self.path), idx)

    def _resolve_indices(self, col: Optional[Collection]) -> Tuple:
        target_col = col or self.collection
        if not target_col:
            raise ValueError("No collection provided and auto-detection failed.")
        return target_col.value

    def ndvi(self, out: str, col: Optional[Collection] = None) -> None:
        indices = self._resolve_indices(col)
        r_idx, nir_idx = indices[2], indices[3]
        
        if r_idx is None or nir_idx is None:
            raise ValueError(f"Collection {self.collection} does not support NDVI.")

        expr = (self._get_band(nir_idx) - self._get_band(r_idx)) / \
               (self._get_band(nir_idx) + self._get_band(r_idx))
        
        expr.save(out) # No tile_size needed

    def ndwi(self, out: str, col: Optional[Collection] = None) -> None:
        indices = self._resolve_indices(col)
        g_idx, nir_idx = indices[1], indices[3]
        
        if g_idx is None or nir_idx is None:
            raise ValueError(f"Collection {self.collection} does not support NDWI.")

        expr = (self._get_band(g_idx) - self._get_band(nir_idx)) / \
               (self._get_band(g_idx) + self._get_band(nir_idx))
        
        expr.save(out)

    def process_index(self, output_path: str, index_name: str, 
                      band_map: Optional[dict[str, int]] = None) -> None:
        if band_map is None:
            if not self.collection:
                raise ValueError("No band_map provided and auto-detection failed.")
            
            vals = self.collection.value
            band_map = {
                "B": vals[0], "G": vals[1], "R": vals[2], 
                "N": vals[3], "S1": vals[4], "S2": vals[5]
            }
            band_map = {k: v for k, v in band_map.items() if v is not None}

        lazy_mapping = {name: self._get_band(idx) for name, idx in band_map.items()}
        LazyIndex(index_name, lazy_mapping).save(output_path)

    def compute(self, expression: str, output_path: str):
        self.process_index(output_path, expression.upper())