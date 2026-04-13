A python library for quickly doing spectral imaging with less ram usage.
There are 4 ways to use Geodium.

# Method 1 Direct kernal usage (Fastest method)
The Geodium library can handle some functions natively such as normalized diffrerence, ratios, and adjusted difference spectral imaging processes if the correct bands are provided. This can be done using the following code:
```
import geodium
NDVI=geodium.calculate_normalized_difference_inplace(nir_arr, red_arr, out_buf)
```

# Method 2 Geospatial module
The GeospatialImage module allows the user to process data from a .tiff Image, and to do a specific premade function on it. THe following satelite collections have been added LANDSAT_8_9,SENTINEL_2,NAIP, and PLANET. Currently the only two equations are premade this way `.ndvi` and `.ndwi`
```
from geodium import GeospatialImage, Collection
tiff = GeospatialImage("./path to input tiff")
tiff.ndvi("./path to output", Collection.LANDSAT_8_9)
```

# Method 3 Spyndex equations
If spyndex is imported Spyndex equations can be loaded into byteecode using the `compute index` module and used by the engine using the following code
```
from geodium import compute_index 
compute_index("NDVI", {"N": nir_arr, "R": red_arr}, out_buf) 
```
# Method 4 Lazy equations
If a user requires an equation that is not in the premade functions or a spyndex eqaution they can use the built in lazy compiler for math equations to run the expression. The `Lazyband()` function imports the bands into the equation loader and the `.save()` function computes it to an output tif.

```
from geodium import LazyBand
N = LazyBand("./path to input tiff", 4)
R = LazyBand("./path to input tiff", 3)
OSAVI=(N - R) / (N + R + 0.16)
OSAVI.save("./path to output")
``` 
Lazy loading can also be done through the `LazyIndex()` which will lazy load a precompiled spyndex formula.
```
from geodium import LazyIndex, Lazyband
evi_node = LazyIndex("EVI", {
    "N": LazyBand("input.tiff", 4),
    "R": LazyBand("input.tiff", 3),
    "B": LazyBand("input.tiff", 1)
})
evi_node.save("output.tiff")
```