# geogrampy - python binding for [geogram](https://github.com/BrunoLevy/geogram)

`geogrampy` aims at providing `geogram`'s state-of-the-art Voronoi diagram builder and related algorithms.

Pre-built binary wheels are provided on PyPI which can then be installed with

```sh
pip install grampy
```

## Example

A 2D Laguerre diagram can be computed with:

```py
import geogram as geo
import numpy as np

(vertices, triangles) = geo.shape.quad()

seeds = np.random.rand(10000, 2)
voronoi = geo.Voronoi(seeds, domain_vertices=vertices, domain_simplices=triangles)
```

Elements of the diagram are provided, regardless of the dimension through a decomposition of the diagram into simplices.
It should not to be confused with the dual of the diagram, the Delaunay triangulation. Instead, each Voronoi cell is
decomposed into a set of simplex:

- `q (#q, d)`: Voronoi vertices coordinates
- `t (#t, d)`: Simplices composing the Voronoi cells
- `tadj (#t, d)`: Simplex's neighbour index
- `tseed (#t)`: Simplex's corresponding seed index

See `examples` for dedicated notebooks.
