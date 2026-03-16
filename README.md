# python binding for [geogram](https://github.com/BrunoLevy/geogram)

This repos provides python bindings for [`geogram`](https://github.com/BrunoLevy/geogram)'s state-of-the-art Voronoi diagram builder and related algorithms.

Pre-built binary wheels are provided on PyPI which can then be installed with

```sh
pip install geogram
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

Plotting the simplices edges `t` between the voronoi vertices `q` leads to
<p align="center">
  <img width="300" height="300" alt="image" src="https://github.com/user-attachments/assets/cc7e360f-d2c8-491e-a92b-b405cdd4a5a5" />
  <img width="300" height="300" alt="image" src="https://github.com/user-attachments/assets/0c102038-f24c-4c2a-892a-fbdf22ab3f90" />
</p>


See `examples` for additional notebooks.
