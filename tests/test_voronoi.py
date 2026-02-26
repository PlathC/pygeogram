import geogram as geo
import numpy as np


def test_voronoi_2d():
    (vertices, triangles) = geo.domain.quad()

    seeds = np.random.rand(10, 2)
    voronoi = geo.Voronoi(seeds, domain_vertices=vertices, domain_simplices=triangles)
    assert voronoi.q.shape[0] > 0
    assert (voronoi.q != voronoi.seeds).all()
