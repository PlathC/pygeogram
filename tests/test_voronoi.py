import geogram as geo
import numpy as np


def test_voronoi_2d():
    (vertices, triangles) = geo.shape.quad()

    seed_nb = 10
    seeds = np.random.rand(seed_nb, 2)
    voronoi = geo.Voronoi(seeds, domain_vertices=vertices, domain_simplices=triangles)

    assert voronoi.q.shape[0] > 0


"""
def test_laguerre_2d():
    (vertices, triangles) = geo.shape.quad()

    seed_nb = 10
    seeds = np.random.rand(seed_nb, 2)
    weights = np.array([1. for _ in range(seed_nb)])

    laguerre = geo.Voronoi(seeds, weights, domain_vertices=vertices, domain_simplices=triangles)
    assert laguerre.q.shape[0] > 0
    assert laguerre.t.shape[0] > 0
    assert laguerre.tadj.shape[0] > 0
    assert laguerre.tseed.shape[0] > 0

    weights[2] = 1.1
    laguerre = geo.Voronoi(seeds, weights, domain_vertices=vertices, domain_simplices=triangles)
    assert laguerre.q.shape[0] > 0
    assert laguerre.t.shape[0] > 0
    assert laguerre.tadj.shape[0] > 0
    assert laguerre.tseed.shape[0] > 0
"""


def test_voronoi_3d():
    (vertices, triangles) = geo.shape.cube()
    (vertices, tets) = geo.mesh.tetrahedralize(vertices, triangles)

    seed_nb = 10
    seeds = np.random.rand(seed_nb, 3)
    voronoi = geo.Voronoi(seeds, domain_vertices=vertices, domain_simplices=tets)

    assert voronoi.q.shape[0] > 0


def test_laguerre_3d():
    (vertices, triangles) = geo.shape.cube()
    (vertices, tets) = geo.mesh.tetrahedralize(vertices, triangles)

    seed_nb = 10
    seeds = np.random.rand(seed_nb, 3)
    weights = np.array([1. for _ in range(seed_nb)])

    laguerre = geo.Voronoi(seeds, weights, domain_vertices=vertices, domain_simplices=tets)
    assert laguerre.q.shape[0] > 0
    assert laguerre.t.shape[0] > 0
    # assert laguerre.tadj.shape[0] > 0
    assert laguerre.tseed.shape[0] > 0

    weights[2] = 1.1
    laguerre = geo.Voronoi(seeds, weights, domain_vertices=vertices, domain_simplices=tets)
    assert laguerre.q.shape[0] > 0
    assert laguerre.t.shape[0] > 0
    # assert laguerre.tadj.shape[0] > 0
    assert laguerre.tseed.shape[0] > 0
