import geogram
import numpy as np


def test_power_diagram():
    np.random.seed(0)
    diagram = geogram.PowerDiagram3d(np.random.rand(10, 4))
    assert diagram.size() > 0
