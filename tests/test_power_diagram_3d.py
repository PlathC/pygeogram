import grampy
import numpy as np


def test_power_diagram():
    np.random.seed(0)
    diagram = grampy.PowerDiagram3d(np.random.rand(10, 4))
    assert diagram.size() > 0
