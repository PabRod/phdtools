import pytest
import numpy as np
from phdtools.dyntools import stoTrajectory


def test_constructor():

    from phdtools.models import decay

    def G(y, t=0):
        return y**2

    y0 = 1
    ts = np.linspace(0, 10, 100)
    stot = stoTrajectory(decay, G, y0, ts)

@pytest.mark.parametrize("method", [
    ('EuMa'),
    ('Ito')
])
def test_solve_1D(method):

    a = 1.0
    b = 0.2
    ts = np.linspace(0.0, 5.0, 5001)
    y0 = 0.1

    def f(x, t=0):
        return -(a + x*b**2)*(1 - x**2)

    def g(x, t=0):
        return b*(1 - x**2)

    stot = stoTrajectory(f, g, y0, ts, method)
    stot.solve()

    tol = 5e-1
    assert(stot.sol[-1] == pytest.approx(-1, tol))

@pytest.mark.parametrize("method", [
    ('EuMa'),
    ('Ito')
])
def test_solve_2D(method):

    A = np.array([[-0.5, -2.0],
              [ 2.0, -1.0]])

    B = np.diag([0.0, 0.0]) # diagonal, so independent driving Wiener processes

    ts = np.linspace(0.0, 10.0, 10001)
    y0 = np.array([3.0, 3.0])

    def f(x, t=0):
        return A.dot(x)

    def g(x, t=0):
        return B

    stot = stoTrajectory(f, g, y0, ts, method)
    stot.solve()

    tol = 1e-1
    assert(np.max(np.abs(stot.sol[-1,:])) < tol)
