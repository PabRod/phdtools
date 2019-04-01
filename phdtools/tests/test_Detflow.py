import pytest
import numpy as np
from phdtools.dyntools import Detflow


def test_constructor():

    from phdtools.models import decay
    flow = Detflow(decay, 1)


def test_jacobian():

    def fun(r, t = 0):

        x,y = r
        dydt = [1*x + 2*y,
                3*x + 4*y]
        return dydt

    traj = Detflow(fun, 2)
    y0 = [0, 0]

    J = traj.jac(y0)
    J_expected = np.array([[1, 2], [3, 4]])

    tol = 1e-6
    assert(J == pytest.approx(J_expected, tol))

def test_lyapunov():

    def fun(r, t = 0):

        x,y = r
        dydt = [1*x + 0*y,
                0*x + 2*y]
        return dydt

    traj = Detflow(fun, 2)
    y0 = [0, 0]

    lyaps = traj.lyapunovs(y0)
    lyaps_expected = np.array([1, 2])

    tol = 1e-6
    assert(lyaps == pytest.approx(lyaps_expected, tol))


def test_phase():

    from phdtools.models import lotkavolterra

    # Construct the flow
    flow = Detflow(lotkavolterra, 2)
    roiX = np.linspace(0, 3, 10)
    roiY = np.linspace(0, 5, 20)

    # Plot a phase
    flow.plotPhase(roiX, roiY, color = "black", linewidth = 1, density = 1.5)
