import pytest
import numpy as np
from phdtools.dyntools import Detflow


def test_constructor():

    from phdtools.models import decay
    flow = Detflow(decay, 1)

def test_findeq():

    from phdtools.models import lotkavolterra

    flow = Detflow(lotkavolterra, 2)
    flow.findeq((1.5, 1.3))
    flow.findeq((0.1, 0.1))

    eqs_set = flow.eqs

    assert((0, 0) in eqs_set)
    assert((1, 1) in eqs_set)

def test_claseq():

    from phdtools.models import lotkavolterra

    flow = Detflow(lotkavolterra, 2)
    eqs_types = (flow.claseq((0,0)), flow.claseq((1,1)))
    exp_types = ("unstable", "center")

    assert(eqs_types == exp_types)

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

    lyap_max = traj.maxlyapunov(y0)
    lyap_max_expected = 2

    tol = 1e-6
    assert(lyaps == pytest.approx(lyaps_expected, tol))
    assert(lyap_max == pytest.approx(lyap_max_expected, tol))

def test_phase():

    from phdtools.models import lotkavolterra

    # Construct the flow
    flow = Detflow(lotkavolterra, 2)
    roiX = np.linspace(0, 3, 10)
    roiY = np.linspace(0, 5, 20)

    # Plot a phase
    flow.plotPhase(roiX, roiY, color = "black", linewidth = 1, density = 1.5)
