import pytest
import numpy as np
from phdtools.dyntools import Trajectory


def test_constructor():

    from phdtools.models import decay
    y0 = [1]
    ts = np.linspace(0, 10, 100)
    traj = Trajectory(decay, y0, ts)

def test_solve():

    from phdtools.models import decay

    y0 = [1]
    ts = np.linspace(0, 20, 100)

    traj = Trajectory(decay, y0, ts)
    traj.solve()

    sol = traj.sol
    ylast = sol[-1,:]

    tol = 1e-2
    assert(np.abs(ylast[0]) < tol)

def test_plotTimeseries():

    from phdtools.models import lotkavolterra

    # Construct the flow
    y0 = [0.5, 0.6]
    ts = np.linspace(0, 10, 100)
    traj = Trajectory(lotkavolterra, y0, ts)

    # Plot a phase
    plt = traj.plotTimeseries()

def test_ploty0():

    from phdtools.models import lotkavolterra

    # Construct the flow
    y0 = [0.5, 0.6]
    ts = np.linspace(0, 10, 100)
    traj = Trajectory(lotkavolterra, y0, ts)

    # Plot a phase
    plt = traj.ploty0()

def test_plotTrajectory():

    from phdtools.models import lotkavolterra

    # Construct the flow
    y0 = [0.5, 0.6]
    ts = np.linspace(0, 10, 100)
    traj = Trajectory(lotkavolterra, y0, ts)
    roiX = np.linspace(0, 3, 10)
    roiY = np.linspace(0, 5, 20)

    # Plot a phase
    traj.plotPhase(roiX, roiY, color = "black", linewidth = 1, density = 1.5)
