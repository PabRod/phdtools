import pytest
import numpy as np
from phdtools.dyntools import Detflow


def test_constructor():

    from phdtools.models import decay
    flow = Detflow(decay)


def test_phase():

    from phdtools.models import lotkavolterra

    # Construct the flow
    flow = Detflow(lotkavolterra)
    roiX = np.linspace(0, 3, 10)
    roiY = np.linspace(0, 5, 20)

    # Plot a phase
    flow.plotPhase(roiX, roiY, color = "black", linewidth = 1, density = 1.5)
