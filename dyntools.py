import numpy as np
from scipy.integrate import odeint


def stabil(model, y0, tstabil, *args):
    """(Try to) stabilize a dynamical system
    """
    ys = odeint(model, y0, tstabil, *args)
    ylast = ys[-1, :]

    return ylast
