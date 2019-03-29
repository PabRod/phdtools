import numpy as np
from scipy.integrate import odeint


def decay(y, t=0, l=1):
    """Linear decay model
    """
    dydt = -l * y
    return dydt

def lotkavolterra(y, t, a=1, b=1, c=1, d=1):
    """ Lotka-Volterra predator prey model
    """
    prey, pred = y
    dydt = [a * prey - b * pred * prey,
            c * pred * prey - d * pred]
    return dydt
