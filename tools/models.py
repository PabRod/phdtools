import numpy as np
from scipy.integrate import odeint


def decay(y, t=0, l=1):
    '''Linear decay model
    '''
    dydt = -l * y
    return dydt
