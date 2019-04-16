import numpy as np
from scipy.integrate import odeint


def decay(y, t=0, l=1):
    """Linear decay model
    """
    dydt = -l * y
    return dydt


def ass(y, t=0):
    """Basic model with two alternative stable states
    """
    dydt = -y**3 + y
    return dydt


def lotkavolterra(y, t=0, a=1, b=1, c=1, d=1):
    """Lotka-Volterra predator prey model
    """
    prey, pred = y
    dydt = [a * prey - b * pred * prey,
            c * pred * prey - d * pred]
    return dydt

def rosmac(y, t=0, r0=0.5, k=10, g0=0.4, h=2, l=0.15, e=0.6):
    """ Rosenzweig-MacArthur predator prey model
    """
    prey, cons = y

    def r(x):
        """ Growth rate """
        return r0*(1 - x/k)

    def g(x):
        """ Grazing rate """
        return g0/(x + h)

    dydt = [r(prey)*prey -g(prey)*prey*cons,
            -l*cons + e*g(prey)*prey*cons]
    return dydt


def hopf(state, t=0, a=1, b=1, l=-1):
    """Normal form for the Hopf bifurcation
    """
    from phdtools.dyntools import polarToCartesian

    x, y = state

    def hopf_pol(state, t=t, a=a, b=b, l=l):
        """Normal form for the Hopf bifurcation in polar coordinates
        """
        r, th = state
        drdt = [r * (l + a * r**2),
                1 + b * r**2]
        return drdt

    hopf_cart = polarToCartesian(hopf_pol)
    return hopf_cart(state)

def oscgen(state, t=0, amp=1, w=2*np.pi, g=1):
    """Oscillations generator
    """
    from phdtools.dyntools import polarToCartesian

    x, y = state

    def oscgen_pol(state, t=t, amp=amp, w=w, g=g):
        """Oscillations generator in polar coordinates
        """
        r, th = state
        drdt = [g * (amp - r),
                w]
        return drdt

    oscgen_cart = polarToCartesian(oscgen_pol)
    return oscgen_cart(state)
