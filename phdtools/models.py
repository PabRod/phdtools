import numpy as np
from scipy.integrate import odeint


def decay(y, t=0, l=1):
    """Linear decay model
    """
    dydt = -l * y
    return dydt

def logistic(y, t=0, r=1, k=1):
    """Logistic growth model
    """
    dydt = r*y*(1 - y/k)
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

def duffing(state, t, g=0.2, w=1.2, a=-1, b=1, d=0.3):
    """ Duffing oscillator
    Ref: https://en.wikipedia.org/wiki/Duffing_equation
    """
    x, v = state
    dydt = [v,
            -a*x - b*x**3 - d*v + g*np.cos(w*t)]
    return dydt

def strogatz(state, t=0, w=(2,1), k=(2,1)):
    """ Strogatz coupled oscillators model
    """
    if callable(w) & callable(k):
        w = w(t)
        k = k(t)

    th1, th2 = state
    dydt = [w[0] + k[0]*np.sin(th2 - th1),
            w[1] + k[1]*np.sin(th1 - th2)]

    return dydt

def dphilrob(state, t=0, tmax='default', Qmax=100*3600, theta=10, sigma=3, vmaSa=1, vvm=1.9/3600, vmv=1.9/3600, vvc=6.3, vvh=0.19, Xi=10.8, mu=1e-3, taum=10/3600, tauv=10/3600, w=2*np.pi/24, alpha=0.0):

    # Define auxiliary functions

    ## Saturation function
    S = lambda V : Qmax/(1 + np.exp((theta-V)/sigma))

    ## External forcing
    if tmax == 'default':
        C = lambda t : 0.5*(1 + np.cos(w*(t - alpha)))
    else:
        C = lambda t : (1 - t / tmax)*0.5*(1 + np.cos(w*(t - alpha)))
        
    Vv, Vm, H = state
    dydt =[(             -vvm*S(Vm) + vvh*H - vvc*C(t)         - Vv)/tauv, # Ventro-lateral preoptic area activity
           (-vmv*S(Vv)                     + vmaSa             - Vm)/taum, # Mono-aminergic group activity
           (              mu*S(Vm)                             -  H)/Xi] # Homeostatic pressure

    return dydt

def lorenz(state, t=0, a=10, b=28, c=8/3):
    x, y, z = state
    dydt = [a*(y - x),
            x*(b - z) - y,
            x*y - c*z]

    return dydt

def competition(y, t=0, r=1, a=1):
    """ Basic competition model
    """
    ry = np.multiply(r, y)
    ay = np.dot(a, y)
    dydt = np.multiply(ry, (1-ay))

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

def reset_up(df, th_data, y0, ts, tinit):

    nsteps = len(ts)
    ys = np.zeros(nsteps)
    ys[0] = y0
    for i in range(1, nsteps):
        if ts[i] > tinit:
            if ys[i-1] < th_data[i-1]:
                ys[i] = ys[i-1] + df(ys[i-1], ts[i-1])*(ts[i] - ts[i-1])
            else:
                ys[i] = 0.0
        else:
            ys[i] = ys[i-1]

    return ys
