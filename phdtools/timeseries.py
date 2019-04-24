import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def measure(xt, **kwargs):
    """ Introduce measure noise
    """
    Dt = xt + np.random.normal(size=len(xt), **kwargs)
    return Dt

def fluctuations(Dt):
    """ Fluctuations around the mean
    """
    Vt = Dt - np.mean(Dt)
    return Vt

def symm_subset(Vt, k):
    """ Trims symmetrically the beginning and the end of the vector
    """
    if k == 0:
        Vt_k = Vt
    else:
        Vt_k = Vt[k:]
        Vt = Vt[0:-k]

    return Vt, Vt_k

def find_index(t, ts = []):
    """ Links times with indices """
    if len(ts) == 0:
        i = t
    else:
        i = np.min(np.where((t <= ts)))

    return i

def __autocorrelation_discrete(Dt, k=1):
    """ Correlation coefficient (discrete version)
    """
    if k == 0:
        rho = 1
    else:
        Vt = fluctuations(Dt)
        Vt, Vt_k = symm_subset(Vt, k)

        rho = np.sum(Vt_k * Vt) / np.sum(Vt**2)

    return rho

def autocorrelation(Dt, l, ts=[]):
    """ Correlation coefficient

    Parameters:

    Dt (array): the timeseries data
    l (double): the lag
    ts (array): the reference times. If not provided, t is understood as index
    """
    if len(ts) == 0: # l understood as index
        return __autocorrelation_discrete(Dt, l)
    else: # l understood as time
        ac = np.empty(len(ts))
        for i in range(0, len(ts)):
            ac[i] = __autocorrelation_discrete(Dt, i)

        return np.interp(l, ts, ac)

def plot_return(ax, Dt, k = 1, marker=".", **kwargs):
    """ Plots signal vs delayed signal
    """
    Vt = fluctuations(Dt)
    Vt, Vt_k = symm_subset(Vt, k)

    ax.set_title("Return map")
    ax.scatter(Vt, Vt_k, marker=marker, **kwargs)
    ax.set_xlabel('V_t')
    ax.set_ylabel('V_{t+k}')

    return ax

def plot_autocorrelation(ax, Dt, ls, ts=[], marker=".", **kwargs):
    """ Plot several values of the autocorrelation
    """
    ax.set_title('Autocorrelation vs. lag')
    for l in ls:
        ax.plot(l, autocorrelation(Dt, l, ts), marker=marker, **kwargs)
        ax.set_xlabel('k')

    return ax

def plot_approx_phas(ax, Dt, ts, marker='.', **kwargs):
    """ Plots the reconstructed phase plane
    """
    DDt = np.gradient(Dt, ts)

    ax.set_title('Reconstructed phase plane')
    ax.scatter(Dt, DDt, marker=marker, **kwargs)
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    return ax
