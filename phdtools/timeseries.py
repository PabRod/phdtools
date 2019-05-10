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

def window_discrete(Dt, i, width):
    """ Sample timeseries along window

    Arguments:

    Dt: the time series
    i: the index
    width: window's width
    """
    delta = int(np.floor(width))
    if (i < delta-1) | (i > len(Dt)-1):
        # Return nan if out of boundaries
        return np.nan
    else:
        return Dt[i-delta+1:i+1]

def mean_window_discrete(Dt, width):
    """ Mean along window
    """
    N = len(Dt)
    s = np.empty(N)
    for i in range(0, N):
        s[i] = np.mean(window_discrete(Dt, i, width))

    return s

def std_window_discrete(Dt, width):
    """ Standard deviation along window
    """
    N = len(Dt)
    s = np.empty(N)
    for i in range(0, N):
        s[i] = np.std(window_discrete(Dt, i, width))

    return s

def var_window_discrete(Dt, width):
    """ Variance along window
    """
    N = len(Dt)
    s = np.empty(N)
    for i in range(0, N):
        s[i] = np.var(window_discrete(Dt, i, width))

    return s

def lissajous(Dt, period, ts):
    """ Generate lissajous figure for a given period
    """
    X = Dt
    Y = np.sin(2*np.pi*ts/period)

    return X, Y

def hideJumps(series):
    """ Hides the jumps from 2 pi to 0 in periodic boundary plots, such as the torus
    """
    jumps = np.abs(np.diff(series))
    mask = np.hstack([ jumps > jumps.mean()+3*jumps.std(), [False]])
    masked_series = np.ma.MaskedArray(series, mask)

    return masked_series

def torify(th1, th2, r_tube = 1, r_hole = 3):
    """ Plot two timeseries in a toroidal topology
        http://mathworld.wolfram.com/Torus.html """
    return [(r_hole + r_tube * np.cos(th2))*np.cos(th1),
            (r_hole + r_tube * np.cos(th2))*np.sin(th1),
            r_tube * np.sin(th2)]

def plot_lissajous(ax, Dt, period, ts, **kwargs):
    """ Plots the lissajous figure
    """
    X, Y = lissajous(Dt, period, ts)

    ax.plot(X, Y, **kwargs)

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

def plot_poincare(Dt, period, t0 = 0, ts = [], color = 'black', s = .1, alpha = 0.5, **kwargs):
    """ Plots the Poincaré map for the given period
    """

    t_max = np.max(ts)
    dims = Dt.shape[1]

    if dims == 1:
        xs = Dt[:, 0]
        x_sample = []
        for i in range(0, len(xs)):
            t_sample = t0 + i*period
            if (t_sample < t_max):
                x_sample.append(np.interp(t0 + i*period, ts, xs))
            else:
                break

        for i in range(0, len(x_sample)-1):
            plt.scatter(x_sample[i], x_sample[i+1], color = color, s = s, **kwargs)

        plt.xlabel('x_i')
        plt.ylabel('x_{i+1}')

    elif dims == 2:
        xs = Dt[:, 0]
        ys = Dt[:, 1]

        x_sample = []
        y_sample = []
        for i in range(0, len(xs)):
            t_sample = t0 + i*period
            if (t_sample < t_max):
                x_sample.append(np.interp(t0 + i*period, ts, xs))
                y_sample.append(np.interp(t0 + i*period, ts, ys))
            else:
                break

        plt.scatter(x_sample, y_sample, color = color, s = s, alpha = alpha, **kwargs)

    else:
        raise ValueError('Poincaré maps only supported for 1 or 2 dimensions')

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
