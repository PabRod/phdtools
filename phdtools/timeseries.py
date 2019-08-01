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

def ac_window_discrete(Dt, width, lag = 1):
    """ Autocorrelation along window
    """
    N = len(Dt)
    ac = np.empty(N)
    for i in range(0, N):
        x = window_discrete(Dt, i, width)
        if np.isnan(x).any():
            ac[i] = np.nan
        else:
            ac[i] = autocorrelation(x, lag)

    return ac

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

def fit_delay(fun, ts, ys, T=2*np.pi, bounds = (-3.14, 3.14), method = 'bounded', debug = False, info = '', **kwargs):
    """ Fit a set of points to a given function just by displacing it in the horizontal axis
    """
    def D(ys, ts, delay):
        """ Sum of all square distances
        """

        def d(y, t):
            """ Square distance of a single point
            """
            return (y - fun(t - delay))**2

        distances = list(map(d, ys, ts))
        return np.sum(distances)

    from scipy.optimize import minimize_scalar, minimize
    res = minimize_scalar(lambda delay : D(ys, ts, delay), bounds=bounds, method=method, **kwargs)

    if debug:
        optimal_delay = res.x
        D2 = res.fun

        ## Plotting
        delays = np.linspace(np.min(bounds), np.max(bounds), 250)
        Ds = list(map(lambda delay: D(ys, ts, delay), delays))

        t_plot = np.linspace(np.min(ts)-T, np.max(ts)+T, 1000)
        fig, axs = plt.subplots(2, 1)
        plt.suptitle(info)
        axs[0].plot(t_plot, fun(t_plot), label = 'Fitting function')
        axs[0].plot(ts, ys, color = 'r', marker = '.', label = 'Original points', alpha = 0.2)
        axs[0].plot(ts - optimal_delay, ys, color = 'g', marker = '.', label = 'Optimized points', alpha = 0.2)
        axs[0].set_xlabel('t')
        axs[0].set_ylabel('f(t)')
        axs[0].legend()

        axs[1].set_title('Target function')
        axs[1].plot(delays, Ds, color = 'k')
        axs[1].scatter(optimal_delay, D(ys, ts, optimal_delay), color = 'k')
        # axs[1].set_xlim(axs[0].get_xlim())
        axs[1].set_xlabel('Delay')
        axs[1].set_ylabel('Square distance')

        plt.show()

    # res.x contains the position of the minima
    # res.fun contains the value of the minima (f(x))
    return res

def multi_fit_delay(y_ref, y_measured, ts, T, N_samples=20, ts_per_sample=75, N_bounds=1, debug=False):
    """ Robustly applies the fit_delay function to a subset of points

    parameters:
    y_ref: reference values
    y_measured: displaced values
    ts: sampling times
    T: estimated period

    (optional)
    N_samples: number of partitions of ts
    ts_per_sample: length of each time partition
    N_bounds: number of sub-bounds to look for minima (increase to filter out non-absolute minima)
    debug: True for debug mode
    """
    ## Input interpretation
    if callable(y_ref): # The input is already a function, no need to interpolate
        f_ref = y_ref
    else: # Turn input into callable function by interpolation
        from scipy.interpolate import interp1d
        f_ref = periodify(interp1d(ts, y_ref, kind = 'cubic'), T)

    if callable(y_measured): # The input is already a function, no need to interpolate
        f_measured = y_measured
    else: # Turn input into callable function by interpolation
        from scipy.interpolate import interp1d
        f_measured = interp1d(ts, y_measured, kind = 'cubic')

    ts_samples = np.linspace(ts[0]+T, ts[-1]-T, N_samples+1) # Exclude borders

    ## Create subpartitions of the bounds where the minima is going to be searched
    #
    # For example: (0, 3) partitioned with N_bounds = 3 yields (0, 1), (1, 2), (2, 3)
    bounds = (0, T)
    partitioned_bounds = np.zeros((N_bounds, 2))
    aux_vals = np.linspace(bounds[0], bounds[1], N_bounds + 1)
    for i in range(0, N_bounds):
        partitioned_bounds[i, 0] = aux_vals[i]
        partitioned_bounds[i, 1] = aux_vals[i+1]

    ## Sweep in a time window
    optimal_delay = np.nan*np.empty(N_samples)
    for i in range(0, N_samples):
        ts_subsample = np.linspace(ts_samples[i], ts_samples[i+1], ts_per_sample)
        ys_subsample = f_measured(ts_subsample)

        ## The optimization method looks for local minima inside given bounds
        delay_candidates = np.zeros(N_bounds)
        D2s = np.zeros(N_bounds)
        for j in range(0, N_bounds): # Use several bounds' partitions if required
            res = fit_delay(f_ref, ts_subsample, ys_subsample, T=T, bounds=partitioned_bounds[j,:], debug=debug)
            delay_candidates[j] = res.x
            D2s[j] = res.fun

        ## Identify the absolute minimum...
        absolute_min_index = np.argmin(D2s)

        ## ... and choose only that one
        optimal_delay[i] = delay_candidates[absolute_min_index]

    return optimal_delay

def periodify(f, period = 2*np.pi):
    """ Forces a piecewise periodic function
    """

    def f_p(t):
        return f(np.mod(t, period))

    return f_p
