## Import modules
import numpy as np
from scipy.interpolate import interp1d
from phdtools.timeseries import fit_delay

T = 2*np.pi
N_cycles = 15
ts = np.linspace(0, N_cycles*T, 1000)

# Reference function
f_ref = lambda t : np.sin(t)-np.cos(2*t) # interp1d(ts, obs(reference(flow)), kind = 'cubic')

# Used to subsample states
forced_delay = np.pi/3
f_subsample = lambda t : f_ref(t - forced_delay) # interp1d(ts, measured_data, kind = 'cubic')

N_samples = 20
ts_per_sample = 75
ts_samples = np.linspace(T, ts[-1]-T, N_samples) # Exclude borders

## Create subpartitions of the bounds where the minima is going to be looked for
bounds = (0, T)
N_partitions = 2
partitioned_bounds = np.zeros((N_partitions, 2))
aux_vals = np.linspace(bounds[0], bounds[1], N_partitions + 1)
for i in range(0, N_partitions):
    partitioned_bounds[i, 0] = aux_vals[i]
    partitioned_bounds[i, 1] = aux_vals[i+1]

optimal_delay = np.nan*np.empty(N_samples-1)
for i in range(0, N_samples-1):
    ts_subsample = np.linspace(ts_samples[i], ts_samples[i+1], ts_per_sample)
    ys_subsample = f_subsample(ts_subsample)

    ## The optimization method looks for local minima inside given bounds
    delay_candidates = np.zeros(N_partitions)
    D2s = np.zeros(N_partitions)
    for j in range(0, N_partitions): # Use several bounds' partitions if required
        res = fit_delay(f_ref, ts_subsample, ys_subsample, bounds = partitioned_bounds[j,:], debug = True)
        delay_candidates[j] = res.x
        D2s[j] = res.fun

    ## Identify the absolute minimum
    absolute_min_index = int(np.where(D2s == np.min(D2s))[0])

    ## And choose only that one
    optimal_delay[i] = delay_candidates[absolute_min_index]

print(optimal_delay/np.pi)
