## Import modules
import numpy as np
from scipy.interpolate import interp1d
from phdtools.timeseries import fit_delay, multi_fit_delay

T = 2*np.pi
N_cycles = 10
ts = np.linspace(0, N_cycles*T, 1000)

# Reference function
f_ref = lambda t : np.sin(t)-np.cos(2*t) # interp1d(ts, obs(reference(flow)), kind = 'cubic')

# Used to subsample states
forced_delay = np.pi/3
f_subsample = lambda t : f_ref(t - forced_delay) # interp1d(ts, measured_data, kind = 'cubic')

optimal_delay = multi_fit_delay(f_ref, f_subsample, ts, T, N_samples=20, ts_per_sample=75, N_bounds=3)

print(optimal_delay/np.pi)