## Import modules
import numpy as np
from scipy.interpolate import interp1d
from phdtools.timeseries import fit_delay, multi_fit_delay

T = 2*np.pi
N_cycles = 10
ts = np.linspace(0, N_cycles*T, 1000)

# Reference function
f_ref = lambda t : np.sin(t)-np.cos(2*t) # interp1d(ts, obs(reference(flow)), kind = 'cubic')

# Measured data
## Generated
mode = 'data'
forced_delay = np.pi/3
if mode == 'function':
    f_measured = lambda t : f_ref(t - forced_delay)
    optimal_delay = multi_fit_delay(f_ref, f_measured, ts, T, N_samples=20, ts_per_sample=75, N_bounds=3)
elif mode == 'data':
    data_gen = lambda t : f_ref(t - forced_delay) # Use whatever method to generate the data

    ref_data = f_ref(ts)
    measured_data = data_gen(ts)
    optimal_delay = multi_fit_delay(ref_data, measured_data, ts, T, N_samples=20, ts_per_sample=75, N_bounds=3)
else:
    raise Exception('Supported modes are: data and function')

print(f'mode: {mode}')
print(optimal_delay/np.pi)
