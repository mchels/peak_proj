import numpy as np
import numpy.random as rnd
from math_funcs import sum_of_func, lorentzian_no_base

def construct_data_and_labels(n_samples, noise_amplitude=0.01, n_points=101):
    data = np.zeros((n_samples,n_points), dtype=float)
    labels = np.zeros((n_samples,1), dtype=int)
    x_vals = np.linspace(0.0, 1.0, n_points)
    for i in range(n_samples):
#         n_peaks_in = rnd.randint(1, high=3)# For 1vs2 peaks.
        n_peaks_in = rnd.randint(0, high=2)# For 0vs1 peak.
        n_peaks_out = rnd.randint(3)
        n_peaks = n_peaks_in + n_peaks_out
        params = rnd.rand(3*n_peaks)
        peak_poss_out = params[n_peaks_in*3::3]
        # Move 'out' peak positions to [-0.5;0] or [1.0;1.5]
        temp = peak_poss_out + 0.5*np.sign(peak_poss_out-0.5)
        params[n_peaks_in*3::3] = temp
        y_vals = sum_of_func(lorentzian_no_base, x_vals, *params)
        y_vals += noise_amplitude * rnd.normal(size=y_vals.shape)
        data[i] = y_vals
#         labels[i] = 1
#         labels[i,n_peaks_in-1] = 1# For 1vs2 peaks.
        if n_peaks_in == 1:
            labels[i] = 1# For 0vs1 peak.
    return data, labels

def make_cat_data_and_labels(n_samples, n_max_peaks, noise_amplitude=0.01, n_points=101):
    data = np.zeros((n_samples,n_points), dtype=float)
    labels = np.zeros((n_samples,1), dtype=int)
    x_vals = np.linspace(0.0, 1.0, n_points)
    for i in range(n_samples):
        n_peaks_in = rnd.randint(0, high=n_max_peaks+1)# For 1vs2 peaks.
        # n_peaks_in = rnd.randint(0, high=2)# For 0vs1 peak.
        n_peaks_out = rnd.randint(3)
        n_peaks = n_peaks_in + n_peaks_out
        params = rnd.rand(3*n_peaks)
        peak_poss_out = params[n_peaks_in*3::3]
        # Move 'out' peak positions to [-0.5;0] or [1.0;1.5]
        temp = peak_poss_out + 0.5*np.sign(peak_poss_out-0.5)
        params[n_peaks_in*3::3] = temp
        y_vals = sum_of_func(lorentzian_no_base, x_vals, *params)
        y_vals += noise_amplitude * rnd.normal(size=y_vals.shape)
        data[i] = y_vals
#         labels[i] = 1
#         labels[i,n_peaks_in-1] = 1# For 1vs2 peaks.
        labels[i] = n_peaks_in
    labels = to_categorical(labels)
    return data, labels
