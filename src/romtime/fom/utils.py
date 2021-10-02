import numpy as np


def find_first_positive_peak(y, locs):

    peaks = y[locs]

    # -----------------------------------------------------------------
    # Remove zeros
    not_zero = ~np.isclose(peaks, 0.0, rtol=1e-3, atol=1e-3)
    positive = peaks > 0.0

    mask = not_zero & positive
    idx = np.where(mask)
    idx = idx[0][0]

    return locs[idx], locs[idx + 1]


def compute_time_between_peaks(ts, indices):

    t0 = ts[indices[0]]
    t1 = ts[indices[1]]

    return t1 - t0