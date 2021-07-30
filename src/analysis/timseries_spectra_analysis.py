import numpy as np
from scipy import signal
from neurodsp import filt
import matplotlib.pyplot as plt
import seaborn as sns

# This Function can only grab times within the same Epoch


def event_related(data, fs, indices, window, subtract_mean=None):
    """Take an input time series, vector of event indices, and window sizes,
    and return a 2d matrix of windowed trials around the event indices.

    Parameters
    ----------
    data : array-like 1d
        Voltage time series
    fs : float
        Data sampling rate (Hz)
    indices : array-like 1d of integers
        Indices of event onset indices
    window : tuple (integers)
        Window (in ms) around event onsets
    subtract_mean : tuple (intengers), optional
        if present, subtract the mean value in the subtract_mean window for each
        trial from that trial's time series (this is a trial-by-trial baseline)

    Returns
    -------
    event_times : array-like 1d
        Time index of the events relative to their onset
    event_related_matrix : array-like 2d
        Event-related times series around each index
        Each row is a separate event
        Shape = (Event, Time)
    """

    # convert times (in ms) to indices of points along the array
    def windows_to_indices(fs, window_times):
        conversion_factor = (1 / fs) * 1000  # convert from time points to ms
        window_times = np.floor(np.asarray(window_times) / conversion_factor)  # convert from ms to samples
        window_times = window_times.astype(int)  # turn to ints

        return window_times

    window_idx = windows_to_indices(fs, window)  # convert times (in ms) to indices
    inds = indices + np.arange(window_idx[0], window_idx[1])[:, None]  # build matrix of indices
    event_times = np.arange(window[0], window[1], fs / 1000)

    event_related_matrix = data[inds]  # grab the data
    event_related_matrix = np.squeeze(event_related_matrix).T  # make sure it's in the right format (Events, Time)

    # baseline, if requested
    if subtract_mean is not None:
        basewin = [0, 0]
        basewin[0] = np.argmin(np.abs(event_times - subtract_mean[0]))
        basewin[1] = np.argmin(np.abs(event_times - subtract_mean[1]))
        event_related_matrix = event_related_matrix - event_related_matrix[:, basewin[0]:basewin[1]].mean(axis=1,
                                                                                                          keepdims=True)

    return event_times, event_related_matrix


def pretty_spectral_analysis(data, fs, window=None, zscore=False):
    """ Creates Overlapping Narrow Bandpass Filters and Aligns Time Sequences of A Repeated Behavior of Interest

    Parameters
    ----------
    data : array-like 1d
        Voltage time series for 1 Channel
    fs : float
        Data sampling rate (Hz)
    window : tuple (integers)
        Window (in ms) around event onsets
    zscore : bool (Optional)
        If True function returns the z_score result for each frequency band

    Returns
    -------
    ersp : array-like 2d
        A psuedo Spectrogram with High Temporal Resolution, An arrary of overlapping filters corresponding to votage of one channel
        (Filters, Time)
    """

    # TODO: Make it such that you can control the filters if you want to
    fc_lo = np.arange(3, 249, 2)
    fc_hi = np.arange(5, 251, 2)

    # Filter the data
    ersp = []
    i_freqs = np.size(fc_hi)
    for i in range(i_freqs):
        filt_dat = filt.filter_signal(data, fs, 'bandpass', (fc_lo[i], fc_hi[i]), remove_edges=False)
        filt_dat = np.abs(signal.hilbert(filt_dat))
        if zscore:
            filt_dat = scipy.stats.zscore(filt_dat)

        ersp.append(filt_dat)
    ersp = np.asarray(ersp)
    #      if zscore:
    #             ersp = scipy.stats.zscore(ersp, axis = 1)
    return ersp


def event_related_spectral_perturbation(data, fs, indices, window, subtract_mean=None):
    """ Creates Overlapping Narrow Bandpass Filters and Aligns Time Sequences of A Repeated Behavior of Interest

    Parameters
    ----------
    data : array-like 1d
        Voltage time series
    fs : float
        Data sampling rate (Hz)
    indices : array-like 1d of integers
        Indices of event onset indices
    window : tuple (integers)
        Window (in ms) around event onsets
    subtract_mean : tuple (intengers), optional
        if present, subtract the mean value in the subtract_mean window for each
        trial from that trial's time series (this is a trial-by-trial baseline)

    Returns
    -------
    ersp : array-like 3d
        The Event-Related Spectral Perturbation, An arrary of overlapping filters corresponding to the single repeated
        behavior over the course of one epoch
        (Filters, Events, Time)
    """

    # TODO: Make it such that you can control the filters if you want to
    fc_lo = np.arange(3, 249, 2)
    fc_hi = np.arange(5, 251, 2)

    # Filter the data
    ersp = []
    i_freqs = np.size(fc_hi)
    for i in range(i_freqs):
        filt_dat = filt.filter_signal(data, fs, 'bandpass', (fc_lo[i], fc_hi[i]), remove_edges=False)
        filt_dat = np.abs(signal.hilbert(filt_dat))
        event_times, foo = event_related(filt_dat, fs, indices, window, subtract_mean)
        ersp.append(foo)
    # print(np.shape(ersp))
    # print(np.shape(ersp[0]))
    ersp = np.asarray(ersp)
    # print(np.shape(ersp))
    return ersp, event_times

