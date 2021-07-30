import numpy as np
import scipy
import matplotlib.pyplot as plt
import searborn as sns


def spectrogram(x, s_f, cut_off=1, window=None, n_perseg=None, n_overlap=None, sigma=0.3, nfft=None):
    if window is None:
        window = scipy.signal.gaussian(n_perseg, sigma)

    f, t, s = scipy.signal.spectrogram(np.transpose(x), fs=s_f, window=window,
                                       nperseg=n_perseg,
                                       noverlap=n_overlap,
                                       nfft=nfft,
                                       detrend='constant', return_onesided=True,
                                       scaling='density', axis=-1, mode='psd')

    # Apply Threshold to the Spectrogram
    s[s < cut_off] = cut_off

    return f, t, s


# From Zeke
def normalize(u, axis=0):
    # normalize to (0-1) along axis
    # (axis=0 to normalize every col to its max value, axis=0 to normalize every row to its max value)
    u_max = np.repeat(np.amax(u, axis=axis, keepdims=True), u.shape[axis], axis=axis)
    # print(u_max.shape)
    u_min = np.repeat(np.amin(u, axis=axis, keepdims=True), u.shape[axis], axis=axis)

    u_range = u_max - u_min
    u_range[u_range == 0] = 1  # prevent nans, if range is zero set the value to 1.
    return (u - u_min) / u_range


# Adapted From Zeke
def my_spectrogram(Audio, s_f=30000, step_s=.001, window=None, n_perseg=192, f_cut=1, nfft=None, sigma=40):
    n_overlap = n_perseg - int(s_f * step_s)

    f, t, s = spectrogram(Audio, s_f=s_f, cut_off=f_cut, window=window, n_perseg=n_perseg, n_overlap=n_overlap,
                          sigma=sigma, nfft=nfft)

    s_normalized = normalize(np.log(s), axis=-1)
    return f, t, s_normalized


def plot_sonogram(s, t=False, f=False, cmap=None, **kwargs):
    if cmap is None:
        cmap = 'inferno'

    ax = sns.heatmap(s, xticklabels=t, yticklabels=f, cbar=False, cmap=cmap, **kwargs)

    ax.invert_yaxis()

    # for ind, label in enumerate(ax.get_xticklabels()):
    #     if ind % 100 == 0:
    #         label.set_visible(True)
    #     else:
    #         label.set_visible(False)
    # for ind, label in enumerate(ax.get_yticklabels()):
    #     if ind % 10 == 0:
    #         label.set_visible(True)
    #     else:
    #         label.set_visible(False)

    if ax is None:
        plt.show()


#
# def overlap(X, window_size, window_step):
#     """Create an overlapped version of X
#
#     Parameters
#     ----------
#     X : ndarray, shape=(n_samples,)
#         Input signal to window and overlap
#     window_size : int
#         Size of windows to take
#     window_step : int
#         Step size between windows
#
#     Returns
#     -------
#     X_strided : shape=(n_windows, window_size)
#         2D array of overlapped X
#     """
#     window_size, window_step = map(int, (window_size, window_step))
#     if window_size % 2 != 0:
#         raise ValueError("Window size must be even!")
#
#     # Make sure there are an even number of windows before stridetricks
#     append = np.zeros((window_size - len(X) % window_size))
#     X = np.hstack((X, append))
#
#     ws = window_size
#     ss = window_step
#     a = X
#
#     valid = len(a) - ws
#     nw = (valid) // ss
#     out = np.ndarray((nw, ws), dtype=a.dtype)
#
#     for i in range(nw):
#         # "slide" the window along the samples
#         start = i * ss
#         stop = start + ws
#         out[i] = a[start: stop]
#
#     return out
#
#
# def stft(X, fftsize=128, step=65, mean_normalize=True, real=False, compute_onesided=True):
#     """ Compute Short Time Fourier Transform for 1D real valued input X
#
#     Parameters
#     ----------
#     X : ndarray | shape=(n_samples,)
#         Input signal to window and overlap
#     fftsize : int
#         Size of windows to take
#     step : int
#         Step size between windows
#     mean_normalize : bool, optional
#         If True, the Mean is subtracted from the timeseries, defaults to True
#     real : bool, optional
#         defaults to False, determines whether the calculate the DFTs from a real signal or not
#
#     compute_onesided : bool, optional
#         If true it returns the one-sided spectrum
#
#     Returns
#     -------
#         X : complex ndarray
#         if real  is False the function will return:
#             The truncated or zero-padded input, transformed along the axis indicated by axis, or the last one if axis
#             is not specified.
#          if real is True the function will return:
#             The truncated or zero-padded input, transformed along the axis indicated by axis, or the last one if axis is
#             not specified. If n is even, the length of the transformed axis is (n/2)+1.
#             If n is odd, the length is (n+1)/2.
#     """
#     if real:
#         local_fft = np.fft.rfft
#         cut = -1
#     else:
#         local_fft = np.fft.fft
#         cut = None
#     if compute_onesided:
#         cut = fftsize // 2
#     if mean_normalize:
#         X -= X.mean()
#
#     X = overlap(X, fftsize, step)
#
#     size = fftsize
#     win = 0.54 - .46 * np.cos(2 * np.pi * np.arange(size) / (size - 1))
#     X = X * win[None]
#     X = local_fft(X)[:, :cut]
#     return X
#
#
# def pretty_spectrogram(d, log=True, thresh=5, fft_size=512, step_size=64):
#     """creates a spectrogram
#
#     log: take the log of the spectrgram
#     thresh: threshold minimum power for log spectrogram
#
#     Parameters
#     ----------
#     d : 1d-array
#         audio pressure waveform
#     log : bool
#         if True takes the log of the spectrogram
#     thresh : int
#         Threshold the minimum power for the spectrogram
#
#     fft_size :int
#         Size of windows to take
#
#     step_size : int
#         Step size between windows
#
#     Returns
#     -------
#     """
#     specgram = np.abs(stft(d, fftsize=fft_size, step=step_size, real=False, compute_onesided=True))
#
#     if log:
#         specgram /= specgram.max()  # volume normalize to max 1
#         specgram = np.log10(specgram)  # take log
#         specgram[specgram < -thresh] = -thresh  # set anything less than the threshold as the threshold
#     else:
#         specgram[specgram < thresh] = thresh  # set anything less than the threshold as the threshold
#
#     return specgram
#
#
# def plot_sonogram(Audio, Fs=30000, window=192, overlap=191, NFFT=384, ax=None):
#     """
#     Above Parameters were made by multiplying those Implemented in Anderson et. al 1996 by 1.5
#     Window Length = L; DFT Length = N; Time Sampling Interval = R [Oppenheim et. al]
#
#     window = Length of the Window ## Has to Be <= to NFFT, but > than Overlap
#     Fs = Sample Frequency (default 30 kHz)
#     NFFT = Number of Samples in Frequency Dimension of the Spectrogram [aka. DFT Length]
#     Difference between Window and Overlap is Sampling interval in Time Dimension
#     """
#
#     #     Audio = cp.deepcopy(Audio[:,0])
#     #     Audio = np.transpose(Audio)
#     dt = 1.0 / Fs  # the sample time
#
#     f, t, Sxx = scipy.signal.spectrogram(np.transpose(Audio), fs=Fs, nperseg=window, window='hamming',
#                                          noverlap=overlap, nfft=NFFT, scaling='spectrum',
#                                          mode='magnitude')
#
#     if not ax:
#         plt.figure(1, figsize=(15, 4))
#
#         plt.pcolormesh(t, f, 20 * np.log10(Sxx + sys.float_info.epsilon), cmap='viridis', vmin=1,
#                        vmax=20 * np.log10(Sxx.max()))
#         #         plt.pcolormesh(t, f, (Sxx + sys.float_info.epsilon), cmap='viridis', vmin = 0, vmax = (Sxx.max()))
#         plt.ylim([0, 10000])  # Changed Scale to (0-10) KHz #
#         plt.colorbar()
#         plt.ylabel('Frequency [Hz]')
#         plt.xlabel('Time [sec]')
#         plt.show()
#     if ax:
#         ax.pcolormesh(t, f, 20 * np.log10(Sxx + sys.float_info.epsilon), cmap='viridis',
#                       vmin=1, vmax=20 * np.log10(Sxx.max()))
#     #         print 'At least this works'
#
#
# #     return f, t, Sxx
#
# ##### Zeke's Code #######
#
# # From Zeke
# def normalize(u, axis=0):
#     # normalize to (0-1) along axis
#     # (axis=0 to normalize every col to its max value, axis=0 to normalize every row to its max value)
#     u_max = np.repeat(np.amax(u, axis=axis, keepdims=True), u.shape[axis], axis=axis)
#     # print(u_max.shape)
#     u_min = np.repeat(np.amin(u, axis=axis, keepdims=True), u.shape[axis], axis=axis)
#
#     u_range = u_max - u_min
#     u_range[u_range == 0] = 1  # prevent nans, if range is zero set the value to 1.
#     return (u - u_min) / u_range
#
#
# def spectrogram(x, s_f, n_perseg=None, n_overlap=None, cut_off=90, sigma=0.3, NFFT=None):
#     f, t, s = scipy.signal.spectrogram(np.transpose(x), fs=s_f, window=scipy.signal.gaussian(n_perseg, sigma),
#                                        nperseg=n_perseg,
#                                        noverlap=n_overlap,
#                                        nfft=NFFT,
#                                        detrend='constant', return_onesided=True,
#                                        scaling='density', axis=-1, mode='psd')
#
#     # Apply Threshold to the Spectrogram
#     lower_bound = np.max(s) / np.exp(cut_off / 10)
#     s[s < lower_bound] = lower_bound
#
#     return f, t, s
#
#
# def spectrogram(x, s_f, cut_off=1, window=None, n_perseg=None, n_overlap=None, sigma=0.3, nfft=None):
#     if window is None:
#         window = scipy.signal.gaussian(n_perseg, sigma)
#
#     f, t, s = scipy.signal.spectrogram(np.transpose(x), fs=s_f, window=window,
#                                        nperseg=n_perseg,
#                                        noverlap=n_overlap,
#                                        nfft=nfft,
#                                        detrend='constant', return_onesided=True,
#                                        scaling='density', axis=-1, mode='psd')
#
#     # Apply Threshold to the Spectrogram
#     s[s < cut_off] = cut_off
#
#     return f, t, s
#
#
# # Adapted From Zeke
# def good_spectrogram(Audio, s_f=30000, step_s=.001, window=None, n_perseg=192, f_cut=1, nfft=None, sigma=40):
#
#     n_overlap = n_perseg - int(s_f * step_s)
#
#     f, t, s = spectrogram(Audio, s_f=s_f, cut_off=f_cut, window=window, n_perseg=n_perseg, n_overlap=n_overlap,
#                           sigma=sigma, nfft=nfft)
#
#     s_normalized = normalize(np.log(s), axis=-1)
#     return f, t, s_normalized
#
#
# def plot_pretty_ersp(ersp, event_times, cmap=None, **kwargs):
#
#     if cmap is None:
#         cmap = 'RdBu_r'
#
#     ax = sns.heatmap(ersp, xticklabels=event_times, yticklabels=(fc_lo + fc_hi) / 2, cmap=cmap, **kwargs)
#
#     ax.invert_yaxis()
#     for ind, label in enumerate(ax.get_xticklabels()):
#         if ind % 100 == 0:
#             label.set_visible(True)
#         else:
#             label.set_visible(False)
#     for ind, label in enumerate(ax.get_yticklabels()):
#         if ind % 10 == 0:
#             label.set_visible(True)
#         else:
#             label.set_visible(False)
#     if ax is None:
#         plt.show()

