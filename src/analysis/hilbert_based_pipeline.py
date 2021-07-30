from BirdSongToolbox.preprocess import multi_bpf, hilbert_module, common_average_reference_array

import numpy as np
import scipy


def spectral_pertubation_module(neural_data, fs, l_freqs, h_freqs, bad_channels=None, verbose=False,
                                super_verbose=False):
    """ For One Chunk repeatedly Bandpass Filter and calculate the Chunk Averaged Power for each Frequency Band"""

    # 1. Common Average Reference
    car_data = common_average_reference_array(neural_data=neural_data, bad_channels=bad_channels)

    if verbose:
        print('CAR Done')

    # 2. Band Pass Filter the Data
    filtered_data = multi_bpf(chunk_neural_data=car_data, fs=fs, l_freqs=l_freqs, h_freqs=h_freqs,
                              verbose=super_verbose)
    filtered_data = np.asarray(filtered_data)  # Make a view as a ndarray

    if verbose:
        print('Filters Done')

    # 3. Take the Hilbert Transform (Seems like the Rate Limiting Step. Slow Computation)
    data_power = hilbert_module(filtered_data, output='amplitude')

    if verbose:
        print('Hilbert Done')

    # 4. Divide by the Mean of the entire time series
    norm_data = data_power / np.mean(data_power, axis=-1)[:, :, None]

    if verbose:
        print('Normalize Done')

    return norm_data


def spectral_perturbation_chunk(neural_chunks, fs, l_freqs, h_freqs, bad_channels=None, verbose=False, super_verbose=False):
    norm_chunks = []
    for index, chunk in enumerate(neural_chunks):
        if verbose:
            print(f"On Chunk # {index}")
        norm_data = spectral_pertubation_module(neural_data=chunk, fs=fs, l_freqs=l_freqs, h_freqs=h_freqs,
                                                bad_channels=bad_channels, verbose=verbose, super_verbose=super_verbose)
        norm_chunks.append(norm_data)

    return norm_chunks


def itc_module(neural_data, fs, l_freqs, h_freqs, bad_channels: list = None, verbose=False, super_verbose=False):
    """ For One Chunk repeatedly Bandpass Filter and calculate the Chunk Averaged Power for each Frequency Band"""

    # 1. Common Average Reference
    car_data = common_average_reference_array(neural_data=neural_data, bad_channels=bad_channels)

    if verbose:
        print('CAR Done')

    # 2. Band Pass Filter the Data
    filtered_data = multi_bpf(chunk_neural_data=car_data, fs=fs, l_freqs=l_freqs, h_freqs=h_freqs,
                              verbose=super_verbose)
    filtered_data = np.asarray(filtered_data)  # Make a view as a ndarray

    if verbose:
        print('Filters Done')

    # 3. Take the Hilbert Transform (Seems like the Rate Limiting Step. Slow Computation)
    data_phase = hilbert_module(filtered_data, output='phase')

    if verbose:
        print('Hilbert Done')

    return data_phase


def itc_phase_chunk(neural_chunks, fs, l_freqs, h_freqs, bad_channels: list = None, verbose=False, super_verbose=False):
    phase_chunks = []
    for index, chunk in enumerate(neural_chunks):
        if verbose:
            print(f"On Chunk # {index}")
        data_phase = itc_module(neural_data=chunk, fs=fs, l_freqs=l_freqs, h_freqs=h_freqs, verbose=verbose,
                                bad_channels=bad_channels, super_verbose=super_verbose)
        phase_chunks.append(data_phase)

    return phase_chunks


def runningMeanFast(x, n):
    """ Calculate the running mean of a 1darray

    note if mode = 'valid'
    the edge effects show in array, however will only occur during the buffer times.
    """
    return np.convolve(x, np.ones((n,)) / n, mode='valid')


def moving_average_nd(data, n=10):
    """ Calculate the running mean of a ndarray | shape (..., samples)

    note if mode = 'valid'
    the edge effects show in array, however will only occur during the buffer times.
    """
    mv_data = np.apply_along_axis(runningMeanFast, axis=-1, arr=data, n=n)

    return mv_data


# TODO: Fix the Feature Extraction Module. The Order of Operations is wrong for the hilbert
def feature_extraction_module(neural_data, fs, l_freqs, h_freqs, hilbert=None, car=True, norm=False,
                              bad_channels: list = None, mv_avg=None, z_score=None, drop_bad=False, verbose=False,
                              super_verbose=False):
    """ For One Chunk repeatedly Bandpass Filter and calculate the Chunk Averaged Power for each Frequency Band

    Parameters
    ----------
    neural_data :

    fs : int

    l_freqs :

    h_freqs :

    hilbert : str, optional
        String that instructs what information to extract from the analytical signal, options: 'phase', 'amplitude',
        defaults to None which mean no hilbert transform is done
    car : bool, optional
        defaults to True
    norm : bool, optional
        True,
    bad_channels : list, optional
        list of Channels To Exclude from Common Average Reference
    mv_avg : int, optional
        Defaults to None,
    drop_bad : bool, optional
        Defaults to False, if true it removes the bad channels from the returned array
    verbose : bool, optional
        Defaults to False,
    super_verbose : bool,
        Defaults to False

    Returns
    -------
    norm_data : ndarray | (freqs, channels, samples)

    """

    if car:
        # 1. Common Average Reference
        car_data = common_average_reference_array(neural_data=neural_data, bad_channels=bad_channels)

        if verbose:
            print('CAR Done')
    else:
        car_data = neural_data

    # 2. Band Pass Filter the Data
    filtered_data = multi_bpf(chunk_neural_data=car_data, fs=fs, l_freqs=l_freqs, h_freqs=h_freqs,
                              verbose=super_verbose)
    filtered_data = np.asarray(filtered_data)  # Make a view as a ndarray

    if verbose:
        print('Filters Done')

    if hilbert:
        # 3. Take the Hilbert Transform (Seems like the Rate Limiting Step. Slow Computation)
        if hilbert == "phase":
            data_feature = hilbert_module(filtered_data, output=hilbert, smooth=True)
            if verbose:
                print('The Instantaneous Phase Was Calculated')
        if hilbert == "amplitude":
            data_feature = hilbert_module(filtered_data, output=hilbert)
            if verbose:
                print('The Instantaneous Power was Calculated')
        if verbose:
            print('Hilbert Done')

    else:
        data_feature = filtered_data

    # if hilbert == "amplitude":
    #     if norm:
    #         # 4. Divide by the Mean of the entire time series
    #         norm_data = data_feature / np.mean(data_feature[:, :, 0:300000], axis=-1)[:, :, None]
    #
    #         if verbose:
    #             print('Normalize Done (Reminder 0:300000 used)')
    #
    # elif hilbert == "phase":
    #     norm_data = data_feature
    #
    #     if verbose:
    #         print('No Normalization')

    if norm:
        if hilbert == "phase":
            raise ValueError("It doesn't make sense to normalize Instantaneous phase")

        # 4. Divide by the Mean of the entire time series
        norm_data = data_feature / np.mean(data_feature[:, :, 0:300000], axis=-1)[:, :, None]

        if verbose:
            print('Normalize Done (Reminder 0:300000 used)')
    else:
        norm_data = data_feature

        if verbose:
            print('No Normalization')

    if z_score:
        if hilbert == "phase":
            raise ValueError("It doesn't make sense to Z-Score Instantaneous phase")


        norm_data = scipy.stats.zscore(data_feature, axis=-1)

        if verbose:
            print('Z-Score Done')
    else:
        if verbose:
            print("No Z-Score")

    if mv_avg:

        if hilbert == "phase":
            raise ValueError("It doesn't make sense to take a moving average of Instantaneous phase")

        norm_data = moving_average_nd(data=norm_data, n=mv_avg)

        if verbose:
            print(f'Moving_average of {mv_avg} samples')

    if drop_bad:
        norm_data = np.delete(norm_data, bad_channels, axis=1)

    return norm_data


def feature_extraction_chunk(neural_chunks, fs, l_freqs, h_freqs, hilbert=None, car=True, norm=False, mv_avg=None,
                             bad_channels: list = None, z_score=None, drop_bad=False, verbose=False, super_verbose=False):
    norm_chunks = []
    for index, chunk in enumerate(neural_chunks):
        if verbose:
            print(f"On Chunk # {index}")
        norm_data = feature_extraction_module(neural_data=chunk, fs=fs, l_freqs=l_freqs, h_freqs=h_freqs,
                                              hilbert=hilbert, mv_avg=mv_avg, car=car, norm=norm,
                                              bad_channels=bad_channels, z_score=z_score, drop_bad=drop_bad, verbose=verbose,
                                              super_verbose=super_verbose)
        norm_chunks.append(norm_data)

    return norm_chunks
