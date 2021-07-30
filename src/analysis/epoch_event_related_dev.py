import numpy as np
from neurodsp import filt
from scipy import signal


def get_event_related_1d(data, fs, indices, window, subtract_mean=None, overlapping=None, **kwargs):
    """Take an input time series, vector of event indices, and window sizes,
        and return a 2d matrix of windowed trials around the event indices.

        Parameters
        ----------
        data : array-like 1d
            Voltage time series
        fs : int
            Sampling Frequency
        data : float
            Data sampling rate (Hz)
        indices : array-like 1d of integers
            Indices of event onset indices
        window : tuple (integers)
            Window (in ms) around event onsets
        subtract_mean : tuple (intengers), optional
            if present, subtract the mean value in the subtract_mean window for each
            trial from that trial's time series (this is a trial-by-trial baseline)
        overlapping : list
            Samples that overlap with another Epoch (overlapping_samples - epoch_abs_start)

        Returns
        -------
        event_related_matrix : array-like 2d
            Event-related times series around each index
            Each row is a separate event

        Example
        -------
        # >>>import BirdSongToolbox as tb
        # >>>import BirdSongToolbox.file_utility_functions as fuf
        #
        # >>>bird_id = 'z020'
        # >>>session = 'day-2016-06-03'
        # >>>data = tb.Import_PrePd_Data(bird_id, session)
        # # Get Hand Labels
        # >>>data_hl = bep.get_hand_labels(bird_id=bird_id, sess_name=session, local = False)
        # # Import the Absolute Times
        # >>>epoch_times = fuf._load_numpy_data(data_name='EpochTimes', bird_id=bird_id, session=session)
        # # Convert to List Style
        # >>>data_labels, data_onsets = bep.prep_handlabels_for_ml(Hand_labels= data_hl, Index= sorted(list(data_hl.keys())))
        # >>>overlapping_epochs, overlapping_samples = fuf._load_pckl_data(data_name='Overlap_Data', bird_id=bird_id,session=session)
        # >>>example_starts = data_onsets[0][10]
        # >>>test_array = np.transpose(np.array(data.Song_Neural), (0,2,1))
        # >>>example_epochs_times = epoch_times[46]
        # >>>example_overlapping_samples = overlapping_samples[0]
        # >>>times_test, events_test = get_event_related_1d(data=test_array[0,0,:], fs=1000, indices= example_starts,
        # >>>                                                  window = (-500, 500), subtract_mean=None,
        # >>>                                      overlapping = example_overlapping_samples - example_epochs_times[0])


        """

    def windows_to_indices(fs, window_times):
        """convert times (in ms) to indices of points along the array"""
        conversion_factor = (1 / fs) * 1000  # convert from time points to ms
        window_times = np.floor(np.asarray(window_times) / conversion_factor)  # convert
        window_times = window_times.astype(int)  # turn to ints

        return window_times

    def convert_index(fs, indexes):
        """convert the start times to their relative sample based on the fs parameter"""
        conversion_factor = (1 / fs) * 30000  # Convert from 30Khs to the set sampling rate
        indexes = np.rint(np.array(indexes) / conversion_factor)
        indexes = indexes.astype(int)
        return indexes

    # Remove overlapping labels
    if overlapping is not None:
        overlaps = [index for index, value in enumerate(indices) if value in overlapping]  # Find overlapping events
        indices = np.delete(indices, overlaps, axis=0)  # Remove them from the inds

    window_idx = windows_to_indices(fs=fs, window_times=window)  # convert times (in ms) to indices
    inds = convert_index(fs=fs, indexes=indices) + np.arange(window_idx[0], window_idx[1])[:,
                                                   None]  # build matrix of indices

    # Remove Edge Instances from the inds
    bad_label = []
    bad_label.extend([index for index, value in enumerate(inds[0, :]) if value < 0])  # inds that Start before Epoch
    bad_label.extend([index for index, value in enumerate(inds[-1, :]) if value >= len(data)])  # inds End after Epoch
    inds = np.delete(inds, bad_label, axis=1)  # Remove Edge Instances from the inds

    event_times = np.arange(window[0], window[1], fs / 1000)
    event_related_matrix = data[inds]  # grab the data
    event_related_matrix = np.squeeze(event_related_matrix).T  # make sure it's in the right format

    # baseline, if requested
    if subtract_mean is not None:
        basewin = [0, 0]
        basewin[0] = np.argmin(np.abs(event_times - subtract_mean[0]))
        basewin[1] = np.argmin(np.abs(event_times - subtract_mean[1]))
        event_related_matrix = event_related_matrix - event_related_matrix[:, basewin[0]:basewin[1]].mean(axis=1,
                                                                                                          keepdims=True)

    return event_times, event_related_matrix


def get_event_related_spectral_component_1d(data, fs, indices, window, comp_type: str = 'power', **kwargs):
    """ Creates Overlapping Narrow Bandpass Filters and Aligns Time Sequences of A Repeated Behavior of Interest

        Parameters
        ----------
        data : array-like 1d
            Voltage time series for 1 Channel
        fs : float
            Data sampling rate (Hz)
        indices : array-like 1d of integers
            Indices of event onset indices
        window : tuple (integers)
            Window (in ms) around event onsets
        comp_type : str
            signal component to return, defaults to 'power',  options ('power', 'phase')
        subtract_mean : tuple (intengers), optional
            if present, subtract the mean value in the subtract_mean window for each
            trial from that trial's time series (this is a trial-by-trial baseline)
        overlapping : list, optional
            Samples that overlap with another Epoch (overlapping_samples - epoch_abs_start)

        Returns
        -------
        erc : array-like 3d
            The Event-Related Components, An arrary of overlapping filters corresponding to the single repeated
            behavior over the course of one epoch
            (Filters, Events, Time)
        """
    assert comp_type == 'power' or comp_type == 'phase', "comp_type can only be 'power' or 'phase' "

    fc_lo = np.arange(3, 249, 2)
    fc_hi = np.arange(5, 251, 2)

    # Filter the data
    erc = []
    i_freqs = np.size(fc_hi)
    for i in range(i_freqs):
        filt_dat = filt.filter_signal(data, fs, 'bandpass', (fc_lo[i], fc_hi[i]), remove_edges=False)
        filt_dat = signal.hilbert(filt_dat)
        if comp_type == 'power':
            filt_dat = np.abs(filt_dat)  # get amplitude
        else:
            filt_dat = np.angle(filt_dat, deg=False)  # get phase
        event_times, event_related_matrix = get_event_related_1d(filt_dat, fs, indices, window, **kwargs)  # Get Events
        erc.append(event_related_matrix)
    print(np.shape(erc))
    print(np.shape(erc[0]))
    erc = np.asarray(erc)
    print(np.shape(erc))
    return erc


def _make_overlap_dict(overlapping_epochs, overlapping_samples, epoch_index):
    """ Make a dictionary of overlapping sanples that can easily be used to look up samples to exclude

    Parameters
    ----------
    overlapping_epochs :
    overlapping_samples :
    epoch_index :

    Returns
    -------
    overlap_dict : dict
        Dictionary whos keys correspond to Epochs that overlap with others, and values are 1d array of samples
        that overlap
        {Epoch: (Samples,)}
    """
    index_list = list(epoch_index)
    overlap_dict = {}
    for epoch, samples in zip(overlapping_epochs[:,1], overlapping_samples):
        overlap_dict[index_list.index(epoch)] = samples
    return overlap_dict


def get_ersc_for_1_channel(single_chan_data, fs, label_index, window, overlap_dict: dict, comp_type: str = 'power',
                           **kwargs):
    """Get the Phase Spectral Components for all instances of 1 label for 1 Channel

    Parameters:
    -----------
    single_chan_data : np.ndarray
            Voltage time series for 1 Channel for All Epochs
            shape:(Epochs, Samples)
    fs : float
        Data sampling rate (Hz)
    label_index : list
        List of array-like 1d of integers representing all start frames of every instances of the label of focus
        [Num_Trials]->[Num_Exs]
    window : tuple (integers)
        Window (in ms) around event onsets
    overlap_dict : dict
        Dictionary whos keys correspond to Epochs that overlap with others, and values are 1d array of samples
        that overlap
        {Epoch: (Samples,)}
    comp_type : str
        signal component to return, defaults to 'power',  options ('power', 'phase')
    subtract_mean : tuple (integers), optional
        if present, subtract the mean value in the subtract_mean window for each
        trial from that trial's time series (this is a trial-by-trial baseline)
    overlapping : list, optional
        Samples that overlap with another Epoch (overlapping_samples - epoch_abs_start)

    Returns
    -------
    all_ersc :
        All Event Related Spectral Components for all instances of the designated label for 1 Channel
        shape:(Instances, Filters, Samples)

    """
    all_ersc = []
    for index, (epoch, indices) in enumerate(zip(single_chan_data, label_index)):
        if index in overlap_dict.keys():
            overlap = overlap_dict[index]
        else:
            overlap = None
        if len(epoch) > 0:
            ersc, times = get_event_related_spectral_component_1d(data=epoch, fs=fs, indices=indices,
                                                                  window=window, comp_type=comp_type,
                                                                  overlapping=overlap, **kwargs)
            if ersc.ndim > 2:
                for i in range(ersc.shape[1]):
                    all_ersc.append(ersc[:, i, :])
            else:
                all_ersc.append(ersc)
        else:
            pass

        return all_ersc


