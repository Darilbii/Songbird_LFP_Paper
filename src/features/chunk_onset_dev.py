import BirdSongToolbox.free_epoch_tools as fet
import BirdSongToolbox.chunk_analysis_tools as cat

import numpy as np


def clip_instances_for_templates(freq_preprocessed_data, all_offsets, all_bin_widths, label_events):
    """ Clip the Data into the trials that can be used to make the Pearson Templates

    Parameters
    ----------
    freq_preprocessed_data : list | [Freqs]->[Chunks]->(1, channels, samples)
        Pre-processed data with each Frequency separated
    all_offsets : list
        list of each frequency's best performing offset
    all_bin_widths : list
        list of each frequency's best performing bin_width
    label_events : list, shape [Label]->[Chunks]->[Events]
        Onsets of the Labels to be Clipped

    Returns
    -------
    all_best_chunk_events : list | [Freq]->[labels]->(Instances, 1, Channels, Samples)

    """
    all_best_chunk_events = []

    for pred_data, offset, bin_width in zip(freq_preprocessed_data, all_offsets, all_bin_widths):
        # Grab the Neural Activity Centered on Each event
        set_window = (offset - bin_width, offset)

        chunk_events = fet.event_clipper_nd(data=pred_data, label_events=label_events, fs=1000, window=set_window)
        # print(len(chunk_events))
        # print(len(chunk_events[0]))

        all_best_chunk_events.append(chunk_events)

    return all_best_chunk_events


def fix_label_instructions(label_instructions):
    """ reduce the label_instructions to just the ones I will be testing the "WHEN" analysis

    :param label_instructions:
    :return:
    """
    fixed_instructions = []
    for i in label_instructions:
        if isinstance(i, str):
            pass
        elif isinstance(i, int):
            fixed_instructions.append(i)

    fixed_instructions = fixed_instructions[1:]
    return fixed_instructions


def get_motif_identifier(focus, context, labels):
    """ Get all of the indexes of the Motifs each Syllable Instance occurs in

    Parameters
    ----------
    focus : list
        List of the Keys of the labels of interest
    context : list
        list of arrays of context labels for each Epoch.
        [Epoch #] -> (labels, 4)
            col: (Motif Sequence in Bout, First Motif (1-hot), Last Motif (1-hot), Last Syllable Dropped (1-hot))
    labels : list | [Epoch] -> [Labels]
        list of labels for all epochs for one day

    Returns
    -------
    focus_index : dict | {label: [motif indexes]}
        List of the motif index that each instance of a syllable belongs to
    """

    focus_index = dict()  # Initiate a Dictionary

    for i in focus:
        focus_index[i] = []  # Initiate the Nested List Structure

    sequential_counter = -1

    for chunk_contexts, chunk_labels in zip(context, labels):
        current_counter = 0  # Make sure to reset the Current Counter with each Chunk
        chunk_just_started = 1

        for motif_seq_numb, curr_label in zip(chunk_contexts[:, 0], chunk_labels):
            if motif_seq_numb == 0:
                current_counter = 0  # Reset the Current Motif Recognition

            if motif_seq_numb != 0:  # For the Duration of this Motif

                if motif_seq_numb != current_counter:  # If in a New Motif
                    current_counter = motif_seq_numb  # Update the Current Motif Recognition

                    if curr_label == 1:  # If in a Usable Motif
                        occured_during_motif = []  # Keep Track of What Occurs during this motif
                        sequential_counter += 1  # Increase Motif Counter
                        chunk_just_started = 0

                if curr_label in focus:
                    if curr_label in occured_during_motif or chunk_just_started == 1:  # Syllable w/o the First Motif
                        focus_index[curr_label].append('bad')  # If no 1st syllable then can't used Motif
                    else:
                        focus_index[curr_label].append(
                            sequential_counter)  # If syll occurs in current Motif then index it
                        occured_during_motif.append(curr_label)  # Keep track of what

    return focus_index


def organize_absolute_times(times_of_interest):
    """ Re-organize all of the absolute times to a more useful format

    Parameters
    ----------
    times_of_interest : list

    Returns
    -------
    all_absolute_times : list | [label types]->[Instances]
        list of all of the (Start of End) times from the times_of_interest parameter

    """
    all_absolute_times = []  # Labels
    for label_type in times_of_interest:
        absolute_times = []
        for chunk in label_type:
            for instances in chunk:
                absolute_times.append(instances)  # Chunks
        all_absolute_times.append(absolute_times)
    return all_absolute_times


# Make a Function to get the Relative Starts of All Instances of each Label

# focus = [2,3,4,5,6]


def relative_times(first_starts, focus_starts, motif_index):
    """ Calculate the Relative Starts of All Selected Starts for One Label

    Parameters
    ----------
    first_starts : list | [Instances]
        list of all of the (Start or End) times for the first_starts parameter
    focus_starts : list | [Instances]
        list of all of the (Start or End) times for the focus_starts parameter
    motif_index : list
        index of the first_starts motifs to use to the the relative starts from the focus_starts

    Returns
    -------

    """
    return np.asarray(focus_starts) - np.asarray(first_starts)[motif_index]


def get_all_relative_starts(first_starts, other_starts, all_other_index, labels_used):
    """ Calculate the Relative Starts of All  Instances of each Label, it also handles concerns regarding syllables
    that occur without a first syllable

    Parameters
    ----------
    first_starts : list | [Instances]
        list of all of the (Start or End) times for the first_starts parameter
    other_starts : list | [label types]->[[Instances]
        list of all of the (Start or End) times from the other_starts parameter
    all_other_index : dict | {label: [Motif Number]}
        dictionary of a list of all of the start times for each label
    labels_used : list | [Syllable Label]
        list of keys for the all_other_index variable indicating the labels to be used

    Returns
    -------
    all_rel_starts : list | [label]-> [Instances]
        list of all of the relative start times for each label in the order of labels_used

    Notes
    -----
    # Make an array of all of the relative start times of the syllable within there respective Motifs
    # Thes will be index to the Train and Test Sets for the Within Motif Onset Detection
    # The Train Set will create the Stereotyped Syllable Gap ( Mean_Gap - True_Gap)
    # The Test Set will find the Stereotyped Gap (Train_Mean_Gap - True_Gap),
    # and the Predicted Gap (Max_Location - True Gap - Before_Buffer)

    """

    all_rel_starts = []

    for label_starts, label_key in zip(other_starts, labels_used):
        good_syll_instances = [index for index, value in enumerate(all_other_index[label_key]) if value != 'bad']
        good_motif_instances = [value for value in all_other_index[label_key] if value != 'bad']

        relative_starts = np.asarray(label_starts)[good_syll_instances] - np.asarray(first_starts)[good_motif_instances]

        all_rel_starts.append(relative_starts)

    return all_rel_starts


def get_time_series_1d(data, bin_width):
    """ Get a Overlapping time windows given a 1d array
    """
    return np.array([x for x in zip(*(data[i:] for i in range(bin_width)))])


def get_time_series(data, bin_width):
    """  Get a Overlapping time windows given a nd array assuming the last dimmension is samples
    """
    return np.apply_along_axis(func1d=get_time_series_1d, axis=-1, arr=data, bin_width=bin_width)


def clip_motif_time_series(freq_preprocessed_data, all_offsets, all_bin_widths, motif_start_times, motif_length: int):
    """ Clips the times around the first syllable using the optimal Bin Width for each frequency

    Parameters
    ----------
    freq_preprocessed_data : [Freqs]->[Chunks]->(1, channels, samples)
        Pre-processed data with each Frequency separated
    all_offsets : list
        list of each frequency's best performing offset
    all_bin_widths : list
        list of each frequency's best performing bin_width
    motif_start_times : list | [Chunks]->[[Instances]
        list of all of the Start times for the first syllable in the motif
    motif_length : int
        Length of the stereotyped Motif Duration for the selected Bird

    Returns
    -------
    motif_events_series : list | [Frequency]->(Instances, Channels, Time-Steps, Bin Width)
        list of the times overlapping bins for each frequency to do the mini-onset detection
    """
    # [Freq]->(Instances, Frequency, Channels, Time-Steps, Bin Width)
    # Only need to get the times around the first syllable

    motif_events_series = []
    for pred_data, offset, bin_width in zip(freq_preprocessed_data, all_offsets, all_bin_widths):
        # Grab the Neural Activity Centered on Each event
        set_window = (offset - bin_width, offset + motif_length)
        chunk_events = fet.get_event_related_nd_chunk(chunk_data=pred_data, chunk_indices=motif_start_times, fs=1000,
                                                      window=set_window)  # clip the data at the start times

        corrected_chunk_events = []
        for chunk in chunk_events:
            corrected_chunk_events.append(np.squeeze(chunk))

        chunk_events = fet.event_shape_correction(chunk_events=corrected_chunk_events,
                                                  original_dim=2)  # Reformat to be array-like

        chunk_events_series = get_time_series(data=chunk_events, bin_width=bin_width)  # clip samples based on bin_width

        motif_events_series.append(np.squeeze(chunk_events_series))  # Remove Single axis and append to list

    return motif_events_series


# Use each template to get th Templates
# freq_time_series_data : ndarray | (instances, channels, time-steps, bin width)
# freq_template_data : ndarray |(channels, samples)

def get_freq_pearson_coefficients(freq_template_data, freq_time_series_data, selection=None):
    """
    Parameters
    ----------
    freq_template_data : ndarray |(channels, samples)
    freq_time_series_data : ndarray | (instances, channels, time-steps, bin width)
    selection : array, optional

    Returns
    -------
    corr_trials : array \ (instances, channels, time-steps)
        ndarray of Pearson Correlation Values for each time-step
    """
    # freq_time_series_data : ndarray | (instances, channels, time-steps, bin width)
    # freq_template_data : ndarray |(channels, samples)

    num_instances, num_channels, _, _ = np.shape(freq_time_series_data)
    if selection is not None:
        instance_index = selection
    else:
        instance_index = np.arange(num_instances)

    corr_trials = []  # Create Lists
    for instance in instance_index:
        channel_trials = []
        for channel in range(num_channels):
            corr_holder = cat.efficient_pearson_1d_v_2d(freq_template_data[channel, :],
                                                        freq_time_series_data[instance, channel, :, :])
            channel_trials.append(corr_holder)
        corr_trials.append(channel_trials)
    corr_trials = np.asarray(corr_trials)

    return corr_trials


# time_series first stored as [Frequency]->(Instances, Frequency, Channels, Time-Steps, Bin Width)
# Used to get the Templates [labels]->(instances, freq, channel, bin width)

def get_freq_labels_pearson_coefficients(freq_templates, freq_time_series_data, selections):
    labels_pearson = []
    for label_template, selection in zip(freq_templates, selections):
        pearson_holder = get_freq_pearson_coefficients(freq_template_data=label_template,
                                                       freq_time_series_data=freq_time_series_data,
                                                       selection=selection)

        labels_pearson.append(pearson_holder)
    return labels_pearson


# time_series first stored as [Frequency]->(Instances, Frequency, Channels, Time-Steps, Bin Width)
# Used to get the Templates [Frequency]->[labels]->(instances, freq, channel, bin width)

def get_all_freq_pearson_coefficients(all_freq_templates, freqs_time_series_data, selections):
    freqs_peason = []
    for freq_templates in all_freq_templates:
        pearson_holder = get_freq_labels_pearson_coefficients(freq_templates=all_freq_templates,
                                                              freq_time_series_data=freqs_time_series_data,
                                                              selections=selections)
        freqs_peason.append(pearson_holder)

    return freqs_peason


