import numpy as np


import pycircstat

import BirdSongToolbox.free_epoch_tools as fet

from BirdSongToolbox.import_data import ImportData
from BirdSongToolbox.context_hand_labeling import label_focus_context, first_context_func, last_context_func
from src.analysis.ml_pipeline_utilities import all_label_instructions

import src.analysis.hilbert_based_pipeline as hbp

import src.analysis.ml_pipeline_utilities as mlpu
from src.analysis.context_utility import birds_context_obj, all_last_syllable


def selected_motifs_to_remove(bird_id='z007', session='day-2016-09-11'):
    """For visualization motifs were removed to illustrate more stereotyped behavior, this is to compensate for not
    dynamically time warping. The reports created to first show this phenomena doesn't remove these motifs"""

    if bird_id == 'z020':

        exemplar_chan = 11

        if session == 'day-2016-06-03':  # Day 1
            # Cherry Pick Motifs for the Visualization:
            first_rm = [0, 1, 3, 4, 5, 8, 11, 16, 17, 19, 20, 23, 26, 32, 35, 36, 39]  # Last 3 are from code glitch
            last_rm = [0, 1, 2, 3, 15, 16, 17, 19, 21, 25, 26, 27, 28, 31, 34, 36, 37, 39, 42, 44]

        elif session == 'day-2016-06-05':  # Day 2
            # Cherry Pick Motifs for the Visualization:
            # 4
            first_rm = [2, 3, 7, 9, 10, 15, 17, 18, 27, 29]  # Last 3 are from code glitch
            last_rm = [0, 2, 4, 10, 11, 12, 19, 25, 27, 29, 31]

    elif bird_id == 'z007':

        exemplar_chan = 17

        if session == 'day-2016-09-10':  # Day 1
            # Cherry Pick Motifs for the Visualization:
            first_rm = [11, 12, 13]  # Last 3 are from code glitch
            last_rm = [1, 5]

        elif session == 'day-2016-09-11':  # Day 2
            # Cherry Pick Motifs for the Visualization:
            first_rm = [6, 13, 14, 15, 16, 20, 31, 7, 8, 36]  # Last 3 are from code glitch
            last_rm = [6, 11, 13, 17, 19, 20, 21, 33]

    elif bird_id == 'z017':

        exemplar_chan = 14

        if session == 'day-2016-06-19':  # Day 1
            # Cherry Pick Motifs for the Visualization:
            first_rm = [0, 1, 6, 7, 21, 30, 33]  # Last 3 are from code glitch
            last_rm = [6, 16, 17, 22, 27, 28, 34]

        elif session == 'day-2016-06-21':  # Day 2
            # Cherry Pick Motifs for the Visualization:
            first_rm = [1, 4, 13, 19, 20, 24, 29, 31, 32]  # Last 3 are from code glitch
            last_rm = [1, 2, 8, 11, 12, 20, 26, 30, ]
            # 10?
    else:
        raise NameError  # Somehow Used a Subject and Day that wasn't shown in the paper

    return first_rm, last_rm, exemplar_chan


# This should be moved to a visualization specific module
from src.analysis.chunk_spectral_perturbation_report import plot_behavior_test
from src.analysis.context_utility import birds_context_obj


def get_itpc_statistical_significance(bird_id='z007', session='day-2016-09-11'):
    zdata = ImportData(bird_id=bird_id, session=session)

    # Get Handlabels
    chunk_labels_list, chunk_onsets_list = fet.get_chunk_handlabels(handlabels_list=zdata.song_handlabels)

    # Switch to the Log Spaced Bins
    freq_bins = 100
    fc_lo = np.logspace(np.log10(2), np.log10(220), freq_bins)
    fc_hi = np.logspace(np.log10(3), np.log10(250), freq_bins)

    proc_data = hbp.itc_phase_chunk(neural_chunks=zdata.song_neural,
                                    fs=1000,
                                    l_freqs=fc_lo,
                                    h_freqs=fc_hi,
                                    verbose=True)

    # Helper Function to create the properly initialized context class
    testclass = birds_context_obj(bird_id=bird_id)

    # Get the Context Array for the Day's Data
    test_context = testclass.get_all_context_index_arrays(chunk_labels_list)

    # Select Labels Using Flexible Context Selection
    first_syll = label_focus_context(focus=1,
                                     labels=chunk_labels_list,
                                     starts=chunk_onsets_list[0],
                                     contexts=test_context,
                                     context_func=first_context_func)

    last_syll = label_focus_context(focus=all_last_syllable[bird_id],
                                    labels=chunk_labels_list,
                                    starts=chunk_onsets_list[1],
                                    contexts=test_context,
                                    context_func=last_context_func)

    # Set the Context Windows

    first_window = (-500, 800)
    last_window = (-800, 300)

    first_rm, last_rm, exemplar_chan = selected_motifs_to_remove(bird_id=bird_id, session=session)

    # Clip around Events of Interest
    all_firsts = fet.get_event_related_nd_chunk(chunk_data=proc_data, chunk_indices=first_syll,
                                                fs=1000, window=first_window)

    all_lasts = fet.get_event_related_nd_chunk(chunk_data=proc_data, chunk_indices=last_syll,
                                               fs=1000, window=last_window)

    # Correct The Shape of the Data

    all_firsts = fet.event_shape_correction(all_firsts, original_dim=3)

    all_lasts = fet.event_shape_correction(all_lasts, original_dim=3)

    # Remove the selected motifs
    all_firsts = np.delete(all_firsts, first_rm, axis=0)
    all_lasts = np.delete(all_lasts, last_rm, axis=0)

    # First Motif ITPC
    first_itc = pycircstat.resultant_vector_length(np.asarray(all_firsts), axis=0)
    first_itc_p, first_itc_z = pycircstat.rayleigh(np.asarray(all_firsts), axis=0)

    # Last Motif ITPC
    last_itc = pycircstat.resultant_vector_length(np.asarray(all_lasts), axis=0)
    last_itc_p, last_itc_z = pycircstat.rayleigh(np.asarray(all_lasts), axis=0)

    # Steps to Getting the Values that I want:
    # Print the Maximum P-value for First
    print("Print the Maximum P-value for First:")
    print(np.max(first_itc_p[:, exemplar_chan, :][first_itc_z[:, exemplar_chan, :] > 5]))
    print("")

    # Print the P-values for Z>5 for First
    print("Print the P-values for Z>5 for First:")
    print(np.max(first_itc_p[:, exemplar_chan, :][first_itc_z[:, exemplar_chan, :] > 5]))
    print("")

    # Print the Maximum P-value for last
    print("Print the Maximum P-value for last:")
    print(np.max(last_itc_p[:, exemplar_chan, :][last_itc_z[:, exemplar_chan, :] > 5]))
    print("")

    # Print the P-values for Z>5 for last
    print("Print the P-values for Z>5 for last:")
    print(np.max(last_itc_p[:, exemplar_chan, :][last_itc_z[:, exemplar_chan, :] > 5]))
    print("")


def get_itpc_single_statistical_significance(bird_id='z007', session='day-2016-09-11'):
    zdata = ImportData(bird_id=bird_id, session=session)

    # Get Handlabels
    chunk_labels_list, chunk_onsets_list = fet.get_chunk_handlabels(handlabels_list=zdata.song_handlabels)

    # Switch to the Log Spaced Bins
    freq_bins = 100
    fc_lo = np.logspace(np.log10(2), np.log10(220), freq_bins)
    fc_hi = np.logspace(np.log10(3), np.log10(250), freq_bins)

    proc_data = hbp.itc_phase_chunk(neural_chunks=zdata.song_neural,
                                    fs=1000,
                                    l_freqs=fc_lo,
                                    h_freqs=fc_hi,
                                    verbose=True)

    # Helper Function to create the properly initialized context class
    testclass = birds_context_obj(bird_id=bird_id)

    # Get the Context Array for the Day's Data
    test_context = testclass.get_all_context_index_arrays(chunk_labels_list)

    label_instructions = all_label_instructions[bird_id]  # Removing the Silence due to its special needs
    times_of_interest = fet.label_extractor(all_labels=chunk_labels_list,
                                            starts=chunk_onsets_list[0],
                                            label_instructions=label_instructions)

    # Grab the Neural Activity Centered on Each event
    set_window = (-500, 500)
    chunk_events = fet.event_clipper_nd(data=proc_data, label_events=times_of_interest,
                                        fs=1000, window=set_window)

    chunk_events = mlpu.balance_classes(chunk_events)

    def run_itc_analysis(chunk_events_data):
        # Run the ITC over each Label Type

        # test_itc = pycircstat.resultant_vector_length(np.asarray(label_focus), axis=0)
        # test_itc_p, test_itc_z = pycircstat.rayleigh(np.asarray(label_focus), axis=0)

        itc_results_vector = []
        itc_results_p = []
        itc_results_z = []

        for label_type in chunk_events_data:
            itc_vector = pycircstat.resultant_vector_length(np.asarray(label_type), axis=0)
            itc_p, itc_z = pycircstat.rayleigh(np.asarray(label_type), axis=0)

            itc_results_vector.append(itc_vector)
            itc_results_p.append(itc_p)
            itc_results_z.append(itc_z)

        return np.asarray(itc_results_vector), np.asarray(itc_results_p), np.asarray(itc_results_z)

    _, _, exemplar_chan = selected_motifs_to_remove(bird_id=bird_id, session=session)

    itc_results_vector, itc_results_p, itc_results_z = run_itc_analysis(chunk_events_data=chunk_events)



    # Steps to Getting the Values that I want:
    # Print the Maximum P-value Accross all Syllables
    print("Print the Maximum P-value for First:")
    print(np.max(itc_results_p[:, :, exemplar_chan, :][itc_results_z[:, :, exemplar_chan, :] > 5]))
    print("")

    # Print the P-values for Z>5 Accross all Syllables
    print("Print the P-values for Z>5 for First:")
    print(np.max(itc_results_p[:, :, exemplar_chan, :][itc_results_z[:, :, exemplar_chan, :] > 5]))
    print("")



