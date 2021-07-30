import numpy as np
import numpy as np
import pandas as pd
import pycircstat

import BirdSongToolbox.free_epoch_tools as fet
from BirdSongToolbox.import_data import ImportData

import src.analysis.chunk_when_analysis_naive as cwan
import src.analysis.hilbert_based_pipeline as hbp
import src.features.chunk_onset_dev as cod
from src.analysis.chunk_when_analysis_naive import _remove_unusable_syllable
from src.analysis.chunk_when_analysis_naive import all_when_label_instructions, motif_length_dict
from src.analysis.chunk_when_analysis_naive import save_pandas_to_pickle
from src.analysis.context_utility import birds_context_obj
from src.analysis.ml_pipeline_utilities import all_bad_channels, all_label_instructions

syllable_phase_preference_path = '/home/debrown/syllable_phase_preference'


def _remove_unusable_syllable(all_chunk_events, motif_ledger, when_label_instruct):
    # [syllables]->(instances, freqs, chans, samples)
    # fix the all_chunk_events
    for label in when_label_instruct:
        need_to_drop = [index for index, value in enumerate(motif_ledger[label]) if value == 'bad']
        all_chunk_events[label - 1] = np.delete(all_chunk_events[label - 1], obj=need_to_drop, axis=0)


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

    return itc_results_vector, itc_results_p, itc_results_z


def get_syll_z_results(itc_results_z, stereo_starts):
    # Get the Difference between the Syllable Centered Z and the Stereotyped Z

    diff_z_results = []
    for itc_result, stereo_time in zip(itc_results_z[1:], stereo_starts):
        center_result = itc_result[:, :, 0]
        stereo_result = itc_results_z[0][:, :, stereo_time]
        diff_results = center_result - stereo_result
        diff_z_results.append(diff_results)

    return diff_z_results


def create_syll_z_pd(when_labels, flat_diff_z_results):
    holder = []
    for syll, syll_diffs in zip(when_labels, flat_diff_z_results):
        syll_pd = pd.DataFrame(syll_diffs, columns=['Difference'])  # Make the DataFrame
        syll_pd['Syllable'] = str(syll)
        holder.append(syll_pd)
    full_results = pd.concat(holder, axis=0)

    return full_results


def run_syllable_phase_preference_analysis(bird_id='z007', session='day-2016-09-09'):
    zdata = ImportData(bird_id=bird_id, session=session)

    # Get bird Specific information
    when_label_instructions = all_when_label_instructions[bird_id]
    # Get the Bird Specific Machine Learning Meta Data
    bad_channels = all_bad_channels[bird_id]
    # Get the Window
    motif_dur = motif_length_dict[bird_id]

    # Format Hand labels for use
    chunk_labels_list, chunk_onsets_list = fet.get_chunk_handlabels(handlabels_list=zdata.song_handlabels)

    # Switch to the Log Spaced Bins
    freq_bins = 100
    fc_lo = np.logspace(np.log10(2), np.log10(220), freq_bins)
    fc_hi = np.logspace(np.log10(3), np.log10(250), freq_bins)

    # Pre-process all of the Data

    proc_data = hbp.itc_phase_chunk(neural_chunks=zdata.song_neural,
                                    fs=1000,
                                    l_freqs=fc_lo,
                                    h_freqs=fc_hi,
                                    verbose=True)

    # Get Starts of Events of Interest

    # Helper Function to create the properly initialized context class
    testclass = birds_context_obj(bird_id=bird_id)

    # Get the Context Array for the Day’s Data
    test_context = testclass.get_all_context_index_arrays(chunk_labels_list)

    # Get the Bird Specific label Instructions
    label_instructions = all_label_instructions[bird_id]  # get this birds default label instructions

    times_of_interest = fet.label_extractor(all_labels=chunk_labels_list,
                                            starts=chunk_onsets_list[0],
                                            label_instructions=label_instructions)

    # Grab the Neural Activity Centered on Each event
    set_window = (0, motif_dur)
    chunk_events = fet.event_clipper_nd(data=proc_data, label_events=times_of_interest,
                                        fs=1000, window=set_window)
    when_label_instructions = all_when_label_instructions[bird_id]

    #  {label: [motif indexes]}
    motif_ledger = cod.get_motif_identifier(focus=when_label_instructions,
                                            context=test_context,
                                            labels=chunk_labels_list)

    # Get all of the Relative Starts of the Syllables in their Respective Motifs
    # [label]-> [Instances]
    organized_times_of_interest = cod.organize_absolute_times(times_of_interest)

    # Get the Relative Start Times (in 30KHz Samples) [Need to divide by 30 to get the onset in ms]
    absolute_relative_starts = cod.get_all_relative_starts(first_starts=organized_times_of_interest[0],
                                                           other_starts=organized_times_of_interest[1:],
                                                           all_other_index=motif_ledger,
                                                           labels_used=when_label_instructions)

    stereo_starts = []
    for label, times in zip(when_label_instructions, absolute_relative_starts):
        start_mean = int(np.mean(times) / 30)
        stereo_starts.append(start_mean)

    # Remove Syllable that don't have a relative first Syllable
    _remove_unusable_syllable(all_chunk_events=chunk_events,
                              motif_ledger=motif_ledger,
                              when_label_instruct=when_label_instructions)

    # New Function to remove 'bad' from the ledger
    cwan._fix_the_ledger(motif_ledger)

    itc_results_vector, itc_results_p, itc_results_z = run_itc_analysis(chunk_events_data=chunk_events)

    # Run Comparison Analysis
    # Make a index for the y axis
    test_y = np.round(((fc_hi - fc_lo) / 2) + fc_lo)

    diff_z_results = get_syll_z_results(itc_results_z=itc_results_z, stereo_starts=stereo_starts)

    # Remove Bad Channels
    diff_z_results_corr = np.delete(diff_z_results, bad_channels, axis=2)

    flat_diff_z_results = np.reshape(np.asarray(diff_z_results_corr), (len(when_label_instructions), -1))

    full_results = create_syll_z_pd(when_labels=when_label_instructions, flat_diff_z_results=flat_diff_z_results)

    # Save the Motif Ledger
    save_pandas_to_pickle(data=full_results, data_name="syllable_phase_preference", bird_id=bird_id,
                          session=session, destination=syllable_phase_preference_path, make_parents=True, verbose=True)
