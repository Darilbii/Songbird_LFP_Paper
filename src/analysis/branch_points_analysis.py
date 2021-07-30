# Import the new package made specifically for this analysis
from src.utils.paths import REPORTS_DIR

import src.features.chunk_onset_dev as cod

import BirdSongToolbox.free_epoch_tools as fet
from BirdSongToolbox.import_data import ImportData
from BirdSongToolbox.file_utility_functions import _handle_data_path, _save_pckl_data

import src.analysis.hilbert_based_pipeline as hbp
from src.analysis.ml_pipeline_utilities import all_bad_channels, all_drop_temps, all_label_instructions
from src.analysis.context_utility import birds_context_obj
from src.analysis.chunk_feature_dropping_pearson import best_bin_width, best_offset
import src.analysis.chunk_when_analysis_naive as cwan

import numpy as np
import scipy
from sklearn.model_selection import KFold
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib.backends.backend_pdf import PdfPages

# Location to save Data
branch_analysis_path = "/home/debrown/branch_analysis_path"

# Data Source for Script
onset_detection_path = '/home/debrown/onset_detection_results'

from BirdSongToolbox.file_utility_functions import _load_pckl_data

all_label_instruct = {'z007': [5],
                      'z020': [4],
                      'z017': [6, 7]
                      }


def naive_branch_analysis(absolute_relative_starts, motif_ledger, all_best_chunk_events, motif_time_series,
                          time_buffer, label_instruct):
    """ Algorithm to run a naive onset detection/predicition analysis.

    Parameters
    ----------
    absolute_relative_starts : list | [labels] -> [instances]
        list of all of the relative start times for each label in the order of labels_used
    motif_ledger : dict | { label_num : [Motif_identity]}
        A ledger of which motif a syllable instance occurred during. It is in order the syllables occurred assuming they
        started with the first syllable
    all_best_chunk_events : list | [Freq]->[labels]->(Instances, 1, Channels, Samples)
        Data clipped into the trials that can be used to make the Pearson Templates
    motif_time_series : list | [Frequency]->(Instances, Channels, Time-Steps, Bin Width)
        list of the times overlapping bins for each frequency to do the mini-onset detection
    time_buffer : int
        #Window around true start time

    label_instruct : list
        list of the syllable labels to use for the Particular bird

    Returns
    -------
    [syllables]->[Freqs]->[instances]

    elections: | [syllables]->[Freqs]->(instances, samples)
    """

    super_all_elections = []

    for sel_syll in label_instruct:

        syll_all_elections = []
        for sel_freq in range(len(all_best_chunk_events)):
            sel_label = sel_syll - 1

            expected_time = int(np.mean(absolute_relative_starts[sel_label - 1] / 30))
            expected_before = expected_time - time_buffer
            expected_after = expected_time + time_buffer
            print(expected_before, 'to', expected_after)

            syll_ex = np.arange(len(motif_ledger[sel_syll]))  # Index fo all of the Syllable examples
            branch_ex = [x for x in motif_ledger[2] if x not in motif_ledger[sel_syll]]  # Index of Motifs w/o the Syll

            # 1.) Get the Templates
            # 1.1) Sub-select Templates using Template Index
            sel_for_template = all_best_chunk_events[sel_freq][sel_label][syll_ex]
            sel_template = np.squeeze(np.mean(sel_for_template, axis=0))  # 1.2) Take the Mean (Chan, Samples)

            # 3.) Get the Time-Series [syllable](Instances, channels, samples, bin_width)
            # 3.1) Sub-select Motif Time Series using Test Index
            sel_time_series = motif_time_series[sel_freq][branch_ex]

            # 4.) Run the Test
            # 4.1) Get the Pearson Coefficients
            when_results = cod.get_freq_pearson_coefficients(freq_template_data=sel_template,
                                                             freq_time_series_data=sel_time_series, selection=None)

            when_results[when_results < 0] = 0  # Bottom Threshold (Remove negative correlation)
            when_elections = np.sum(when_results, axis=1)  # collapse across channels
            # onset_predictions = np.argmax(when_elections[:, expected_before:expected_after], axis=-1)

            syll_all_elections.append(when_elections)
            print('expected_before: ', expected_before)

        super_all_elections.append(syll_all_elections)

    return super_all_elections


def naive_branch_analysis_folded(absolute_relative_starts, motif_ledger, all_best_chunk_events, motif_time_series,
                                 time_buffer, label_instruct):
    """ Algorithm to run a naive onset detection/predicition analysis.


    absolute_relative_starts : list | [labels] -> [instances]
        list of all of the relative start times for each label in the order of labels_used
    motif_ledger : dict | { label_num : [Motif_identity]}
        A ledger of which motif a syllable instance occurred during. It is in order the syllables occurred assuming they
        started with the first syllable
    all_best_chunk_events : list | [Freq]->[labels]->(Instances, 1, Channels, Samples)
        Data clipped into the trials that can be used to make the Pearson Templates
    motif_time_series : list | [Frequency]->(Instances, Channels, Time-Steps, Bin Width)
        list of the times overlapping bins for each frequency to do the mini-onset detection
    time_buffer : int
        #Window around true start time

    label_instruct : list
        list of the syllable labels to use for the Particular bird

    :return:
    [syllables]->[Freqs]->[instances]

    elections: | [syllables]->[Freqs]->[folds]->(instances, samples)
    """

    super_all_elections = []

    for sel_syll in label_instruct:

        syll_all_elections = []
        for sel_freq in range(len(all_best_chunk_events)):
            sel_label = sel_syll - 1

            expected_time = int(np.mean(absolute_relative_starts[sel_label - 1] / 30))
            expected_before = expected_time - time_buffer
            expected_after = expected_time + time_buffer
            print(expected_before, 'to', expected_after)

            # Get the Folds used to make the templates in the Onset Detection analysis
            X = np.arange(len(motif_ledger[sel_syll]))
            y = np.ones((len(motif_ledger[sel_syll]),))
            kf = KFold(n_splits=5, shuffle=True, random_state=0)
            kf.get_n_splits(X)

            print(kf)

            all_elections = []

            # Get the Motifs that do Not have the Syllable (For Branch Analysis)

            # syll_ex = np.arange(len(motif_ledger[sel_syll]))  # Index fo all of the Syllable examples
            branch_ex = [x for x in motif_ledger[2] if x not in motif_ledger[sel_syll]]  # Index of Motifs w/o the Syll

            for train_index, test_index in kf.split(X):
                print("TRAIN:", train_index, "TEST:", test_index)  # These are the only Parts I need for the splitting

                # 1.) Get the Templates
                # 1.1) Sub-select Templates using Template Index
                sel_for_template = all_best_chunk_events[sel_freq][sel_label][train_index]
                sel_template = np.squeeze(np.mean(sel_for_template, axis=0))  # 1.2) Take the Mean (Chan, Samples)

                # 3.) Get the Time-Series [syllable](Instances, channels, samples, bin_width)
                # 3.1) Sub-select Motif Time Series using Branch Index
                sel_time_series = motif_time_series[sel_freq][branch_ex]

                # 4.) Run the Test
                # 4.1) Get the Pearson Coefficients
                when_results = cod.get_freq_pearson_coefficients(freq_template_data=sel_template,
                                                                 freq_time_series_data=sel_time_series, selection=None)

                when_results[when_results < 0] = 0  # Bottom Threshold (Remove negative correlation)
                when_elections = np.sum(when_results, axis=1)  # collapse across channels
                # onset_predictions = np.argmax(when_elections[:, expected_before:expected_after], axis=-1)
                all_elections.append(when_elections)
                print('expected_before: ', expected_before)

            syll_all_elections.append(all_elections)

        super_all_elections.append(syll_all_elections)

    return super_all_elections


def create_shuffled_ledger(motif_ledger):
    """Function to create a Functional Ledger of the location of each Motif after collapsing the shuffled folds

    out:
    shuffled_fold_ledger: dict | {syll: [index_of_instances]
        Note: They go from 0 to num_instances. They are the index of the order of the instances of each syllable
            and not the motif identity.
    """
    shuffled_fold_ledger = {}
    for syll in motif_ledger.keys():

        syllable_holder = []  # Make Holder

        # Get the Folds used to make the templates in the Onset Detection analysis
        X = np.arange(len(motif_ledger[syll]))
        y = np.ones((len(motif_ledger[syll]),))
        kf = KFold(n_splits=5, shuffle=True, random_state=0)
        kf.get_n_splits(X)
        for train_index, test_index in kf.split(X):
            #         print("TRAIN:", train_index, "TEST:", test_index)  # These are the only Parts I need for the splitting
            syllable_holder.append(test_index)  # add folds index order to the the list

        collapsed_index_test = cwan.collaspse_folds(syllable_holder)  # collapse folds
        shuffled_fold_ledger[syll] = collapsed_index_test  # enter entry for current syllable

    return shuffled_fold_ledger


def create_correct_shuffled_ledger(motif_ledger, shuffled_fold_ledger):
    """ function to correct the index value using the shuffled_fold_ledger

    Parameters
    ----------
    motif_ledger : dict | { label_num : [Motif_identity]}
        A ledger of which motif a syllable instance occurred during. It is in order the syllables occurred assuming they
        started with the first syllable
    shuffled_fold_ledger: dict | {syll: [index_of_instances]
        Note: They go from 0 to num_instances. They are the index of the order of the instances of each syllable
            and not the motif identity.

    Returns
    -------
    correct_shuff_ledger :

    """

    # Make a function to correct the index value using the shuffled_fold_ledger

    correct_shuff_ledger = {}
    for syll in motif_ledger.keys():
        holder = []

        for shuff_index in shuffled_fold_ledger[syll]:
            holder.append(motif_ledger[syll][shuff_index])

        correct_shuff_ledger[syll] = np.asarray(holder)
    return correct_shuff_ledger


def organize_onset_prediction(national_election: list, correct_shuff_ledger: dict, full_motif_index: list,
                              full_syllables: list, ):
    """Make an Index of all of the Full Motifs

    Parameters
    ----------
    national_election : list | [syllables]->[folds]->(instances, samples)
         The time-series of all Voting Results for each syllable
    ///
    abs_rel_starts : list | [syllables]->-[folds]>(instances,)
    ///
    correct_shuff_ledger : dict | {syll: [index_of_instances]
        Dictionary of the correct motif index each syllable occurs during in the same order that results from collapsing
        the random shuffled validation folds. This value is the result of using two helper functions.
    full_motif_index : array | (instances,)
        Index of all of the Motifs that have all of the syllables prior to the branch point of interest

    full_syllables : list | [syllables]
        list of syllables prior to the branch point, if there is a branch point, these syllables use the same coding as
        the motif_ledger.


    Returns
    -------
    prediction_timeseries : ndarray | (instances, syllables, samples)
        The prediction/confidence timeseries organized into a more usable format

    abs_rel_starts: ndarray | (instances, syllables,)

    make empty array to hold all of the relevant trials
    """

    # Make empty array to house organized data
    prediction_timeseries = np.zeros((len(full_motif_index), len(full_syllables),
                                      np.shape(national_election[0][0])[-1]))

    # For each index of syllables and prediction results
    for syll_index, (syll, syll_inst_data) in enumerate(zip(full_syllables, national_election)):

        collapsed_elections_data = cwan.collaspse_folds(syll_inst_data)  # Collapse Folds
        viable_motifs_index = correct_shuff_ledger[syll]  # get the Indexes of Motif that have all syllables

        # For each full Motif, get the specified syllable's data
        for motif_index, motif_inst in enumerate(full_motif_index):
            correct_shuffled_index = np.where(viable_motifs_index == motif_inst)  # Find motif index in shuffled ledger
            prediction_timeseries[motif_index, syll_index, :] = collapsed_elections_data[correct_shuffled_index, :]

    return prediction_timeseries


def organize_onset_abs_starts(abs_rel_starts: list, motif_ledger: dict, full_motif_index: list,
                              full_syllables: list, ):
    """Make an Index of all of the Full Motifs

    Parameters
    ----------
    abs_rel_starts : list | [syllables]->[folds]->(instances, samples)
         The time-series of all Voting Results for each syllable
    ///
    abs_rel_starts : list | [syllables]->(instances,)
    ///
    motif_ledger : dict | {syll: [index_of_instances]
       [INSERT HERE]

    full_motif_index : array | (instances,)
        Index of all of the Motifs that have all of the syllables prior to the branch point of interest

    full_syllables : list | [syllables]
        list of syllables prior to the branch point, if there is a branch point, these syllables use the same coding as
        the motif_ledger.


    Returns
    -------
    prediction_timeseries : ndarray | (instances, syllables, samples)
        The prediction/confidence timeseries organized into a more usable format

    abs_rel_starts: ndarray | (instances, syllables,)

    make empty array to hold all of the relevant trials
    """

    # Make empty array to house organized data
    prediction_timeseries = np.zeros((len(full_motif_index), len(full_syllables)))

    # For each index of syllables and prediction results
    for syll_index, (syll, relative_starts) in enumerate(zip(full_syllables, abs_rel_starts)):

        # collapsed_elections_data = cwan.collaspse_folds(relative_starts)  # Collapse Folds
        viable_motifs_index = np.asarray(motif_ledger[syll])  # get the Indexes of Motif that have all syllables

        # For each full Motif, get the specified syllable's data
        for motif_index, motif_inst in enumerate(full_motif_index):
            correct_shuffled_index = np.where(viable_motifs_index == motif_inst)  # Find motif index in shuffled ledger
            prediction_timeseries[motif_index, syll_index] = relative_starts[correct_shuffled_index]

    return prediction_timeseries


def organize_branch_prediction(branch_elections: list, motif_ledger: dict, branch_motif_index: list,
                               branch_syllables: list):
    """Make an Index of all of the Full Motifs

    Parameters
    ----------
    branch_elections : list | [syllables]->[folds]->(instances, samples)
         The time-series of all Voting Results for each syllable
    ///
    abs_rel_starts : list | [syllables]->-[folds]>(instances,)
    ///
    motif_ledger : dict | { label_num : [Motif_identity]}
        A ledger of which motif a syllable instance occurred during. It is in order the syllables occurred assuming they
        started with the first syllable
    branch_motif_index : array | (instances,)
        Index of all of the Motifs that branch at the point of focus

    branch_syllables : list | [syllables]
        list of syllables prior to the branch point, if there is a branch point, these syllables use the same coding as
        the motif_ledger.


    Returns
    -------
    branch_pred_timeseries : ndarray | (instances, syllables, folds, samples)
        The prediction/confidence timeseries for branch syllables organized into a more usable format

    abs_rel_starts: ndarray | (instances, syllables,)

    make empty array to hold all of the relevant trials
    """

    # Make empty array to house organized data
    branch_pred_timeseries = np.zeros((len(branch_motif_index), len(branch_syllables), 5,
                                       np.shape(branch_elections[0][0])[-1]))

    # For each index of syllables and prediction results
    for syll_index, (syll, syll_inst_data) in enumerate(zip(branch_syllables, branch_elections)):

        branch_elections_data = np.asarray(syll_inst_data)  # Shape: [folds]-> (instances, samples)

        # Create a Index of Motifs w/o the Syllable relavent to the branch point of interest
        viable_motifs_index = np.asarray([x for x in motif_ledger[2] if x not in motif_ledger[syll]])

        # For each full Motif, get the specified syllable's data
        for motif_index, motif_inst in enumerate(branch_motif_index):
            correct_shuffled_index = np.where(viable_motifs_index == motif_inst)  # Find motif index in shuffled ledger
            branch_pred_timeseries[motif_index, syll_index, :, :] = np.squeeze(
                np.squeeze(branch_elections_data[:, correct_shuffled_index, :]))

    return branch_pred_timeseries


def organize_branch_prediction_single(branch_elections: list, motif_ledger: dict, branch_motif_index: list,
                                      branch_syllables: list):
    """Make an Index of all of the Full Motifs

    Parameters
    ----------
    branch_elections : list | [folds]->(instances, samples)
         The time-series of all Voting Results for each syllable
    ///
    abs_rel_starts : list | [folds]>(instances,)
    ///
    motif_ledger : dict | { label_num : [Motif_identity]}
        A ledger of which motif a syllable instance occurred during. It is in order the syllables occurred assuming they
        started with the first syllable
    branch_motif_index : array | (instances,)
        Index of all of the Motifs that branch at the point of focus

    branch_syllables : list | [syllables]
        list of syllables prior to the branch point, if there is a branch point, these syllables use the same coding as
        the motif_ledger.


    Returns
    -------
    branch_pred_timeseries : ndarray | (instances, syllables, folds, samples)
        The prediction/confidence timeseries for branch syllables organized into a more usable format

    abs_rel_starts: ndarray | (instances, syllables,)

    make empty array to hold all of the relevant trials
    """

    # Make empty array to house organized data
    branch_pred_timeseries = np.zeros((len(branch_motif_index), len(branch_syllables), 5,
                                       np.shape(branch_elections[0][0])[-1]))

    # For each index of syllables and prediction results
    #     for syll_index, (syll, syll_inst_data) in enumerate(zip(branch_syllables, branch_elections)):

    branch_elections_data = np.asarray(branch_elections)  # Shape: [folds]-> (instances, samples)

    # Create a Index of Motifs w/o the Syllable relavent to the branch point of interest
    viable_motifs_index = np.asarray([x for x in motif_ledger[2] if x not in motif_ledger[branch_syllables[0]]])

    # For each full Motif, get the specified syllable's data
    for motif_index, motif_inst in enumerate(branch_motif_index):
        correct_shuffled_index = np.where(viable_motifs_index == motif_inst)  # Find motif index in shuffled ledger
        branch_pred_timeseries[motif_index, 0, :, :] = np.squeeze(
            np.squeeze(branch_elections_data[:, correct_shuffled_index, :]))

    return branch_pred_timeseries


def make_nested_list_of_results(absolute_relative_starts, national_election, branch_election_collapsed, syll_focus,
                                time_buff=50):
    syll_with = []
    syll_without = []

    for sel_fold in range(5):
        #     print(sel_fold)
        expected_time = int(np.mean(absolute_relative_starts[syll_focus] / 30))
        expected_before = expected_time - time_buff
        expected_after = expected_time + time_buff

        # for Syllable 6
        result_with = np.max(national_election[syll_focus][sel_fold][:, expected_before:expected_after], axis=-1)
        result_without = np.max(branch_election_collapsed[sel_fold][:, expected_before:expected_after],
                                axis=-1)

        syll_with.append(result_with)
        syll_without.append(result_without)

    syll_with = np.asarray(syll_with)
    syll_without = np.asarray(syll_without)

    return syll_with, syll_without


def make_array_of_max_results(absolute_relative_starts, national_election, branch_election_collapsed, syll_focus,
                              time_buff=50):
    syll_with = []
    syll_without = []

    for sel_fold in range(5):
        #     print(sel_fold)
        expected_time = int(np.mean(absolute_relative_starts[syll_focus] / 30))
        expected_before = expected_time - time_buff
        expected_after = expected_time + time_buff

        # for Syllable 6
        result_with = np.max(national_election[syll_focus][sel_fold][:, expected_before:expected_after], axis=-1)
        result_without = np.max(branch_election_collapsed[sel_fold][:, expected_before:expected_after],
                                axis=-1)

        syll_with.extend(result_with)
        syll_without.extend(result_without)

    syll_with = np.asarray(syll_with)
    syll_without = np.asarray(syll_without)

    return syll_with, syll_without


def get_branch_statistics(syll_with_list, syll_without_list):
    all_p_values = []
    for with_syllable, without_syllable in zip(syll_with_list, syll_without_list):
        t_stat, p_value = scipy.stats.ttest_ind(with_syllable, without_syllable, equal_var=False, alternative="greater")
        print(p_value)
        all_p_values.append(p_value)
    return all_p_values


def run_branch_point_analysis_script(bird_id, session):
    # Import the Data
    zdata = ImportData(bird_id=bird_id, session=session)

    # Get the Bird Specific Machine Learning Meta Data
    bad_channels = all_bad_channels[bird_id]
    drop_temps = all_drop_temps[bird_id]
    when_label_instructions = cwan.all_when_label_instructions[bird_id]
    time_buffer = 50

    if session == 'day-2016-09-09':
        bin_widths = [185, 155, 120, 50, 120]
        offsets = [-40, -25, 0, 0, 0]
    else:
        # Get the Best Parameters for Bindwidth and Offset
        bin_widths = best_bin_width[session]
        offsets = best_offset[session]

    # Reshape Handlabels into Useful Format
    chunk_labels_list, chunk_onsets_list = fet.get_chunk_handlabels(handlabels_list=zdata.song_handlabels)

    # Set the Frequency Bands to Be Used for Feature Extraction
    fc_lo = [4, 8, 25, 30, 50]
    fc_hi = [8, 12, 35, 50, 70]

    # Pre-Process the Data # TODO: Make this a function
    # [Freqs]->[Chunks]->(1, channels, samples)

    freq_pred_data = []
    for low, high in zip(fc_lo, fc_hi):
        pred_data = hbp.feature_extraction_chunk(neural_chunks=zdata.song_neural,
                                                 fs=1000,
                                                 l_freqs=[low],
                                                 h_freqs=[high],
                                                 hilbert=None,
                                                 bad_channels=bad_channels,
                                                 norm=False,
                                                 drop_bad=True,
                                                 verbose=True)
        freq_pred_data.append(pred_data)

    # Get the Bird Specific label Instructions
    label_instructions = all_label_instructions[bird_id]  # get this birds default label instructions

    times_of_interest = fet.label_extractor(all_labels=chunk_labels_list,
                                            starts=chunk_onsets_list[0],
                                            label_instructions=label_instructions)

    # [Freq]->[labels]->(Instances, 1, Channels, Samples)
    all_best_chunk_events_test = cod.clip_instances_for_templates(freq_preprocessed_data=freq_pred_data,
                                                                  all_offsets=offsets,
                                                                  all_bin_widths=bin_widths,
                                                                  label_events=times_of_interest)

    motif_length = cwan.motif_length_dict[bird_id]

    # [Frequency]->(Instances, Channels, Time-Steps, Bin Width)
    motif_time_series = cod.clip_motif_time_series(freq_preprocessed_data=freq_pred_data,
                                                   all_offsets=offsets,
                                                   all_bin_widths=bin_widths,
                                                   motif_start_times=times_of_interest[0],
                                                   motif_length=motif_length)

    # Create instance of the Context Labels Class
    testclass = birds_context_obj(bird_id=bird_id)

    # Get the Context Array for the Day's Data
    test_context = testclass.get_all_context_index_arrays(chunk_labels_list)

    # Make Index of the Motif the syllable belongs to
    # list of the labels (syllable 2 : 2)
    # when_label_instructions = cod.fix_label_instructions(label_instructions)

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

    cwan._remove_unusable_syllable(all_best_chunk_events=all_best_chunk_events_test, motif_ledger=motif_ledger,
                                   label_instruct=when_label_instructions)  # Remove Bad Syllable Instances

    cwan._fix_the_ledger(motif_ledger)  # New Function to remove 'bad' from the ledger

    label_instruct = all_label_instruct[bird_id]

    branch_elections_test = naive_branch_analysis_folded(absolute_relative_starts=absolute_relative_starts,
                                                         motif_ledger=motif_ledger,
                                                         all_best_chunk_events=all_best_chunk_events_test,
                                                         motif_time_series=motif_time_series,
                                                         time_buffer=time_buffer,
                                                         label_instruct=label_instruct)

    # Collapse Across Frequencies
    branch_election_collapsed = cwan.get_national_elections(elections=branch_elections_test)

    # Make a maping from the audio to the Corresponding Prediction Trials
    shuffled_fold_ledger = create_shuffled_ledger(motif_ledger=motif_ledger)

    # Make a ledger/index to correct the index value using the shuffled_fold_ledger
    correct_shuff_ledger = create_correct_shuffled_ledger(motif_ledger=motif_ledger,
                                                          shuffled_fold_ledger=shuffled_fold_ledger)
    ######
    # Save the National Elections
    national_election = _load_pckl_data(data_name="national_election_run_2", bird_id=bird_id,
                                        session=session,
                                        source=onset_detection_path, verbose=True)

    # Save the Motif Ledger
    absolute_relative_starts = _load_pckl_data(data_name="absolute_relative_starts_run_2", bird_id=bird_id,
                                               session=session, source=onset_detection_path, verbose=True)
    ######

    # Organize the Correlation trace for all of the full syllables
    full_motif_organized = organize_onset_prediction(national_election=national_election,
                                                     correct_shuff_ledger=correct_shuff_ledger,
                                                     full_motif_index=motif_ledger[label_instruct[-1]],
                                                     full_syllables=motif_ledger.keys())

    # Organize the Start Times for all of the full syllables
    full_motif_starts = organize_onset_abs_starts(abs_rel_starts=absolute_relative_starts,
                                                  motif_ledger=motif_ledger,
                                                  full_motif_index=motif_ledger[label_instruct[-1]],
                                                  full_syllables=motif_ledger.keys())

    # Write a Branch Point for The Script to handle the two main birds with Branch Points
    # This needs to save this data as a pickle so that I can import it and use it to both run Statistics and Make Figures

    # Explicitly set three if statements for each bird:
    if bird_id == "z017":
        # Branch 1: Full Syllable

        branch_1 = [x for x in motif_ledger[5] if x not in motif_ledger[6]]  # Index of Motifs w/o the Syll 7

        # Correlation trace for all of the Syllables prior to the branch point
        branch_organized_1 = organize_onset_prediction(national_election=national_election,
                                                       correct_shuff_ledger=correct_shuff_ledger,
                                                       full_motif_index=branch_1, full_syllables=[2, 3, 4, 5])

        # Correlation trace for all of the branch syllable(s)
        branch_predictions_1 = organize_branch_prediction(branch_elections=branch_election_collapsed,
                                                          motif_ledger=motif_ledger, branch_motif_index=branch_1,
                                                          branch_syllables=[6, 7])

        branch_2 = np.asarray(
            [x for x in motif_ledger[6] if x not in motif_ledger[7]])  # Index of Motifs w/o the Syll 7

        # Correlation trace for all of the Syllables prior to the branch point
        branch_organized_2 = organize_onset_prediction(national_election=national_election,
                                                       correct_shuff_ledger=correct_shuff_ledger,
                                                       full_motif_index=branch_2,
                                                       full_syllables=[2, 3, 4, 5, 6])

        # Correlation trace for all of the branch syllable(s)
        branch_predictions_2 = organize_branch_prediction_single(branch_elections=branch_election_collapsed[1],
                                                                 motif_ledger=motif_ledger,
                                                                 branch_motif_index=branch_2,
                                                                 branch_syllables=[7])

        # Get Branch 1: absolute starts for syllables before the branch point
        branch_starts_1 = organize_onset_abs_starts(abs_rel_starts=absolute_relative_starts,
                                                    motif_ledger=motif_ledger,
                                                    full_motif_index=branch_1,
                                                    full_syllables=[2, 3, 4, 5])

        # Get Branch 2: absolute starts for syllables before the branch point
        branch_starts_2 = organize_onset_abs_starts(abs_rel_starts=absolute_relative_starts,
                                                    motif_ledger=motif_ledger,
                                                    full_motif_index=branch_2,
                                                    full_syllables=[2, 3, 4, 5, 6])

        ## Run Stats
        # Make a Nested List to run Statistics for Sylable 6
        with_nest6, without_nest6 = make_nested_list_of_results(absolute_relative_starts=absolute_relative_starts,
                                                                national_election=national_election,
                                                                branch_election_collapsed=branch_election_collapsed[0],
                                                                syll_focus=4, time_buff=time_buffer)

        syll_6_stats = get_branch_statistics(syll_with_list=with_nest6, syll_without_list=without_nest6)

        max_with_6, max_without_6 = make_array_of_max_results(absolute_relative_starts=absolute_relative_starts,
                                                              national_election=national_election,
                                                              branch_election_collapsed=branch_election_collapsed[0],
                                                              syll_focus=4, time_buff=time_buffer)

        # Make a Nested List to run Statistics for Sylable 7
        with_nest7, without_nest7 = make_nested_list_of_results(absolute_relative_starts=absolute_relative_starts,
                                                                national_election=national_election,
                                                                branch_election_collapsed=branch_election_collapsed[1],
                                                                syll_focus=5, time_buff=time_buffer)

        syll_7_stats = get_branch_statistics(syll_with_list=with_nest7, syll_without_list=without_nest7)

        max_with_7, max_without_7 = make_array_of_max_results(absolute_relative_starts=absolute_relative_starts,
                                                              national_election=national_election,
                                                              branch_election_collapsed=branch_election_collapsed[1],
                                                              syll_focus=5, time_buff=time_buffer)

        ## Save the Componenets for the Summary Figure
        # Save the DataFrame of Onset Results
        _save_pckl_data(data=branch_organized_1, data_name="branch_organized_1", bird_id=bird_id,
                        session=session,
                        destination=branch_analysis_path, make_parents=True, verbose=True)
        _save_pckl_data(data=branch_predictions_1, data_name="branch_predictions_1", bird_id=bird_id,
                        session=session,
                        destination=branch_analysis_path, make_parents=True, verbose=True)
        _save_pckl_data(data=branch_organized_2, data_name="branch_organized_2", bird_id=bird_id,
                        session=session,
                        destination=branch_analysis_path, make_parents=True, verbose=True)
        _save_pckl_data(data=branch_predictions_2, data_name="branch_predictions_2", bird_id=bird_id,
                        session=session,
                        destination=branch_analysis_path, make_parents=True, verbose=True)

        _save_pckl_data(data=branch_starts_1, data_name="branch_starts_1", bird_id=bird_id,
                        session=session,
                        destination=branch_analysis_path, make_parents=True, verbose=True)
        _save_pckl_data(data=branch_starts_2, data_name="branch_starts_2", bird_id=bird_id,
                        session=session,
                        destination=branch_analysis_path, make_parents=True, verbose=True)

        # Stats & Analysis
        _save_pckl_data(data=syll_6_stats, data_name="syll_6_stats", bird_id=bird_id,
                        session=session,
                        destination=branch_analysis_path, make_parents=True, verbose=True)
        _save_pckl_data(data=max_with_6, data_name="max_with_6", bird_id=bird_id,
                        session=session,
                        destination=branch_analysis_path, make_parents=True, verbose=True)
        _save_pckl_data(data=max_without_6, data_name="max_without_6", bird_id=bird_id,
                        session=session,
                        destination=branch_analysis_path, make_parents=True, verbose=True)

        _save_pckl_data(data=syll_7_stats, data_name="syll_7_stats", bird_id=bird_id,
                        session=session,
                        destination=branch_analysis_path, make_parents=True, verbose=True)
        _save_pckl_data(data=max_with_7, data_name="max_with_7", bird_id=bird_id,
                        session=session,
                        destination=branch_analysis_path, make_parents=True, verbose=True)
        _save_pckl_data(data=max_without_7, data_name="max_without_7", bird_id=bird_id,
                        session=session,
                        destination=branch_analysis_path, make_parents=True, verbose=True)



    elif bird_id == "z020":
        # Branch 1: Full Syllable

        branch_1 = [x for x in motif_ledger[3] if x not in motif_ledger[4]]  # Index of Motifs w/o the Syll 7

        # Correlation trace for all of the Syllables prior to the branch point
        branch_organized_1 = organize_onset_prediction(national_election=national_election,
                                                       correct_shuff_ledger=correct_shuff_ledger,
                                                       full_motif_index=branch_1, full_syllables=[2, 3])

        # Correlation trace for all of the branch syllable(s)
        branch_predictions_1 = organize_branch_prediction(branch_elections=branch_election_collapsed,
                                                          motif_ledger=motif_ledger, branch_motif_index=branch_1,
                                                          branch_syllables=[4])

        # Get Branch 1: absolute starts for syllables before the branch point
        branch_starts_1 = organize_onset_abs_starts(abs_rel_starts=absolute_relative_starts,
                                                    motif_ledger=motif_ledger,
                                                    full_motif_index=branch_1,
                                                    full_syllables=[2, 3])

        ## Run Statistics
        # Make a Nested List to run Statistics for Sylable 4 (syll 4 => syll_focus = 2)
        with_nest4, without_nest4 = make_nested_list_of_results(absolute_relative_starts=absolute_relative_starts,
                                                                national_election=national_election,
                                                                branch_election_collapsed=branch_election_collapsed[0],
                                                                syll_focus=2, time_buff=time_buffer)

        syll_4_stats = get_branch_statistics(syll_with_list=with_nest4, syll_without_list=without_nest4)

        max_with_4, max_without_4 = make_array_of_max_results(absolute_relative_starts=absolute_relative_starts,
                                                              national_election=national_election,
                                                              branch_election_collapsed=branch_election_collapsed[0],
                                                              syll_focus=2, time_buff=time_buffer)

        _save_pckl_data(data=branch_organized_1, data_name="branch_organized_1", bird_id=bird_id,
                        session=session,
                        destination=branch_analysis_path, make_parents=True, verbose=True)
        _save_pckl_data(data=branch_predictions_1, data_name="branch_predictions_1", bird_id=bird_id,
                        session=session,
                        destination=branch_analysis_path, make_parents=True, verbose=True)
        _save_pckl_data(data=branch_starts_1, data_name="branch_starts_1", bird_id=bird_id,
                        session=session,
                        destination=branch_analysis_path, make_parents=True, verbose=True)

        # Stats & Analysis
        _save_pckl_data(data=syll_4_stats, data_name="syll_4_stats", bird_id=bird_id,
                        session=session,
                        destination=branch_analysis_path, make_parents=True, verbose=True)
        _save_pckl_data(data=max_with_4, data_name="max_with_4", bird_id=bird_id,
                        session=session,
                        destination=branch_analysis_path, make_parents=True, verbose=True)
        _save_pckl_data(data=max_without_4, data_name="max_without_4", bird_id=bird_id,
                        session=session,
                        destination=branch_analysis_path, make_parents=True, verbose=True)

    elif bird_id == "z007":
        pass

    else:
        print("this subject was not accounted for in this script")

    # Save the organized results for the Full Motifs
    _save_pckl_data(data=full_motif_organized, data_name="full_motif_organized", bird_id=bird_id,
                    session=session,
                    destination=branch_analysis_path, make_parents=True, verbose=True)
    _save_pckl_data(data=full_motif_starts, data_name="full_motif_starts", bird_id=bird_id,
                    session=session,
                    destination=branch_analysis_path, make_parents=True, verbose=True)
