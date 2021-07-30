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

import numpy as np
from sklearn.model_selection import KFold
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib.backends.backend_pdf import PdfPages

onset_detection_path = '/home/debrown/onset_detection_results'

# Bird's Default Class Labels
all_when_label_instructions = {'z007': [2, 3, 4, 5],
                               'z020': [2, 3, 4],
                               'z017': [2, 3, 4, 5, 6, 7]
                               }
# Make Motif Length Times for the analysis
motif_length_dict = {'z007': 750,
                     'z020': 500,
                     'z017': 900}


def save_pandas_to_pickle(data: pd.DataFrame, data_name: str, bird_id: str, session: str, destination: str,
                          make_parents=False, verbose=False):
    """

    Parameters
    ----------
    data : ndarray
    data_name : str
    bird_id : str
    session : str
    destination : str, pathlib.Path, optional
        Desitnation to save the pickle file
    make_parents : bool, optional
        If True, it will create all of the parent folders for the Data File

    """

    file_name = data_name + '.pckl'  # Add the .pckl stem to the data_name

    data_file_path = _handle_data_path(data_name=file_name, bird_id=bird_id, session=session,
                                       dir_path=destination,
                                       make_parents=make_parents)  # Handle File Path and Directory Structure

    # Internally use Pandas native saving function
    data.to_pickle(data_file_path)

    if verbose:
        # print(f"Saving {data_name} Data to", data_file_path.name)  # Uncomment once py3.5 support Dropped
        print("Saving {data_name} Data to {file_path}".format(data_name=data_name, file_path=data_file_path.name))


def load_pandas_from_pickle(data_name: str, bird_id: str, session: str, source: str, verbose=False):
    """

    Parameters
    ----------
    data_name : str
    bird_id : str
    session : str
    source : str, pathlib.Path, optional
        Source to load the pickle file from

    Returns
    -------
    data : pckl
        The pickle data object to be loaded
    """
    file_name = data_name + '.pckl'

    data_file_path = _handle_data_path(data_name=file_name, bird_id=bird_id, session=session,
                                       dir_path=source, make_parents=False)

    # Internally use Pandas native loading function
    data = pd.read_pickle(data_file_path)

    if verbose:
        # print(f"Saving {data_name} Data to", data_file_path.name)  # Uncomment once py3.5 support Dropped
        print("Loading from {data_name} Data to {file_path}".format(data_name=data_name, file_path=data_file_path.name))

    return data


def naive_onset_detection(absolute_relative_starts, motif_ledger, all_best_chunk_events, motif_time_series,
                          time_buffer, label_instruct):
    """


    :param absolute_relative_starts: list | [labels] -> [instances]
    :param motif_ledger: dict | { label_num : [Motif_identity]}
    :param all_best_chunk_events: list | [Freq]->[labels]->(Instances, 1, Channels, Samples)
    :param motif_time_series: list | [Frequency]->(Instances, Channels, Time-Steps, Bin Width)
    :param time_buffer: int
    # Window around true start time
    :param label_instruct: list
        list of the syllable labels to use for the Particular bird

    :return:
    [syllables]->[Freqs]->[folds]->[instances]

    elections: | [syllables]->[Freqs]->[folds]->(instances, samples)
    """
    super_all_relative_starts = []
    super_all_stereotype_starts = []
    super_all_predicted_starts = []  # (fold, Instances, Channels)
    super_all_elections = []
    super_all_onset_predictions = []

    # sel_freq = 0

    for sel_syll in label_instruct:
        syll_all_relative_starts = []
        syll_all_stereotype_starts = []
        syll_all_predicted_starts = []
        syll_all_elections = []
        # syll_all_onset_predictions = []
        for sel_freq in range(len(all_best_chunk_events)):
            sel_label = sel_syll - 1

            expected_time = int(np.mean(absolute_relative_starts[sel_label - 1] / 30))
            expected_before = expected_time - time_buffer
            expected_after = expected_time + time_buffer
            print(expected_before, 'to', expected_after)

            X = np.arange(len(motif_ledger[sel_syll]))
            y = np.ones((len(motif_ledger[sel_syll]),))
            kf = KFold(n_splits=5, shuffle=True, random_state=0)
            kf.get_n_splits(X)

            print(kf)

            all_relative_starts = []
            all_stereotype_starts = []
            all_predicted_starts = []
            all_elections = []
            all_onset_predictions = []

            for train_index, test_index in kf.split(X):
                print("TRAIN:", train_index, "TEST:", test_index)  # These are the only Parts I need for the splitting

                # 1.) Get the Templates
                # 1.1) Sub-select Templates using Template Index
                sel_for_template = all_best_chunk_events[sel_freq][sel_label][train_index]
                sel_template = np.squeeze(np.mean(sel_for_template, axis=0))  # 1.2) Take the Mean (Chan, Samples)

                # 3.) Get the Time-Series [syllable](Instances, channels, samples, bin_width)
                # 3.1) Sub-select Motif Time Series using Test Index
                sel_time_series = motif_time_series[sel_freq][np.asarray(motif_ledger[sel_syll])[test_index]]

                # 4.) Run the Test
                # 4.1) Get the Pearson Coefficients
                when_results = cod.get_freq_pearson_coefficients(freq_template_data=sel_template,
                                                                 freq_time_series_data=sel_time_series, selection=None)

                # when_results : ndarray | (instances, channels, time-steps)
                when_results[when_results < 0] = 0  # Bottom Threshold (Remove negative correlation)
                when_elections = np.sum(when_results, axis=1)  # collapse across channels
                onset_predictions = np.argmax(when_elections[:, expected_before:expected_after], axis=-1)  # Find Max

                # 5.) Get Evaluation Metrics
                # 5.1) Relative Starts of Test Set (relative to the expected_before)
                rel_start_test = (absolute_relative_starts[sel_label - 1][test_index] / 30) - expected_before
                # 5.2) Stereotype Start (relative to the expected_before)
                stereotype_start = np.mean(absolute_relative_starts[sel_label - 1][train_index] / 30,
                                           keepdims=True) - expected_before
                # 5.3) Stereotyped Prediction: (Stereotyped Start - Relative_Start)
                stereotyped_predictions = stereotype_start - rel_start_test
                # 5.4) Predicted Start: (Predicted Start - Relative Start)
                predicted_starts = onset_predictions - rel_start_test

                # Store the Results
                all_relative_starts.append(rel_start_test)
                all_stereotype_starts.append(stereotyped_predictions)
                all_predicted_starts.append(predicted_starts)  # (fold, Instances, Channels)
                all_elections.append(when_elections)
                # all_onset_predictions.append(onset_predictions)
                print('expected_before: ', expected_before)

            syll_all_relative_starts.append(all_relative_starts)
            syll_all_stereotype_starts.append(all_stereotype_starts)
            syll_all_predicted_starts.append(all_predicted_starts)  # (fold, Instances, Channels)
            syll_all_elections.append(all_elections)
            # syll_all_onset_predictions.append(all_onset_predictions)

        super_all_relative_starts.append(syll_all_relative_starts)
        super_all_stereotype_starts.append(syll_all_stereotype_starts)
        super_all_predicted_starts.append(syll_all_predicted_starts)
        super_all_elections.append(syll_all_elections)
        # super_all_onset_predictions.append(syll_all_onset_predictions)

    return super_all_relative_starts, super_all_stereotype_starts, super_all_predicted_starts, super_all_elections


def collaspse_folds(data):
    # (fold, Instances, Channels)
    holder2 = []
    for i in data:
        holder2.extend(i)  # (Instances, Channels)
    holder2 = np.asarray(holder2)
    return holder2


def convert_when_to_pandas(predicted_starts, stereotyped_starts, label_instruct, fc_lo, fc_hi):
    """ Convert the Results of naive_onset_detection into useful Pandas Format

    :param predicted_starts:
    :param stereotyped_starts:
    :param label_instruct:
    :param fc_lo:
    :param fc_hi:
    :return:
    """
    # [syllables]->[Freqs]->[folds]->[instances]

    results_pd = []

    for syll_num, syll in enumerate(predicted_starts):
        freq_holder = []
        for freq_num, freq in enumerate(syll):
            sel_results = collaspse_folds(freq)
            sel_results_df = pd.DataFrame(sel_results, columns=['times'])  # Make the DataFrame
            sel_results_df['Freq_band'] = str(fc_lo[freq_num]) + '-' + str(fc_hi[freq_num])
            freq_holder.append(sel_results_df)

        stereo_results = collaspse_folds(stereotyped_starts[syll_num][0])
        stereo_results_df = pd.DataFrame(stereo_results, columns=['times'])  # Make the DataFrame
        stereo_results_df['Freq_band'] = 'Stereotyped'
        freq_holder.append(stereo_results_df)

        syll_df = pd.concat(freq_holder, axis=0)
        syll_df['Syllable'] = label_instruct[syll_num]

        results_pd.append(syll_df)

    full_results = pd.concat(results_pd, axis=0)

    return full_results


def get_national_elections(elections):
    """Collapse Across Frequencies

    Parameters
    ----------
    elections: list | [syllables]->[Freqs]->[folds]->(instances, samples)
        The time-series Voting Results for each frequency for each syllable

    Returns
    -------
    national_election: list | [syllables]->[folds]->(instances, samples)
        The time-series of all Voting Results for each syllable
    """

    national_election = []

    for syll_num, syll in enumerate(elections):
        freq_holder = []
        for fold_num in range(5):
            fold_holder = np.zeros(np.shape(syll[0][fold_num]))
            for freq_num, freq in enumerate(syll):
                fold_holder = fold_holder + freq[fold_num]
            freq_holder.append(fold_holder)
        national_election.append(freq_holder)

    return national_election


def evaluate_national_elections(national_election, abs_rel_starts, relative_starts, label_instruct, time_buff):
    """ Find the Predicted Onset Times using the national elections (Collapse To just syllables by Instances)

    Parameters
    ----------
    national_election: list | [syllables]->[folds]->(instances, samples)
        The time-series of all Voting Results for each syllable
    abs_rel_starts: list | [syllables]->[folds]->(instances,)

    relative_starts: list | [syllables]->[folds]->(instances,)

    label_instruct: list

    time_buff: int


    Returns
    -------
    """
    # Collapse To just syllables by Instances
    # Input: [syllables]->[folds]->(instances, samples)
    # Goal: [syllables]->[folds]->(samples,)

    election_results = []

    for syll_num, (syll, sel_syll) in enumerate(zip(national_election, label_instruct)):

        sel_label = sel_syll - 1

        expected_time = int(np.mean(abs_rel_starts[sel_label - 1] / 30))
        expected_before = expected_time - time_buff
        expected_after = expected_time + time_buff

        syll_holder = []
        for fold_num, fold in enumerate(syll):
            results = np.argmax(fold[:, expected_before:expected_after], axis=-1)
            corr_results = results - relative_starts[syll_num][0][fold_num]
            syll_holder.append(corr_results)
        election_results.append(syll_holder)

    return election_results


def convert_national_elections_to_pandas(election_results, label_instruct):
    results_holder = []
    for syll_num, syll in enumerate(election_results):
        sel_results = collaspse_folds(syll)
        sel_results_df = pd.DataFrame(sel_results, columns=['times'])  # Make the DataFrame
        sel_results_df['Freq_band'] = "All"
        sel_results_df['Syllable'] = label_instruct[syll_num]

        results_holder.append(sel_results_df)
    full_results = pd.concat(results_holder, axis=0)

    return full_results


def _remove_unusable_syllable(all_best_chunk_events, motif_ledger, label_instruct):
    """
        all_best_chunk_events : list | [Freq]->[labels]->(Instances, 1, Channels, Samples)
    """
    # fix the all_best_chunk_events
    for freq_index, freq in enumerate(all_best_chunk_events):
        for label in label_instruct:
            need_to_drop = [index for index, value in enumerate(motif_ledger[label]) if value == 'bad']
            all_best_chunk_events[freq_index][label - 1] = np.delete(freq[label - 1], obj=need_to_drop, axis=0)


def _fix_the_ledger(motif_ledger):
    for key in motif_ledger.keys():
        motif_ledger[key] = [value for value in motif_ledger[key] if value != 'bad']


def run_naive_when_report(bird_id, session):
    # Import the Data
    zdata = ImportData(bird_id=bird_id, session=session)

    # Get the Bird Specific Machine Learning Meta Data
    bad_channels = all_bad_channels[bird_id]
    drop_temps = all_drop_temps[bird_id]
    when_label_instructions = all_when_label_instructions[bird_id]
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

    motif_length = motif_length_dict[bird_id]

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

    _remove_unusable_syllable(all_best_chunk_events=all_best_chunk_events_test, motif_ledger=motif_ledger,
                              label_instruct=when_label_instructions)  # Remove Bad Syllable Instances

    _fix_the_ledger(motif_ledger)  # New Function to remove 'bad' from the ledger

    relative_starts, stereotyped_starts, predicted_starts, elections = naive_onset_detection(
        absolute_relative_starts=absolute_relative_starts, motif_ledger=motif_ledger,
        all_best_chunk_events=all_best_chunk_events_test, motif_time_series=motif_time_series, time_buffer=time_buffer,
        label_instruct=when_label_instructions)

    full_results = convert_when_to_pandas(predicted_starts=predicted_starts, stereotyped_starts=stereotyped_starts,
                                          label_instruct=when_label_instructions, fc_lo=fc_lo, fc_hi=fc_hi)

    national_election = get_national_elections(elections=elections)

    election_results = evaluate_national_elections(national_election=national_election,
                                                   abs_rel_starts=absolute_relative_starts,
                                                   relative_starts=relative_starts,
                                                   label_instruct=when_label_instructions,
                                                   time_buff=time_buffer)
    all_results = convert_national_elections_to_pandas(election_results=election_results,
                                                       label_instruct=when_label_instructions)

    composite_results = pd.concat([full_results, all_results], axis=0)  # Create a Composite Pandas Data Structure

    ## Save the Componenets for the Summary Figure
    # Save the DataFrame of Onset Results
    save_pandas_to_pickle(data=composite_results, data_name="composite_results_run_2", bird_id=bird_id, session=session,
                          destination=onset_detection_path, make_parents=True, verbose=True)

    # Save the National Elections
    _save_pckl_data(data=national_election, data_name="national_election_run_2", bird_id=bird_id, session=session,
                    destination=onset_detection_path, make_parents=True, verbose=True)

    # Save the Motif Ledger
    _save_pckl_data(data=absolute_relative_starts, data_name="absolute_relative_starts_run_2", bird_id=bird_id,
                    session=session, destination=onset_detection_path, make_parents=True, verbose=True)

    # Save the Motif Ledger
    _save_pckl_data(data=election_results, data_name="election_results_run_2", bird_id=bird_id,
                    session=session, destination=onset_detection_path, make_parents=True, verbose=True)

    # Save the Motif Ledger
    _save_pckl_data(data=motif_ledger, data_name="motif_ledger_run_2", bird_id=bird_id,
                    session=session, destination=onset_detection_path, make_parents=True, verbose=True)

    # Save the Motif Ledger
    _save_pckl_data(data=stereotyped_starts, data_name="stereotyped_starts_run_2", bird_id=bird_id,
                    session=session, destination=onset_detection_path, make_parents=True, verbose=True)

    ## Make the Figures:
    # Figure 1: Make Figure of the Relative Onset Times

    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    sns.boxplot(x="Syllable", y="times", hue="Freq_band",
                hue_order=['Stereotyped', 'All', '4-8', '8-12', '25-35', '30-50', '50-70'],
                data=composite_results, linewidth=2.5, ax=ax1)  # hue="Freq_band"
    ax1.set_ylim(-50, 50)
    ax1.set_ylabel('Time Relative to True Onset (ms)')

    # Figure 2: Make Figure of the Confidences

    fig2, ax_2d = plt.subplots(len(when_label_instructions), 1, sharex=False, sharey=False, figsize=(9, 6), dpi=300)

    subsize = 8
    bigsize = 10
    ticksize = 8

    max_confidence = len(all_best_chunk_events_test[0][0][0][0, :, 0]) * 5

    for sub_election, syll_rel_starts, ax in zip(national_election, absolute_relative_starts, ax_2d):
        expected_time = int(np.mean(syll_rel_starts / 30))

        ax.plot(np.transpose(sub_election[0]) / max_confidence)
        ax.plot(np.transpose(sub_election[1]) / max_confidence)
        ax.plot(np.transpose(sub_election[2]) / max_confidence)
        ax.plot(np.transpose(sub_election[3]) / max_confidence)
        ax.axvline(x=expected_time)

        ax.set_xlabel(xlabel='Time (ms)', fontsize=subsize)
        ax.set_ylabel(ylabel='Pearson Sum', fontsize=subsize)
        ax.tick_params(axis='both', which='major', labelsize=ticksize)

        ax.set_xlim(0, motif_length)
        ax.set_ylim(0, 1)

    # Figure 3: Make Figure
    fig3, ax_3d = plt.subplots(len(when_label_instructions), 1, sharex=True, sharey=True, figsize=(9, 6), dpi=300)

    for predict_starts, stereo_starts, ax in zip(election_results, stereotyped_starts, ax_3d):
        fixed_predict_starts = collaspse_folds(predict_starts)
        fixed_stereo_starts = collaspse_folds(stereo_starts[0])
        ax.hist(np.asarray(fixed_predict_starts), bins=time_buffer * 2, range=(-time_buffer, time_buffer),
                label='prediction', alpha=.5)
        ax.hist(np.asarray(fixed_stereo_starts), bins=time_buffer * 2, range=(-time_buffer, time_buffer),
                label='stereotyped', alpha=.5)

        ax.set_xlabel(xlabel='Time (ms)', fontsize=subsize)
        ax.set_ylabel(ylabel='Count', fontsize=subsize)
        ax.tick_params(axis='both', which='major', labelsize=ticksize)
        ax.set_xlim(-time_buffer, time_buffer)

        # Adde Useful Annotations
        pred_summary1 = "Predicted Mean : " + str(np.mean(fixed_predict_starts))
        pred_summary2 = "Predicted STD: " + str(np.std(fixed_predict_starts, ddof=1))
        stereo_summary1 = "Stereotyped Mean : " + str(np.mean(fixed_stereo_starts))
        stereo_summary2 = "Stereotyped STD: " + str(np.std(fixed_stereo_starts, ddof=1))
        plt.text(0.05, 0.90, pred_summary1, fontsize=subsize, transform=ax.transAxes)
        plt.text(0.05, 0.80, pred_summary2, fontsize=subsize, transform=ax.transAxes)
        plt.text(0.05, 0.60, stereo_summary1, fontsize=subsize, transform=ax.transAxes)
        plt.text(0.05, 0.50, stereo_summary2, fontsize=subsize, transform=ax.transAxes)

        ax.legend(fontsize=subsize)

    report_name = 'Onset_Prediction_-aive_' + bird_id + '_' + session + '_' + '_report_run_2.pdf'
    report_type_folder = REPORTS_DIR / 'Onset_Prediction_Naive'

    # Check if Folder Path Exists
    if not report_type_folder.exists():
        report_type_folder.mkdir(parents=True, exist_ok=True)

    report_location = report_type_folder / report_name

    # Create the PdfPages object to which we will save the pages:
    # The with statement makes sure that the PdfPages object is closed properly at
    # the end of the block, even if an Exception occurs.
    with PdfPages(report_location) as pdf:
        pdf.attach_note(f"Maximum Confidence Possible = {max_confidence}", positionRect=[-100, -100, 0, 0])
        pdf.savefig(fig1)
        pdf.savefig(fig2)
        pdf.savefig(fig3)
