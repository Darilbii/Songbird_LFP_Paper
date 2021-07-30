# After preprocessing all of the data
# For each Chunk
# Make an array of the Behavioral data
# Make a Mask based on the Behavioral Data
# Select the Data based off of that mask
# compile all of the Behavioral Data
################### The Above Approach was Tabled for later ########################


import numpy as np
from statsmodels.stats import weightstats as stests
import pingouin as pg
import pandas as pd


from BirdSongToolbox.import_data import ImportData
from BirdSongToolbox.file_utility_functions import _save_numpy_data, _load_numpy_data
import BirdSongToolbox.free_epoch_tools as fet
from BirdSongToolbox.context_hand_labeling import label_focus_context, first_context_func, last_context_func, \
    mid_context_func

from src.analysis.context_utility import birds_context_obj, all_last_syllable
import src.analysis.hilbert_based_pipeline as hbp
from src.analysis.context_utility import birds_context_obj
from src.analysis.ml_pipeline_utilities import all_bad_channels, all_drop_temps, all_label_instructions

power_distribution_path = '/home/debrown/songbird_power_distribution'


def get_power_distribution_results(bird_id='z007', session='day-2016-09-09'):
    # Save the Before: Low Freq
    all_results_increase = _load_numpy_data(data_name="power_distribution_increase",
                                            bird_id=bird_id, session=session, source=power_distribution_path,
                                            verbose=True)
    # Load the Decrease Results
    all_results_decrease = _load_numpy_data(data_name="power_distribution_decrease",
                                            bird_id=bird_id, session=session, source=power_distribution_path,
                                            verbose=True)

    return all_results_increase, all_results_decrease


def run_power_distribution_test(bird_id='z007', session='day-2016-09-09'):
    zdata = ImportData(bird_id=bird_id, session=session)

    # Format Hanlabels for use
    chunk_labels_list, chunk_onsets_list = fet.get_chunk_handlabels(handlabels_list=zdata.song_handlabels)

    fc_lo = [4, 8, 25, 30, 50, 50, 80, ]
    fc_hi = [8, 12, 35, 50, 70, 200, 200]

    # Get Channels to Exclude from CAR
    bad_channels = all_bad_channels[bird_id]

    # Preprocess the Date to Get Amplitude Values

    proc_data = hbp.feature_extraction_chunk(neural_chunks=zdata.song_neural,
                                             fs=1000,
                                             l_freqs=fc_lo,
                                             h_freqs=fc_hi,
                                             hilbert="amplitude",
                                             z_score=True,
                                             bad_channels=bad_channels,
                                             drop_bad=True,
                                             verbose=True)

    # Create instance of the Context Labels Class
    testclass = birds_context_obj(bird_id=bird_id)

    # Get the Context Array for the Day’s Data
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

    # Get Silence Periods
    silent_periods = fet.long_silence_finder(silence=8,
                                             all_labels=chunk_labels_list,
                                             all_starts=chunk_onsets_list[0],
                                             all_ends=chunk_onsets_list[1],
                                             window=(-500, 500))

    # Set the Context Windows

    first_window = (0, 500)
    last_window = (-500, 0)
    silence_window = (-300, 300)

    # Clip around Events of Interest
    all_firsts = fet.get_event_related_nd_chunk(chunk_data=proc_data, chunk_indices=first_syll,
                                                fs=1000, window=first_window)

    all_lasts = fet.get_event_related_nd_chunk(chunk_data=proc_data, chunk_indices=last_syll,
                                               fs=1000, window=last_window)

    all_silences = fet.get_event_related_nd_chunk(chunk_data=proc_data, chunk_indices=silent_periods,
                                                  fs=1000, window=silence_window)

    # Correct The Shape of the Data

    all_firsts = fet.event_shape_correction(all_firsts, original_dim=3)
    all_lasts = fet.event_shape_correction(all_lasts, original_dim=3)
    all_silences = fet.event_shape_correction(all_silences, original_dim=3)

    # Make them ndarray indexable

    all_firsts = np.asarray(all_firsts)
    all_lasts = np.asarray(all_lasts)
    all_silences = np.asarray(all_silences)

    # Collapse First and Last
    all_active = np.concatenate([all_firsts, all_lasts], axis=0)

    # Reshape to (freq, channels, instances, samples)
    all_active_samples = np.transpose(all_active, axes=[1, 2, 0, 3])
    all_inactive_samples = np.transpose(all_silences, axes=[1, 2, 0, 3])

    # Reshape the Arrays to make it easier to get a distrubution

    all_active_collapsed = np.reshape(all_active_samples,
                                      (all_active_samples.shape[0], all_active_samples.shape[1], -1))

    all_inactive_collapsed = np.reshape(all_inactive_samples,
                                        (all_inactive_samples.shape[0], all_inactive_samples.shape[1], -1))

    # Runt the Z-Test (Greater)
    all_results_increase = []
    for freq_active_collapsed, freq_inactive_collapsed in zip(all_active_collapsed, all_inactive_collapsed):
        chan_holder = []
        for chan_active, chan_inactive in zip(freq_active_collapsed, freq_inactive_collapsed):
            sel_active = np.random.choice(chan_active, size=5000, replace=True)
            sel_inactive = np.random.choice(chan_inactive, size=5000, replace=True)
            ztest, pval = stests.ztest(x1=sel_active, x2=sel_inactive, value=0, alternative='larger')
            chan_holder.append(pval)
        all_results_increase.append(chan_holder)

    # Runt the Z-Test (Lesser)
    all_results_decrease = []
    for freq_active_collapsed, freq_inactive_collapsed in zip(all_active_collapsed, all_inactive_collapsed):
        chan_holder = []
        for chan_active, chan_inactive in zip(freq_active_collapsed, freq_inactive_collapsed):
            sel_active = np.random.choice(chan_active, size=5000, replace=True)
            sel_inactive = np.random.choice(chan_inactive, size=5000, replace=True)
            ztest, pval = stests.ztest(x1=sel_active, x2=sel_inactive, value=0, alternative='smaller')
            chan_holder.append(pval)
        all_results_decrease.append(chan_holder)

    # Convert to a ndarray

    all_results_increase = np.asarray(all_results_increase)
    all_results_decrease = np.asarray(all_results_decrease)

    # Save the Before: Low Freq
    _save_numpy_data(data=all_results_increase, data_name="power_distribution_increase",
                     bird_id=bird_id, session=session, destination=power_distribution_path, make_parents=True,
                     verbose=True)
    # Save the Before: High Freq
    _save_numpy_data(data=all_results_decrease, data_name="power_distribution_decrease",
                     bird_id=bird_id, session=session, destination=power_distribution_path, make_parents=True,
                     verbose=True)


def get_power_distribution_results2(bird_id='z007', session='day-2016-09-09'):
    # Save the Before: Low Freq
    all_results_increase = _load_numpy_data(data_name="power_distribution_increase2",
                                            bird_id=bird_id, session=session, source=power_distribution_path,
                                            verbose=True)
    # Load the Decrease Results
    all_results_decrease = _load_numpy_data(data_name="power_distribution_decrease2",
                                            bird_id=bird_id, session=session, source=power_distribution_path,
                                            verbose=True)

    return all_results_increase, all_results_decrease


def run_power_distribution_test2(bird_id='z007', session='day-2016-09-09'):
    zdata = ImportData(bird_id=bird_id, session=session)

    # Format Hanlabels for use
    chunk_labels_list, chunk_onsets_list = fet.get_chunk_handlabels(handlabels_list=zdata.song_handlabels)

    fc_lo = [4, 8, 25, 30, 50, 50, 80, ]
    fc_hi = [8, 12, 35, 50, 70, 200, 200]

    # Get Channels to Exclude from CAR
    bad_channels = all_bad_channels[bird_id]

    # Preprocess the Date to Get Amplitude Values

    proc_data = hbp.feature_extraction_chunk(neural_chunks=zdata.song_neural,
                                             fs=1000,
                                             l_freqs=fc_lo,
                                             h_freqs=fc_hi,
                                             hilbert="amplitude",
                                             z_score=True,
                                             bad_channels=bad_channels,
                                             drop_bad=True,
                                             verbose=True)

    # Create instance of the Context Labels Class
    testclass = birds_context_obj(bird_id=bird_id)

    # Get the Context Array for the Day’s Data
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

    # Get Silence Periods
    silent_periods = fet.long_silence_finder(silence=8,
                                             all_labels=chunk_labels_list,
                                             all_starts=chunk_onsets_list[0],
                                             all_ends=chunk_onsets_list[1],
                                             window=(-500, 500))

    # Set the Context Windows

    first_window = (0, 500)
    last_window = (-500, 0)
    silence_window = (-300, 300)

    # Clip around Events of Interest
    all_firsts = fet.get_event_related_nd_chunk(chunk_data=proc_data, chunk_indices=first_syll,
                                                fs=1000, window=first_window)

    all_lasts = fet.get_event_related_nd_chunk(chunk_data=proc_data, chunk_indices=last_syll,
                                               fs=1000, window=last_window)

    all_silences = fet.get_event_related_nd_chunk(chunk_data=proc_data, chunk_indices=silent_periods,
                                                  fs=1000, window=silence_window)

    # Correct The Shape of the Data

    all_firsts = fet.event_shape_correction(all_firsts, original_dim=3)
    all_lasts = fet.event_shape_correction(all_lasts, original_dim=3)
    all_silences = fet.event_shape_correction(all_silences, original_dim=3)

    # Make them ndarray indexable

    all_firsts = np.asarray(all_firsts)
    all_lasts = np.asarray(all_lasts)
    all_silences = np.asarray(all_silences)

    # Collapse First and Last
    all_active = np.concatenate([all_firsts, all_lasts], axis=0)

    # Reshape to (freq, channels, instances, samples)
    all_active_samples = np.transpose(all_active, axes=[1, 2, 0, 3])
    all_inactive_samples = np.transpose(all_silences, axes=[1, 2, 0, 3])

    # Reshape the Arrays to make it easier to get a distrubution

    all_active_collapsed = np.reshape(all_active_samples,
                                      (all_active_samples.shape[0], all_active_samples.shape[1], -1))

    all_inactive_collapsed = np.reshape(all_inactive_samples,
                                        (all_inactive_samples.shape[0], all_inactive_samples.shape[1], -1))

    # Runt the Z-Test (Greater)
    all_results_increase = []
    for freq_active_collapsed, freq_inactive_collapsed in zip(all_active_collapsed, all_inactive_collapsed):
        chan_holder = []
        for chan_active, chan_inactive in zip(freq_active_collapsed, freq_inactive_collapsed):
            sel_active = chan_active
            sel_inactive = chan_inactive
            ztest, pval = stests.ztest(x1=sel_active, x2=sel_inactive, value=0, alternative='larger')
            chan_holder.append(pval)
        all_results_increase.append(chan_holder)

    # Runt the Z-Test (Lesser)
    all_results_decrease = []
    for freq_active_collapsed, freq_inactive_collapsed in zip(all_active_collapsed, all_inactive_collapsed):
        chan_holder = []
        for chan_active, chan_inactive in zip(freq_active_collapsed, freq_inactive_collapsed):
            sel_active = chan_active
            sel_inactive = chan_inactive
            ztest, pval = stests.ztest(x1=sel_active, x2=sel_inactive, value=0, alternative='smaller')
            chan_holder.append(pval)
        all_results_decrease.append(chan_holder)

    # Convert to a ndarray

    all_results_increase = np.asarray(all_results_increase)
    all_results_decrease = np.asarray(all_results_decrease)

    # Save the Before: Low Freq
    _save_numpy_data(data=all_results_increase, data_name="power_distribution_increase2",
                     bird_id=bird_id, session=session, destination=power_distribution_path, make_parents=True,
                     verbose=True)
    # Save the Before: High Freq
    _save_numpy_data(data=all_results_decrease, data_name="power_distribution_decrease2",
                     bird_id=bird_id, session=session, destination=power_distribution_path, make_parents=True,
                     verbose=True)


def make_dataframe_for_excel(fc_lo, fc_hi, all_results_p, all_results_z, all_results_d):
    """ Helper function to convert results into a pandas dataframe that can easily export to a excel sheet"""
    results_holder = []
    for freq_low, freq_high, freq_results_p, freq_results_z, freq_results_d in zip(fc_lo, fc_hi, all_results_p,
                                                                                   all_results_z,
                                                                                   all_results_d):
        freq_label = str(freq_low) + "-" + str(freq_high)
        num_channels = len(freq_results_p)
        channel_labels = np.arange(1, num_channels + 1)
        data_holder = {"channel #": channel_labels, "Test Statistic": freq_results_z, "p-values": freq_results_p,
                       "Effect Size": freq_results_d}
        results_df = pd.DataFrame(data=data_holder)  # Make the DataFrame
        results_df["Frequency Band"] = freq_label

        results_holder.append(results_df)
    full_results = pd.concat(results_holder, axis=0)

    return full_results


def make_table_for_power_distribution_test2(bird_id='z007', session='day-2016-09-09'):
    zdata = ImportData(bird_id=bird_id, session=session)

    # Format Handlabels for use
    chunk_labels_list, chunk_onsets_list = fet.get_chunk_handlabels(handlabels_list=zdata.song_handlabels)

    fc_lo = [4, 8, 25, 30, 50, 80]
    fc_hi = [8, 12, 35, 50, 70, 200]

    # Get Channels to Exclude from CAR
    bad_channels = all_bad_channels[bird_id]

    # Preprocess the Date to Get Amplitude Values

    proc_data = hbp.feature_extraction_chunk(neural_chunks=zdata.song_neural,
                                             fs=1000,
                                             l_freqs=fc_lo,
                                             h_freqs=fc_hi,
                                             hilbert="amplitude",
                                             z_score=True,
                                             bad_channels=bad_channels,
                                             drop_bad=True,
                                             verbose=True)

    # Create instance of the Context Labels Class
    testclass = birds_context_obj(bird_id=bird_id)

    # Get the Context Array for the Day’s Data
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

    # Get Silence Periods
    silent_periods = fet.long_silence_finder(silence=8,
                                             all_labels=chunk_labels_list,
                                             all_starts=chunk_onsets_list[0],
                                             all_ends=chunk_onsets_list[1],
                                             window=(-500, 500))

    # Set the Context Windows

    first_window = (0, 500)
    last_window = (-500, 0)
    silence_window = (-300, 300)

    # Clip around Events of Interest
    all_firsts = fet.get_event_related_nd_chunk(chunk_data=proc_data, chunk_indices=first_syll,
                                                fs=1000, window=first_window)

    all_lasts = fet.get_event_related_nd_chunk(chunk_data=proc_data, chunk_indices=last_syll,
                                               fs=1000, window=last_window)

    all_silences = fet.get_event_related_nd_chunk(chunk_data=proc_data, chunk_indices=silent_periods,
                                                  fs=1000, window=silence_window)

    # Correct The Shape of the Data

    all_firsts = fet.event_shape_correction(all_firsts, original_dim=3)
    all_lasts = fet.event_shape_correction(all_lasts, original_dim=3)
    all_silences = fet.event_shape_correction(all_silences, original_dim=3)

    # Make them ndarray indexable

    all_firsts = np.asarray(all_firsts)
    all_lasts = np.asarray(all_lasts)
    all_silences = np.asarray(all_silences)

    # Collapse First and Last
    all_active = np.concatenate([all_firsts, all_lasts], axis=0)

    # Reshape to (freq, channels, instances, samples)
    all_active_samples = np.transpose(all_active, axes=[1, 2, 0, 3])
    all_inactive_samples = np.transpose(all_silences, axes=[1, 2, 0, 3])

    # Reshape the Arrays to make it easier to get a distrubution

    all_active_collapsed = np.reshape(all_active_samples,
                                      (all_active_samples.shape[0], all_active_samples.shape[1], -1))

    all_inactive_collapsed = np.reshape(all_inactive_samples,
                                        (all_inactive_samples.shape[0], all_inactive_samples.shape[1], -1))

    # Runt the Z-Test (Greater)
    all_results_increase_p = []
    all_results_increase_z = []
    all_results_increase_d = []  # cohen d effect size

    for freq_active_collapsed, freq_inactive_collapsed in zip(all_active_collapsed, all_inactive_collapsed):
        chan_holder_p = []
        chan_holder_z = []
        chan_holder_d = []
        for chan_active, chan_inactive in zip(freq_active_collapsed, freq_inactive_collapsed):
            sel_active = chan_active
            sel_inactive = chan_inactive
            ztest, pval = stests.ztest(x1=sel_active, x2=sel_inactive, value=0, alternative='larger')
            cohen_d = pg.compute_effsize(sel_active, sel_inactive, eftype='cohen')
            chan_holder_p.append(pval)
            chan_holder_z.append(ztest)
            chan_holder_d.append(cohen_d)
        all_results_increase_p.append(chan_holder_p)
        all_results_increase_z.append(chan_holder_z)
        all_results_increase_d.append(chan_holder_d)

    # Runt the Z-Test (Lesser)
    all_results_decrease_p = []
    all_results_decrease_z = []
    all_results_decrease_d = []
    for freq_active_collapsed, freq_inactive_collapsed in zip(all_active_collapsed, all_inactive_collapsed):
        chan_holder_p = []
        chan_holder_z = []
        chan_holder_d = []
        for chan_active, chan_inactive in zip(freq_active_collapsed, freq_inactive_collapsed):
            sel_active = chan_active
            sel_inactive = chan_inactive
            ztest, pval = stests.ztest(x1=sel_active, x2=sel_inactive, value=0, alternative='smaller')
            cohen_d = pg.compute_effsize(sel_active, sel_inactive, eftype='cohen')
            chan_holder_p.append(pval)
            chan_holder_z.append(ztest)
            chan_holder_d.append(cohen_d)
        all_results_decrease_p.append(chan_holder_p)
        all_results_decrease_z.append(chan_holder_z)
        all_results_decrease_d.append(chan_holder_d)

    # Convert to a ndarray

    all_results_increase_p = np.asarray(all_results_increase_p)
    all_results_decrease_p = np.asarray(all_results_decrease_p)
    all_results_increase_z = np.asarray(all_results_increase_z)
    all_results_decrease_z = np.asarray(all_results_decrease_z)
    all_results_increase_d = np.asarray(all_results_increase_d)
    all_results_decrease_d = np.asarray(all_results_decrease_d)

    # # Save the Before: Low Freq
    # _save_numpy_data(data=all_results_increase_p, data_name="power_distribution_increase2",
    #                  bird_id=bird_id, session=session, destination=power_distribution_path, make_parents=True,
    #                  verbose=True)
    # # Save the Before: High Freq
    # _save_numpy_data(data=all_results_decrease_p, data_name="power_distribution_decrease2",
    #                  bird_id=bird_id, session=session, destination=power_distribution_path, make_parents=True,
    #                  verbose=True)

    increase_results_pd = make_dataframe_for_excel(fc_lo=fc_lo, fc_hi=fc_hi, all_results_p=all_results_increase_p,
                                                   all_results_z=all_results_increase_z,
                                                   all_results_d=all_results_increase_d)


    decrease_results_pd = make_dataframe_for_excel(fc_lo=fc_lo, fc_hi=fc_hi, all_results_p=all_results_decrease_p,
                                                   all_results_z=all_results_decrease_z,
                                                   all_results_d=all_results_decrease_d)

    # storing into the excel file
    increase_output_name = "increase_results_" + str(bird_id) + "_" + str(session) + ".xlsx"
    increase_results_pd.to_excel(increase_output_name)

    decrease_output_name = "decrease_results_" + str(bird_id) + "_" + str(session) + ".xlsx"
    decrease_results_pd.to_excel(decrease_output_name)
