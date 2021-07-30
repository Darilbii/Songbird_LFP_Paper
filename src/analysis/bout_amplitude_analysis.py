import numpy as np

import BirdSongToolbox.free_epoch_tools as fet
from BirdSongToolbox.import_data import ImportData
from BirdSongToolbox.file_utility_functions import _save_numpy_data, _load_numpy_data

import src.analysis.hilbert_based_pipeline as hbp
# from src.analysis.chunk_when_analysis_naive import save_pandas_to_pickle
from src.analysis.context_utility import birds_context_obj, all_last_syllable
from src.analysis.ml_pipeline_utilities import all_bad_channels
from BirdSongToolbox.context_hand_labeling import label_focus_context, first_context_func, last_context_func

bout_amplitude_analysis_path = '/home/debrown/bout_amplitude_analysis'


def get_bout_amplitude_analysis_results(bird_id='z007', session='day-2016-09-09'):
    # Save the Before: Low Freq
    bout_amplitude_analysis_before_low =_load_numpy_data(data_name="bout_amplitude_analysis_before_low",
                     bird_id=bird_id, session=session, source=bout_amplitude_analysis_path, verbose=True)
    # Save the Before: High Freq
    bout_amplitude_analysis_before_high = _load_numpy_data(data_name="bout_amplitude_analysis_before_high",
                     bird_id=bird_id, session=session, source=bout_amplitude_analysis_path, verbose=True)
    # Save the After: High Freq
    bout_amplitude_analysis_after_high = _load_numpy_data(data_name="bout_amplitude_analysis_after_high",
                     bird_id=bird_id, session=session, source=bout_amplitude_analysis_path, verbose=True)
    # Save the After: Low Freq
    bout_amplitude_analysis_after_low = _load_numpy_data(data_name="bout_amplitude_analysis_after_low", bird_id=bird_id,
                     session=session, source=bout_amplitude_analysis_path, verbose=True)

    return bout_amplitude_analysis_before_low, bout_amplitude_analysis_before_high, bout_amplitude_analysis_after_high,\
           bout_amplitude_analysis_after_low

def run_bout_amplitude_analysis(bird_id='z007', session='day-2016-09-09'):

    zdata = ImportData(bird_id=bird_id, session=session)

    # Format Hanlabels for use

    chunk_labels_list, chunk_onsets_list = fet.get_chunk_handlabels(handlabels_list=zdata.song_handlabels)

    # Switch to the Log Spaced Bins
    freq_bins = 100
    fc_lo = np.logspace(np.log10(2), np.log10(220), freq_bins)
    fc_hi = np.logspace(np.log10(3), np.log10(250), freq_bins)

    # Get Channels to Exclude from CAR
    bad_channels = all_bad_channels[bird_id]

    proc_data2 = hbp.feature_extraction_chunk(neural_chunks=zdata.song_neural,
                                              fs=1000,
                                              l_freqs=fc_lo,
                                              h_freqs=fc_hi,
                                              hilbert="amplitude",
                                              z_score=True,
                                              mv_avg=100,
                                              bad_channels=bad_channels,
                                              verbose=True)

    # Get Starts of Events of Interest (Syllables)
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

    # Set the Context Windows
    first_window = (-100, 200)
    last_window = (-100, 200)

    # Clip around Events of Interest
    all_firsts = fet.get_event_related_nd_chunk(chunk_data=proc_data2, chunk_indices=first_syll,
                                                fs=1000, window=first_window)

    all_lasts = fet.get_event_related_nd_chunk(chunk_data=proc_data2, chunk_indices=last_syll,
                                               fs=1000, window=last_window)

    # Correct The Shape of the Data
    all_firsts = fet.event_shape_correction(all_firsts, original_dim=3)
    all_lasts = fet.event_shape_correction(all_lasts, original_dim=3)

    # Make them ndarray indexable
    all_firsts = np.asarray(all_firsts)
    all_lasts = np.asarray(all_lasts)

    # Before Bout Amplitude Change
    diff_amplitude_before = all_firsts[:, :, :, 100] - all_firsts[:, :, :, 200]

    diff_amplitude_before = np.delete(diff_amplitude_before, bad_channels, axis=2)   # Remove Bad Channels

    ## Break Between Above and Below 50 Hz
    diff_amplitude_before_low_freq = diff_amplitude_before[:, :65, :]
    diff_amplitude_before_high_freq = diff_amplitude_before[:, 65:, :]

    # Reshape Arrays (Instances, Features)
    diff_amplitude_before_low_freq = np.reshape(diff_amplitude_before_low_freq, (len(diff_amplitude_before), -1))
    diff_amplitude_before_high_freq = np.reshape(diff_amplitude_before_high_freq, (len(diff_amplitude_before), -1))


    # After Bout Amplitude Change
    diff_amplitude_after = all_lasts[:, :, :, 100] - all_lasts[:, :, :, 200]

    diff_amplitude_after = np.delete(diff_amplitude_after, bad_channels, axis=2)  # Remove Bad Channels

    diff_amplitude_after_high_freq = diff_amplitude_after[:, :65, :]
    diff_amplitude_after_low_freq = diff_amplitude_after[:, 65:, :]

    # Reshape Arrays (Instances, Features)
    diff_amplitude_after_high_freq = np.reshape(diff_amplitude_after_high_freq, (len(diff_amplitude_after), -1))
    diff_amplitude_after_low_freq = np.reshape(diff_amplitude_after_low_freq, (len(diff_amplitude_after), -1))

    # Save the Before: Low Freq
    _save_numpy_data(data=diff_amplitude_before_low_freq, data_name="bout_amplitude_analysis_before_low",
                     bird_id=bird_id, session=session, destination=bout_amplitude_analysis_path, make_parents=True,
                     verbose=True)
    # Save the Before: High Freq
    _save_numpy_data(data=diff_amplitude_before_high_freq, data_name="bout_amplitude_analysis_before_high",
                     bird_id=bird_id, session=session, destination=bout_amplitude_analysis_path, make_parents=True,
                     verbose=True)
    # Save the After: High Freq
    _save_numpy_data(data=diff_amplitude_after_high_freq, data_name="bout_amplitude_analysis_after_high",
                     bird_id=bird_id, session=session, destination=bout_amplitude_analysis_path, make_parents=True,
                     verbose=True)
    # Save the After: Low Freq
    _save_numpy_data(data=diff_amplitude_after_low_freq, data_name="bout_amplitude_analysis_after_low", bird_id=bird_id,
                     session=session, destination=bout_amplitude_analysis_path, make_parents=True, verbose=True)


def get_bout_amplitude_analysis_results2(bird_id='z007', session='day-2016-09-09'):
    # Save the Before: Low Freq
    bout_amplitude_analysis_before_low =_load_numpy_data(data_name="bout_amplitude_analysis_before_low2",
                     bird_id=bird_id, session=session, source=bout_amplitude_analysis_path, verbose=True)
    # Save the Before: High Freq
    bout_amplitude_analysis_before_high = _load_numpy_data(data_name="bout_amplitude_analysis_before_high2",
                     bird_id=bird_id, session=session, source=bout_amplitude_analysis_path, verbose=True)
    # Save the After: High Freq
    bout_amplitude_analysis_after_high = _load_numpy_data(data_name="bout_amplitude_analysis_after_high2",
                     bird_id=bird_id, session=session, source=bout_amplitude_analysis_path, verbose=True)
    # Save the After: Low Freq
    bout_amplitude_analysis_after_low = _load_numpy_data(data_name="bout_amplitude_analysis_after_low2", bird_id=bird_id,
                     session=session, source=bout_amplitude_analysis_path, verbose=True)

    return bout_amplitude_analysis_before_low, bout_amplitude_analysis_before_high, bout_amplitude_analysis_after_high,\
           bout_amplitude_analysis_after_low


def run_bout_amplitude_analysis2(bird_id='z007', session='day-2016-09-09'):

    zdata = ImportData(bird_id=bird_id, session=session)

    # Format Hanlabels for use

    chunk_labels_list, chunk_onsets_list = fet.get_chunk_handlabels(handlabels_list=zdata.song_handlabels)

    # Switch to the Log Spaced Bins
    freq_bins = 100
    fc_lo = np.logspace(np.log10(2), np.log10(220), freq_bins)
    fc_hi = np.logspace(np.log10(3), np.log10(250), freq_bins)

    # Get Channels to Exclude from CAR
    bad_channels = all_bad_channels[bird_id]

    proc_data2 = hbp.feature_extraction_chunk(neural_chunks=zdata.song_neural,
                                              fs=1000,
                                              l_freqs=fc_lo,
                                              h_freqs=fc_hi,
                                              hilbert="amplitude",
                                              z_score=True,
                                              mv_avg=50,
                                              bad_channels=bad_channels,
                                              verbose=True)

    # Get Starts of Events of Interest (Syllables)
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

    # Set the Context Windows
    first_window = (-50, 100)
    last_window = (-50, 100)

    # Clip around Events of Interest
    all_firsts = fet.get_event_related_nd_chunk(chunk_data=proc_data2, chunk_indices=first_syll,
                                                fs=1000, window=first_window)

    all_lasts = fet.get_event_related_nd_chunk(chunk_data=proc_data2, chunk_indices=last_syll,
                                               fs=1000, window=last_window)

    # Correct The Shape of the Data
    all_firsts = fet.event_shape_correction(all_firsts, original_dim=3)
    all_lasts = fet.event_shape_correction(all_lasts, original_dim=3)

    # Make them ndarray indexable
    all_firsts = np.asarray(all_firsts)
    all_lasts = np.asarray(all_lasts)

    # Before Bout Amplitude Change
    diff_amplitude_before = all_firsts[:, :, :, 50] - all_firsts[:, :, :, 100]

    diff_amplitude_before = np.delete(diff_amplitude_before, bad_channels, axis=2)   # Remove Bad Channels

    ## Break Between Above and Below 50 Hz
    diff_amplitude_before_low_freq = diff_amplitude_before[:, :65, :]
    diff_amplitude_before_high_freq = diff_amplitude_before[:, 65:, :]

    # Reshape Arrays (Instances, Features)
    diff_amplitude_before_low_freq = np.reshape(diff_amplitude_before_low_freq, (len(diff_amplitude_before), -1))
    diff_amplitude_before_high_freq = np.reshape(diff_amplitude_before_high_freq, (len(diff_amplitude_before), -1))


    # After Bout Amplitude Change
    diff_amplitude_after = all_lasts[:, :, :, 50] - all_lasts[:, :, :, 100]

    diff_amplitude_after = np.delete(diff_amplitude_after, bad_channels, axis=2)  # Remove Bad Channels

    diff_amplitude_after_high_freq = diff_amplitude_after[:, :65, :]
    diff_amplitude_after_low_freq = diff_amplitude_after[:, 65:, :]

    # Reshape Arrays (Instances, Features)
    diff_amplitude_after_high_freq = np.reshape(diff_amplitude_after_high_freq, (len(diff_amplitude_after), -1))
    diff_amplitude_after_low_freq = np.reshape(diff_amplitude_after_low_freq, (len(diff_amplitude_after), -1))

    # Save the Before: Low Freq
    _save_numpy_data(data=diff_amplitude_before_low_freq, data_name="bout_amplitude_analysis_before_low2",
                     bird_id=bird_id, session=session, destination=bout_amplitude_analysis_path, make_parents=True,
                     verbose=True)
    # Save the Before: High Freq
    _save_numpy_data(data=diff_amplitude_before_high_freq, data_name="bout_amplitude_analysis_before_high2",
                     bird_id=bird_id, session=session, destination=bout_amplitude_analysis_path, make_parents=True,
                     verbose=True)
    # Save the After: High Freq
    _save_numpy_data(data=diff_amplitude_after_high_freq, data_name="bout_amplitude_analysis_after_high2",
                     bird_id=bird_id, session=session, destination=bout_amplitude_analysis_path, make_parents=True,
                     verbose=True)
    # Save the After: Low Freq
    _save_numpy_data(data=diff_amplitude_after_low_freq, data_name="bout_amplitude_analysis_after_low2", bird_id=bird_id,
                     session=session, destination=bout_amplitude_analysis_path, make_parents=True, verbose=True)





