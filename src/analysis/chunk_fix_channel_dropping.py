import BirdSongToolbox.chunk_analysis_tools as cat
import numpy as np
import scipy

from src.analysis.ml_pipeline_utilities import all_bad_channels, all_drop_temps, all_label_instructions

# Functions added for the Report
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

import BirdSongToolbox.free_epoch_tools as fet
from BirdSongToolbox.import_data import ImportData
from BirdSongToolbox.file_utility_functions import _save_numpy_data, _load_numpy_data

import src.analysis.ml_pipeline_utilities as mlpu
from src.analysis.chunk_parameter_sweep_bin_offset import get_priors
import src.analysis.hilbert_based_pipeline as hbp
from src.analysis.chunk_feature_dropping_pearson import best_bin_width, best_offset

import warnings

channel_drop_path = '/home/debrown/channel_dropping_results'

def get_feature_dropping_corrections_repeats(bird_id='z007', session='day-2016-09-09', feat_type: str = 'pow', verbose=True):
    """ Import the results of make_parameter_sweep

    :param bird_id:
    :param session:
    :param verbose:
    :return:

    accuracy : ndarray, (bin_widths, offsets, num_folds, frequencies)
        ndarray of the k-fold accuracies

    confusions : ndarray, (bin_widths, offsets, num_folds, frequencies)
        ndarray of the k-fold confusion matrices
    """
    assert feat_type in ['pow', 'phase', 'both'], "invalid feat_type"

    mean_correction_name = "accuracy_repeat_" + feat_type
    mean_correction = _load_numpy_data(data_name=mean_correction_name, bird_id=bird_id, session=session,
                                       source=channel_drop_path, verbose=verbose)

    std_correction_name = "std_all_repeat_" + feat_type
    std_correction = _load_numpy_data(data_name=std_correction_name, bird_id=bird_id, session=session,
                                      source=channel_drop_path, verbose=verbose)

    return mean_correction, std_correction


def get_feature_dropping_corrections(bird_id='z007', session='day-2016-09-09', feat_type: str = 'pow', verbose=True):
    """ Import the results of make_parameter_sweep

    :param bird_id:
    :param session:
    :param verbose:
    :return:

    accuracy : ndarray, (bin_widths, offsets, num_folds, frequencies)
        ndarray of the k-fold accuracies

    confusions : ndarray, (bin_widths, offsets, num_folds, frequencies)
        ndarray of the k-fold confusion matrices
    """
    assert feat_type in ['pow', 'phase', 'both'], "invalid feat_type"

    mean_correction_name = "accuracy_" + feat_type
    mean_correction = _load_numpy_data(data_name=mean_correction_name, bird_id=bird_id, session=session,
                                       source=channel_drop_path, verbose=verbose)

    std_correction_name = "std_all_" + feat_type
    std_correction = _load_numpy_data(data_name=std_correction_name, bird_id=bird_id, session=session,
                                      source=channel_drop_path, verbose=verbose)

    return mean_correction, std_correction


def single_frequency_cross_valid_accuracy_chunk(event_data, ClassObj, drop_temps, sel_freq, k_folds=5, seed=None,
                                                verbose=False):
    """   K-Fold Validated Accuracy for each Frequency Band Seperately

    Parameters
    ----------
    event_data : ndarray | (classes, instances, frequencies, channels, samples)
        Randomly Rebalanced Neural Data (output of balance_classes)
    ClassObj : class
        classifier object from the scikit-learn package
    drop_temps : list
        list of the indexes of templates to not use as features
    sel_freq : int
        the index of the frequency to be used for the narrow channel dropping
    k_folds : int
        Number of Folds to Split between Template | Train/Test sets, defaults to 5,
    seed : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator; If RandomState instance, random_state is
        the random number generator; If None, the random number generator is the RandomState instance used by np.random.
    verbose : bool
        If True the function will print out useful information for user as it runs, defaults to False.

    Returns
    -------
    all_crossvalid_acc : ndarray, (num_folds, frequencies)
        ndarray of the k-fold accuracies
    all_kfold_confusions : ndarray, (num_folds, frequencies, num_labels, num_labels)
        ndarray of the k-fold confusion matrices
    """

    # 1.) Make Array for Holding all of the feature dropping curves
    nested_crossvalid_acc = []  # np.zeros([])
    nested_kfold_confusions = []

    # 2.) Create INDEX of all instances of interests : create_discrete_index()
    label_identities, label_index = cat.create_discrete_index(event_data=event_data)
    identity_index = np.arange(len(label_index))
    sss = cat.StratifiedShuffleSplit(n_splits=k_folds, random_state=seed)
    sss.get_n_splits(identity_index, label_index)

    if verbose:
        print(sss)

    # --------- For Loop over possible Training Sets---------
    for train_index, test_index in sss.split(identity_index, label_index):
        if verbose:
            print("TRAIN:", train_index, "TEST:", test_index)

        X_train, X_test = identity_index[train_index], identity_index[test_index]
        y_train, y_test = label_index[train_index], label_index[test_index]

        # 4.) Use INDEX to Break into corresponding [template/training set| test set] : ml_selector()
        # 4.1) Get template set/training : ml_selector(event_data, identity_index, label_index, sel_instances)
        sel_train = cat.ml_selector(event_data=event_data, identity_index=label_identities, label_index=label_index,
                                    sel_instances=X_train, )

        # 4.1) Get test set : ml_selector()
        sel_test = cat.ml_selector(event_data=event_data, identity_index=label_identities, label_index=label_index,
                                   sel_instances=X_test)

        ## 5.) Use template/training set to make template : make_templates(event_data)
        templates = cat.make_templates(event_data=sel_train)

        ### 5.2) Remove Template that aren't needed from train
        templates = np.delete(templates, drop_temps, axis=0)

        ## 6.1) Use template/training INDEX and template to create Training Pearson Features : pearson_extraction()
        train_pearson_features = cat.pearson_extraction(event_data=sel_train, templates=templates)

        ## 6.2) Use test INDEX and template to create Test Pearson Features : pearson_extraction()
        test_pearson_features = cat.pearson_extraction(event_data=sel_test, templates=templates)

        # 7.1) Reorganize Test Set into Machine Learning Format : ml_order_pearson()
        ml_trials_train, ml_labels_train = cat.ml_order(extracted_features_array=train_pearson_features)

        # 7.2) Get Ledger of the Features
        num_freqs, num_chans, num_temps = np.shape(train_pearson_features[0][0])  # Get the shape of the Feature data
        ordered_index = cat.make_feature_id_ledger(num_freqs=num_freqs, num_chans=num_chans, num_temps=num_temps)

        # 7.3) Reorganize Training Set into Machine Learning Format : ml_order_pearson()
        ml_trials_test, ml_labels_test = cat.ml_order(extracted_features_array=test_pearson_features)

        fold_frequency_accuracies = []
        fold_frequency_confusions = []
        for _, freq in enumerate([sel_freq]):
            if verbose:
                print("On Frequency Band:", freq, " of:", num_freqs)

            ml_trials_train_cp = ml_trials_train.copy()  # make a copy of the feature extracted Train data
            ml_trials_test_cp = ml_trials_test.copy()  # make a copy of the feature extracted Test data
            ordered_index_cp = ordered_index.copy()  # make a copy of the ordered_index
            all_other_freqs = list(np.delete(np.arange(num_freqs), [freq]))  # Make a index of the other frequencies
            temp_feature_dict = cat.make_feature_dict(ordered_index=ordered_index_cp,
                                                      drop_type='frequency')  # Feature Dict
            # reduce to selected frequency from the COPY of the training data
            ml_trials_train_freq, full_drop = cat.drop_features(features=ml_trials_train_cp, keys=temp_feature_dict,
                                                                desig_drop_list=all_other_freqs)
            # reduce to the selected frequency from the COPY of test data
            ml_trials_test_freq, _ = cat.drop_features(features=ml_trials_test_cp, keys=temp_feature_dict,
                                                       desig_drop_list=all_other_freqs)

            # 8.) Perform Nested Feature Dropping with K-Fold Cross Validation
            acc, _, confusion = cat.clip_classification(ClassObj=ClassObj, train_set=ml_trials_train_freq,
                                                        train_labels=ml_labels_train, test_set=ml_trials_test_freq,
                                                        test_labels=ml_labels_test)

            fold_frequency_accuracies.append(acc)
            fold_frequency_confusions.append(confusion)

        nested_crossvalid_acc.append(fold_frequency_accuracies)
        nested_kfold_confusions.append(fold_frequency_confusions)

    # 9.) Combine all curve arrays to one array
    all_crossvalid_acc = np.array(nested_crossvalid_acc)  # (n_folds, n_freqs)
    all_kfold_confusions = np.array(nested_kfold_confusions)  # (n_folds, n_freqs, n_classes, n_classes)

    return all_crossvalid_acc, all_kfold_confusions


def single_frequency_cross_valid_accuracy_chunk_both(power_data, phase_data, ClassObj, drop_temps, sel_freq, k_folds=5,
                                                     seed=None, verbose=False):
    """ Runs the Random Channel Feature Dropping algorithm on a set of pre-processed data

    Parameters
    ----------
    power_data : ndarray | (classes, instances, frequencies, channels, samples)
        Randomly Rebalanced Neural Data (output of balance_classes)
    phase_data : ndarray | (classes, instances, frequencies, channels, samples)
        Randomly Rebalanced Neural Data (output of balance_classes)
    ClassObj : class
        classifier object from the scikit-learn package
    drop_temps : list
        list of the indexes of templates to not use as features
    sel_freq : int
        the index of the frequency to be used for the narrow channel dropping
    k_folds : int
        Number of Folds to Split between Template | Train/Test sets, defaults to 5,
    seed : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator; If RandomState instance, random_state is
        the random number generator; If None, the random number generator is the RandomState instance used by np.random.
    verbose : bool
        If True the funtion will print out useful information for user as it runs, defaults to False.

    Returns
    -------
    """

    # 1.) Make Array for Holding all of the feature dropping curves
    nested_crossvalid_acc = []  # np.zeros([])
    nested_kfold_confusions = []

    # 2.) Create INDEX of all instances of interests : create_discrete_index()
    label_identities, label_index = cat.create_discrete_index(event_data=power_data)
    identity_index = np.arange(len(label_index))
    sss = cat.StratifiedShuffleSplit(n_splits=k_folds, random_state=seed)
    sss.get_n_splits(identity_index, label_index)

    if verbose:
        print(sss)
        fold_number = 0

    # --------- For Loop over possible Training Sets---------
    for train_index, test_index in sss.split(identity_index, label_index):
        if verbose:
            print("TRAIN:", train_index, "TEST:", test_index)
            fold_number += 1
            print("On Fold #" + str(fold_number) + ' of ' + str(k_folds))

        X_train, X_test = identity_index[train_index], identity_index[test_index]
        y_train, y_test = label_index[train_index], label_index[test_index]

        # 4.) Use INDEX to Break into corresponding [template/training set| test set] : ml_selector()
        # 4.1) Get template set/training : ml_selector(power_data, identity_index, label_index, sel_instances)
        sel_train_pow = cat.ml_selector(event_data=power_data, identity_index=label_identities, label_index=label_index,
                                        sel_instances=X_train, )
        sel_train_phas = cat.ml_selector(event_data=phase_data, identity_index=label_identities,
                                         label_index=label_index,
                                         sel_instances=X_train, )

        # 4.1) Get test set : ml_selector()
        sel_test_pow = cat.ml_selector(event_data=power_data, identity_index=label_identities, label_index=label_index,
                                       sel_instances=X_test)
        sel_test_phas = cat.ml_selector(event_data=phase_data, identity_index=label_identities, label_index=label_index,
                                        sel_instances=X_test)

        # 5.) Use template/training set to make template : make_templates(power_data)
        templates_pow = cat.make_templates(event_data=sel_train_pow)
        templates_phas = cat.make_templates(event_data=sel_train_phas)

        ### 5.2) Remove Template that aren't needed from train
        templates_pow = np.delete(templates_pow, drop_temps, axis=0)
        templates_phas = np.delete(templates_phas, drop_temps, axis=0)

        # 6.1) Use template/training INDEX and template to create Training Pearson Features : pearson_extraction()
        train_pearson_features_pow = cat.pearson_extraction(event_data=sel_train_pow, templates=templates_pow)
        train_pearson_features_phas = cat.pearson_extraction(event_data=sel_train_phas, templates=templates_phas)

        # 6.2) Use test INDEX and template to create Test Pearson Features : pearson_extraction()
        test_pearson_features_pow = cat.pearson_extraction(event_data=sel_test_pow, templates=templates_pow)
        test_pearson_features_phas = cat.pearson_extraction(event_data=sel_test_phas, templates=templates_phas)

        # 7.1) Reorganize Test Set into Machine Learning Format : ml_order_pearson()
        ml_trials_train_pow, ml_labels_train = cat.ml_order(extracted_features_array=train_pearson_features_pow)
        ml_trials_train_phas, _ = cat.ml_order(extracted_features_array=train_pearson_features_phas)
        ml_trials_train = np.concatenate([ml_trials_train_pow, ml_trials_train_phas], axis=-1)

        # 7.2) Get Ledger of the Features
        num_freqs, num_chans, num_temps = np.shape(
            train_pearson_features_pow[0][0])  # Get the shape of the Feature data
        ordered_index = cat.make_feature_id_ledger(num_freqs=num_freqs, num_chans=num_chans, num_temps=num_temps)
        ordered_index = np.concatenate([ordered_index, ordered_index], axis=0)

        # 7.3) Reorganize Training Set into Machine Learning Format : ml_order_pearson()
        ml_trials_test_pow, ml_labels_test = cat.ml_order(extracted_features_array=test_pearson_features_pow)
        ml_trials_test_phas, _ = cat.ml_order(extracted_features_array=test_pearson_features_phas)
        ml_trials_test = np.concatenate([ml_trials_test_pow, ml_trials_test_phas], axis=-1)

        fold_frequency_accuracies = []
        fold_frequency_confusions = []
        for _, freq in enumerate([sel_freq]):
            # if verbose:
            #     print("On Frequency Band:", freq, " of:", num_freqs)

            ml_trials_train_cp = ml_trials_train.copy()  # make a copy of the feature extracted Train data
            ml_trials_test_cp = ml_trials_test.copy()  # make a copy of the feature extracted Test data
            ordered_index_cp = ordered_index.copy()  # make a copy of the ordered_index
            all_other_freqs = list(np.delete(np.arange(num_freqs), [freq]))  # Make a index of the other frequencies
            temp_feature_dict = cat.make_feature_dict(ordered_index=ordered_index_cp,
                                                      drop_type='frequency')  # Feature Dict
            # reduce to selected frequency from the COPY of the training data
            ml_trials_train_freq, full_drop = cat.drop_features(features=ml_trials_train_cp, keys=temp_feature_dict,
                                                                desig_drop_list=all_other_freqs)
            # reduce to but the selected frequency from the COPY of test data
            ml_trials_test_freq, _ = cat.drop_features(features=ml_trials_test_cp, keys=temp_feature_dict,
                                                       desig_drop_list=all_other_freqs)

            # 8.) Perform K-Fold Cross Validation
            acc, _, confusion = cat.clip_classification(ClassObj=ClassObj, train_set=ml_trials_train_freq,
                                                        train_labels=ml_labels_train, test_set=ml_trials_test_freq,
                                                        test_labels=ml_labels_test)

            fold_frequency_accuracies.append(acc)
            fold_frequency_confusions.append(confusion)

        nested_crossvalid_acc.append(fold_frequency_accuracies)
        nested_kfold_confusions.append(fold_frequency_confusions)

    # 9.) Combine all curve arrays to one array
    all_crossvalid_acc = np.array(nested_crossvalid_acc)  # (n_folds, n_freqs)
    all_kfold_confusions = np.array(nested_kfold_confusions)  # (n_folds, n_freqs, n_classes, n_classes)

    return all_crossvalid_acc, all_kfold_confusions


def fix_best_feature_dropping_report(bird_id='z007', session='day-2016-09-09'):
    warnings.filterwarnings("ignore", category=UserWarning)  # So that it doesn't print warnings until oblivion

    zdata = ImportData(bird_id=bird_id, session=session)

    # Get the Bird Specific Machine Learning Meta Data
    bad_channels = all_bad_channels[bird_id]
    drop_temps = all_drop_temps[bird_id]

    # Get the Best Parameters for Bindwidth and Offset
    bin_widths = best_bin_width[session]
    offsets = best_offset[session]

    # Reshape Handlabels into Useful Format
    chunk_labels_list, chunk_onsets_list = fet.get_chunk_handlabels(handlabels_list=zdata.song_handlabels)

    # Set the Frequency Bands to Be Used for Feature Extraction
    fc_lo = [4, 8, 25, 30, 50]
    fc_hi = [8, 12, 35, 50, 70]

    # Pre-Process the Data (Power)
    pred_data_pow = hbp.feature_extraction_chunk(neural_chunks=zdata.song_neural,
                                                 fs=1000,
                                                 l_freqs=fc_lo,
                                                 h_freqs=fc_hi,
                                                 hilbert='amplitude',
                                                 bad_channels=bad_channels,
                                                 drop_bad=True,
                                                 verbose=True)

    # Pre-Process the Data (Phase)
    pred_data_phase = hbp.feature_extraction_chunk(neural_chunks=zdata.song_neural,
                                                   fs=1000,
                                                   l_freqs=fc_lo, h_freqs=fc_hi,
                                                   hilbert='phase',
                                                   bad_channels=bad_channels,
                                                   drop_bad=True,
                                                   verbose=True)

    # Get the Bird Specific label Instructions
    label_instructions = all_label_instructions[bird_id]  # get this birds default label instructions

    times_of_interest = fet.label_extractor(all_labels=chunk_labels_list,
                                            starts=chunk_onsets_list[0],
                                            label_instructions=label_instructions)

    # Get Silence Periods

    silent_periods = fet.long_silence_finder(silence=8,
                                             all_labels=chunk_labels_list,
                                             all_starts=chunk_onsets_list[0],
                                             all_ends=chunk_onsets_list[1],
                                             window=(-500, 500))

    # Append the Selected Silence to the end of the Events array
    times_of_interest.append(silent_periods)

    pow_accuracy_holder = []
    phase_accuracy_holder = []
    both_accuracy_holder = []
    for freq_num, (offset, bin_width) in enumerate(zip(offsets, bin_widths)):
        # Grab the Neural Activity Centered on Each event
        set_window = (offset - bin_width, offset)

        chunk_events_power = fet.event_clipper_nd(data=pred_data_pow, label_events=times_of_interest,
                                                  fs=1000, window=set_window)
        chunk_events_phase = fet.event_clipper_nd(data=pred_data_phase, label_events=times_of_interest,
                                                  fs=1000, window=set_window)

        # Balance the sets

        chunk_events_balanced_pow = mlpu.balance_classes(chunk_events_power)
        chunk_events_balanced_phase = mlpu.balance_classes(chunk_events_phase)

        priors = get_priors(num_labels=len(times_of_interest))  # Set the priors to be equal
        print(priors)

        rand_obj = LinearDiscriminantAnalysis(n_components=None, priors=priors, shrinkage=None,
                                              solver='svd', store_covariance=False, tol=0.0001)

        # Run Analysis on Only Power
        nested_accuracy_pow, _ = single_frequency_cross_valid_accuracy_chunk(event_data=chunk_events_balanced_pow,
                                                                             ClassObj=rand_obj, drop_temps=drop_temps,
                                                                             sel_freq=freq_num,
                                                                             k_folds=5, seed=None, verbose=True)
        pow_accuracy_holder.append(nested_accuracy_pow)

        # Run Analysis on Only Phase
        nested_accuracy_phase, _ = single_frequency_cross_valid_accuracy_chunk(event_data=chunk_events_balanced_phase,
                                                                               ClassObj=rand_obj,
                                                                               drop_temps=drop_temps,
                                                                               sel_freq=freq_num,
                                                                               k_folds=5, seed=None, verbose=True)

        phase_accuracy_holder.append(nested_accuracy_phase)

        # Run Analysis on Both Features Independently
        nested_accuracy_both, _ = single_frequency_cross_valid_accuracy_chunk_both(
            power_data=chunk_events_balanced_pow, phase_data=chunk_events_balanced_phase, ClassObj=rand_obj,
            drop_temps=drop_temps, sel_freq=freq_num, k_folds=5, seed=None, verbose=True)
        both_accuracy_holder.append(nested_accuracy_both)

    # Save the power Values
    accuracy_pow = np.mean(pow_accuracy_holder, axis=1)
    std_pow = np.std(pow_accuracy_holder, axis=1, ddof=1)
    _save_numpy_data(data=accuracy_pow, data_name="accuracy_pow", bird_id=bird_id,
                     session=session, destination=channel_drop_path, make_parents=True, verbose=True)
    _save_numpy_data(data=std_pow, data_name="std_all_pow", bird_id=bird_id,
                     session=session, destination=channel_drop_path, make_parents=True, verbose=True)

    # Save the phase Values
    accuracy_phase = np.mean(phase_accuracy_holder, axis=1)
    std_phase = np.std(phase_accuracy_holder, axis=1, ddof=1)
    _save_numpy_data(data=accuracy_phase, data_name="accuracy_phase", bird_id=bird_id,
                     session=session,
                     destination=channel_drop_path, make_parents=True, verbose=True)
    _save_numpy_data(data=std_phase, data_name="std_all_phase", bird_id=bird_id,
                     session=session, destination=channel_drop_path, make_parents=True, verbose=True)

    # Save the both Values
    accuracy_both = np.mean(both_accuracy_holder, axis=1)
    std_both = np.std(both_accuracy_holder, axis=1, ddof=1)
    _save_numpy_data(data=accuracy_both, data_name="accuracy_both", bird_id=bird_id,
                     session=session, destination=channel_drop_path, make_parents=True, verbose=True)
    _save_numpy_data(data=std_both, data_name="std_all_both", bird_id=bird_id,
                     session=session, destination=channel_drop_path, make_parents=True, verbose=True)


def fix_best_feature_dropping_report_repeats(bird_id='z007', session='day-2016-09-09'):
    warnings.filterwarnings("ignore", category=UserWarning)  # So that it doesn't print warnings until oblivion

    zdata = ImportData(bird_id=bird_id, session=session)

    # Get the Bird Specific Machine Learning Meta Data
    bad_channels = all_bad_channels[bird_id]
    drop_temps = all_drop_temps[bird_id]

    # Get the Best Parameters for Bindwidth and Offset
    bin_widths = best_bin_width[session]
    offsets = best_offset[session]

    # Reshape Handlabels into Useful Format
    chunk_labels_list, chunk_onsets_list = fet.get_chunk_handlabels(handlabels_list=zdata.song_handlabels)

    # Set the Frequency Bands to Be Used for Feature Extraction
    fc_lo = [4, 8, 25, 30, 50]
    fc_hi = [8, 12, 35, 50, 70]

    # Pre-Process the Data (Power)
    pred_data_pow = hbp.feature_extraction_chunk(neural_chunks=zdata.song_neural,
                                                 fs=1000,
                                                 l_freqs=fc_lo,
                                                 h_freqs=fc_hi,
                                                 hilbert='amplitude',
                                                 bad_channels=bad_channels,
                                                 drop_bad=True,
                                                 verbose=True)

    # Pre-Process the Data (Phase)
    pred_data_phase = hbp.feature_extraction_chunk(neural_chunks=zdata.song_neural,
                                                   fs=1000,
                                                   l_freqs=fc_lo, h_freqs=fc_hi,
                                                   hilbert='phase',
                                                   bad_channels=bad_channels,
                                                   drop_bad=True,
                                                   verbose=True)

    # Get the Bird Specific label Instructions
    label_instructions = all_label_instructions[bird_id]  # get this birds default label instructions

    times_of_interest = fet.label_extractor(all_labels=chunk_labels_list,
                                            starts=chunk_onsets_list[0],
                                            label_instructions=label_instructions)

    # Get Silence Periods

    silent_periods = fet.long_silence_finder(silence=8,
                                             all_labels=chunk_labels_list,
                                             all_starts=chunk_onsets_list[0],
                                             all_ends=chunk_onsets_list[1],
                                             window=(-500, 500))

    # Append the Selected Silence to the end of the Events array
    times_of_interest.append(silent_periods)

    pow_accuracy_holder = []
    phase_accuracy_holder = []
    both_accuracy_holder = []
    for freq_num, (offset, bin_width) in enumerate(zip(offsets, bin_widths)):
        # Grab the Neural Activity Centered on Each event
        set_window = (offset - bin_width, offset)

        chunk_events_power = fet.event_clipper_nd(data=pred_data_pow, label_events=times_of_interest,
                                                  fs=1000, window=set_window)
        chunk_events_phase = fet.event_clipper_nd(data=pred_data_phase, label_events=times_of_interest,
                                                  fs=1000, window=set_window)

        # Balance the sets

        chunk_events_balanced_pow = mlpu.balance_classes(chunk_events_power)
        chunk_events_balanced_phase = mlpu.balance_classes(chunk_events_phase)

        priors = get_priors(num_labels=len(times_of_interest))  # Set the priors to be equal
        print(priors)

        rand_obj = LinearDiscriminantAnalysis(n_components=None, priors=priors, shrinkage=None,
                                              solver='svd', store_covariance=False, tol=0.0001)
        nested_accuracy_phase = []
        nested_accuracy_pow = []
        nested_accuracy_both = []
        for index in range(1000):

            # Run Analysis on Only Power
            accuracy_pow, _ = single_frequency_cross_valid_accuracy_chunk(event_data=chunk_events_balanced_pow,
                                                                                 ClassObj=rand_obj, drop_temps=drop_temps,
                                                                                 sel_freq=freq_num,
                                                                                 k_folds=5, seed=None, verbose=True)
            nested_accuracy_pow.extend(accuracy_pow)

            # Run Analysis on Only Phase
            accuracy_phase, _ = single_frequency_cross_valid_accuracy_chunk(event_data=chunk_events_balanced_phase,
                                                                                   ClassObj=rand_obj,
                                                                                   drop_temps=drop_temps,
                                                                                   sel_freq=freq_num,
                                                                                   k_folds=5, seed=None, verbose=True)

            nested_accuracy_phase.extend(accuracy_phase)

            # Run Analysis on Both Features Independently
            accuracy_both, _ = single_frequency_cross_valid_accuracy_chunk_both(
                power_data=chunk_events_balanced_pow, phase_data=chunk_events_balanced_phase, ClassObj=rand_obj,
                drop_temps=drop_temps, sel_freq=freq_num, k_folds=5, seed=None, verbose=True)
            nested_accuracy_both.extend(accuracy_both)
        
        pow_accuracy_holder.append(nested_accuracy_pow)

        phase_accuracy_holder.append(nested_accuracy_phase)

        both_accuracy_holder.append(nested_accuracy_both)
        
    # Save the power Values
    accuracy_pow = np.mean(pow_accuracy_holder, axis=1)
    std_pow = np.std(pow_accuracy_holder, axis=1, ddof=1)
    _save_numpy_data(data=accuracy_pow, data_name="accuracy_repeat_pow", bird_id=bird_id,
                     session=session, destination=channel_drop_path, make_parents=True, verbose=True)
    _save_numpy_data(data=std_pow, data_name="std_all_repeat_pow", bird_id=bird_id,
                     session=session, destination=channel_drop_path, make_parents=True, verbose=True)

    # Save the phase Values
    accuracy_phase = np.mean(phase_accuracy_holder, axis=1)
    std_phase = np.std(phase_accuracy_holder, axis=1, ddof=1)
    _save_numpy_data(data=accuracy_phase, data_name="accuracy_repeat_phase", bird_id=bird_id,
                     session=session,
                     destination=channel_drop_path, make_parents=True, verbose=True)
    _save_numpy_data(data=std_phase, data_name="std_all_repeat_phase", bird_id=bird_id,
                     session=session, destination=channel_drop_path, make_parents=True, verbose=True)

    # Save the both Values
    accuracy_both = np.mean(both_accuracy_holder, axis=1)
    std_both = np.std(both_accuracy_holder, axis=1, ddof=1)
    _save_numpy_data(data=accuracy_both, data_name="accuracy_repeat_both", bird_id=bird_id,
                     session=session, destination=channel_drop_path, make_parents=True, verbose=True)
    _save_numpy_data(data=std_both, data_name="std_all_repeat_both", bird_id=bird_id,
                     session=session, destination=channel_drop_path, make_parents=True, verbose=True)

