""" Functions to make report of the Parameter Sweep Analysis"""
from src.analysis.ml_pipeline_utilities import balance_classes
from src.analysis.ml_pipeline_utilities import all_bad_channels, all_label_instructions, all_drop_temps, get_priors
import src.analysis.hilbert_based_pipeline as hbp

from BirdSongToolbox.import_data import ImportData
import BirdSongToolbox.chunk_analysis_tools as cat
import BirdSongToolbox.free_epoch_tools as fet
from BirdSongToolbox.file_utility_functions import _save_numpy_data, _load_numpy_data

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

param_sweep_path = '/home/debrown/parameter_sweep_results'


def get_parameter_sweep_results(bird_id='z007', session='day-2016-09-09', verbose=True):
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
    accuracy = _load_numpy_data(data_name="accuracies", bird_id=bird_id, session=session, source=param_sweep_path,
                                verbose=verbose)
    confusions = _load_numpy_data(data_name="confusions", bird_id=bird_id, session=session, source=param_sweep_path,
                                  verbose=verbose)
    return accuracy, confusions


def get_parameter_sweep_results_freq(bird_id='z007', session='day-2016-09-09', verbose=True):
    """ Import the results of make_parameter_sweep_freq

    :param bird_id:
    :param session:
    :param verbose:
    :return:

    accuracy : ndarray, (bin_widths, offsets, num_folds, frequencies)
        ndarray of the k-fold accuracies

    confusions : ndarray, (bin_widths, offsets, num_folds, frequencies)
        ndarray of the k-fold confusion matrices
    """
    accuracy = _load_numpy_data(data_name="accuracies_freq", bird_id=bird_id, session=session, source=param_sweep_path,
                                verbose=verbose)
    confusions = _load_numpy_data(data_name="confusions_freq", bird_id=bird_id, session=session,
                                  source=param_sweep_path, verbose=verbose)
    return accuracy, confusions


def cross_valid_accuracy_chunk(event_data, ClassObj, drop_temps, k_folds=5, seed=None, verbose=False):
    """ K-Fold Validated Accuracy grouping all of the Frequency together

    Parameters
    ----------
    event_data : ndarray | (classes, instances, frequencies, channels, samples)
        Randomly Rebalanced Neural Data (output of balance_classes)
    ClassObj : class
        classifier object from the scikit-learn package
    drop_temps : list
        list of the indexes of templates to not use as features
    k_folds : int
        Number of Folds to Split between Template | Train/Test sets, defaults to 5,
    seed : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator; If RandomState instance, random_state is
        the random number generator; If None, the random number generator is the RandomState instance used by np.random.
    verbose : bool
        If True the function will print out useful information for user as it runs, defaults to False.

    Returns
    -------
    all_crossvalid_acc : ndarray, (num_folds, )
        ndarray of the k-fold accuracies
    all_kfold_confusions : ndarray, (num_folds, num_labels, num_labels)
        ndarray of the k-fold confusion matrices
    """

    # 1.) Make Array for Holding all of the feature dropping curves
    fold_frequency_accuracies = []  # np.zeros([])
    fold_frequency_confusions = []

    # 2.) Create INDEX of all instances of interests : create_discrete_index()
    label_identities, label_index = cat.create_discrete_index(event_data=event_data)
    identity_index = np.arange(len(label_index))
    sss = StratifiedShuffleSplit(n_splits=k_folds, random_state=seed)
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

        # 8.) Perform Nested Feature Dropping with K-Fold Cross Validation
        acc, _, confusion = cat.clip_classification(ClassObj=ClassObj, train_set=ml_trials_train,
                                                    train_labels=ml_labels_train, test_set=ml_trials_test,
                                                    test_labels=ml_labels_test)

        fold_frequency_accuracies.append(acc)
        fold_frequency_confusions.append(confusion)

    # 9.) Combine all curve arrays to one array
    all_crossvalid_acc = np.array(fold_frequency_accuracies)  # (n_folds, n_freqs)
    all_kfold_confusions = np.array(fold_frequency_confusions)  # (n_folds, n_freqs, n_classes, n_classes)

    return all_crossvalid_acc, all_kfold_confusions


def single_frequency_cross_valid_accuracy_chunk(event_data, ClassObj, drop_temps, k_folds=5, seed=None, verbose=False):
    """   K-Fold Validated Accuracy for each Frequency Band Seperately

    Parameters
    ----------
    event_data : ndarray | (classes, instances, frequencies, channels, samples)
        Randomly Rebalanced Neural Data (output of balance_classes)
    ClassObj : class
        classifier object from the scikit-learn package
    drop_temps : list
        list of the indexes of templates to not use as features
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
    sss = StratifiedShuffleSplit(n_splits=k_folds, random_state=seed)
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
        for freq in range(num_freqs):
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


def parameter_sweep_single_freq(pred_data, times_of_interest, rand_obj, drop_temps, off_sets, bin_widths, verbose=True):
    """  Parameter Sweep with Each Frequency Kept Seperate

    :param pred_data:
    :param times_of_interest:
    :param rand_obj:
    :param off_sets:
    :param bin_widths:
    :param verbose:
    :return:

    all_acc : ndarray, (bin_widths, offsets, num_folds, frequencies)
        ndarray of the k-fold accuracies

    all_confusions : ndarray, (bin_widths, offsets, num_folds, frequencies)
        ndarray of the k-fold confusion matrices
    """
    # def parameter_sweep():
    # Grab the Neural Activity Centered on Each event

    all_acc = []
    all_confusions = []

    for i, bin_width in enumerate(bin_widths):
        if verbose:
            print(f'Now On bin_width : {i} of {len(bin_widths)}')

        bin_width_list1 = []
        bin_width_list2 = []

        for j, offset in enumerate(off_sets):
            if verbose:
                print(f'Now On Offset : {j} of {len(off_sets)}')
            set_window = (offset - bin_width, offset)
            chunk_events = fet.event_clipper_nd(data=pred_data, label_events=times_of_interest, fs=1000,
                                                window=set_window)
            chunk_events_balanced = balance_classes(chunk_events, safe=True)  # Balance Classes
            test_acc, test_confusions = single_frequency_cross_valid_accuracy_chunk(event_data=chunk_events_balanced,
                                                                                    ClassObj=rand_obj,
                                                                                    drop_temps=drop_temps,
                                                                                    k_folds=5, seed=None, verbose=False)
            bin_width_list1.append(test_acc)
            bin_width_list2.append(test_confusions)
        all_acc.append(bin_width_list1)
        all_confusions.append(bin_width_list2)

    return np.array(all_acc), np.array(all_confusions)


def parameter_sweep_all(pred_data, times_of_interest, rand_obj, drop_temps, off_sets, bin_widths, verbose=True):
    """ Parameter Sweep Grouping all of the frequencies together

    :param pred_data:
    :param times_of_interest:
    :param rand_obj:
    :param off_sets:
    :param bin_widths:
    :param verbose:
    :return:

    all_acc : ndarray, (bin_widths, offsets, num_folds, frequencies)
        ndarray of the k-fold accuracies

    all_confusions : ndarray, (bin_widths, offsets, num_folds, frequencies)
        ndarray of the k-fold confusion matrices
    """
    # def parameter_sweep():
    # Grab the Neural Activity Centered on Each event

    all_acc = []
    all_confusions = []

    for i, bin_width in enumerate(bin_widths):
        if verbose:
            print(f'Now On bin_width : {i} of {len(bin_widths)}')

        bin_width_list1 = []
        bin_width_list2 = []

        for j, offset in enumerate(off_sets):
            if verbose:
                print(f'Now On Offset : {j} of {len(off_sets)}')
            set_window = (offset - bin_width, offset)
            chunk_events = fet.event_clipper_nd(data=pred_data, label_events=times_of_interest, fs=1000,
                                                window=set_window)
            chunk_events_balanced = balance_classes(chunk_events, safe=True)  # Balance Classes
            test_acc, test_confusions = cross_valid_accuracy_chunk(event_data=chunk_events_balanced,
                                                                   ClassObj=rand_obj, drop_temps=drop_temps,
                                                                   k_folds=5, seed=None, verbose=False)
            bin_width_list1.append(test_acc)
            bin_width_list2.append(test_confusions)
        all_acc.append(bin_width_list1)
        all_confusions.append(bin_width_list2)

    return np.array(all_acc), np.array(all_confusions)


def make_parameter_sweep_freq(bird_id='z007', session='day-2016-09-09', verbose=True):
    """ Run Parameter Sweep For each frequencies together and store the results using numpy

    :param bird_id:
    :param session:
    :param verbose:
    :return:
    """
    zdata = ImportData(bird_id=bird_id, session=session)
    bad_channels = all_bad_channels[bird_id]
    drop_temps = all_drop_temps[bird_id]

    # Set the Frequency Bands to Be Used for Feature Extraction
    fc_lo = [4, 8, 25, 35, 50]
    fc_hi = [8, 12, 35, 50, 70]

    chunk_labels_list, chunk_onsets_list = fet.get_chunk_handlabels(handlabels_list=zdata.song_handlabels)

    pred_data = hbp.feature_extraction_chunk(neural_chunks=zdata.song_neural, fs=1000, l_freqs=fc_lo, h_freqs=fc_hi,
                                             norm=False, bad_channels=bad_channels, drop_bad=True, verbose=verbose)
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

    times_of_interest.append(silent_periods)  # Append the Selected Silence to the end of the Events array
    priors = get_priors(num_labels=len(times_of_interest))  # Set the priors to be equal

    rand_obj = LinearDiscriminantAnalysis(n_components=None,
                                          priors=priors,
                                          shrinkage=None,
                                          solver='svd', store_covariance=False, tol=0.0001)

    off_sets = np.arange(0, -150, -5)  # [-10, -20, -30]
    # off_sets = [-10, -20, -30]
    bin_widths = np.arange(5, 200, 5)  # [10, 20, 30, 40]
    # bin_widths = [10, 20, 30, 40]

    all_acc, all_confusions = parameter_sweep_single_freq(pred_data=pred_data, times_of_interest=times_of_interest,
                                                          drop_temps=drop_temps, rand_obj=rand_obj, off_sets=off_sets,
                                                          bin_widths=bin_widths, verbose=True)

    _save_numpy_data(data=all_acc, data_name="accuracies_freq", bird_id=bird_id, session=session,
                     destination=param_sweep_path, make_parents=True, verbose=True)
    _save_numpy_data(data=all_confusions, data_name="confusions_freq", bird_id=bird_id, session=session,
                     destination=param_sweep_path, make_parents=True, verbose=True)


# The Below function needs to be altered so that it doens't do the same as the single_freq version
def make_parameter_sweep(bird_id='z007', session='day-2016-09-09', verbose=True):
    """ RunParameter Sweep Grouping all of the frequencies together and save results using numpy

    :param bird_id:
    :param session:
    :param verbose:
    :return:
    """
    zdata = ImportData(bird_id=bird_id, session=session)
    bad_channels = all_bad_channels[bird_id]
    drop_temps = all_drop_temps[bird_id]

    # Set the Frequency Bands to Be Used for Feature Extraction
    fc_lo = [4, 8, 25, 35, 50]
    fc_hi = [8, 12, 35, 50, 70]

    chunk_labels_list, chunk_onsets_list = fet.get_chunk_handlabels(handlabels_list=zdata.song_handlabels)

    pred_data = hbp.feature_extraction_chunk(neural_chunks=zdata.song_neural, fs=1000, l_freqs=fc_lo, h_freqs=fc_hi,
                                             norm=False, bad_channels=bad_channels, drop_bad=True, verbose=verbose)
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

    times_of_interest.append(silent_periods)  # Append the Selected Silence to the end of the Events array
    priors = get_priors(num_labels=len(times_of_interest))  # Set the priors to be equal

    rand_obj = LinearDiscriminantAnalysis(n_components=None,
                                          priors=priors,
                                          shrinkage=None,
                                          solver='svd', store_covariance=False, tol=0.0001)

    off_sets = np.arange(0, -150, -5)  # [-10, -20, -30]
    # off_sets = [-10, -20, -30]
    bin_widths = np.arange(5, 200, 5)  # [10, 20, 30, 40]
    # bin_widths = [10, 20, 30, 40]

    all_acc, all_confusions = parameter_sweep_single_freq(pred_data=pred_data, times_of_interest=times_of_interest,
                                                          drop_temps=drop_temps, rand_obj=rand_obj, off_sets=off_sets,
                                                          bin_widths=bin_widths, verbose=True)

    _save_numpy_data(data=all_acc, data_name="accuracies", bird_id=bird_id, session=session,
                     destination=param_sweep_path, make_parents=True, verbose=True)
    _save_numpy_data(data=all_confusions, data_name="confusions", bird_id=bird_id, session=session,
                     destination=param_sweep_path, make_parents=True, verbose=True)
