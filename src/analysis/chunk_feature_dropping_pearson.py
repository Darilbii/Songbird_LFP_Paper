import BirdSongToolbox.chunk_analysis_tools as cat
import matplotlib.pyplot as plt
import numpy as np
import scipy
import matplotlib.patches as mpatches
import random

from src.analysis.ml_pipeline_utilities import all_bad_channels, all_drop_temps, all_label_instructions

# Functions added for the Report
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

import BirdSongToolbox.free_epoch_tools as fet
from BirdSongToolbox.import_data import ImportData
from BirdSongToolbox.chunk_analysis_tools import random_feature_drop_multi_narrow_chunk
from BirdSongToolbox.file_utility_functions import _save_numpy_data, _load_numpy_data

import src.analysis.ml_pipeline_utilities as mlpu
from src.analysis.chunk_parameter_sweep_bin_offset import get_priors
import src.analysis.hilbert_based_pipeline as hbp

import warnings

channel_drop_path = '/home/debrown/channel_dropping_results'


def get_optimum_channel_dropping_results(bird_id='z007', session='day-2016-09-09', feat_type: str = 'pow',
                                         verbose=True):
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

    mean_curve_list = []
    std_curve_list = []

    for index in range(5):
        mean_curve_name = "mean_curve_" + feat_type + str(index) + "_2"
        mean_curve_sel = _load_numpy_data(data_name=mean_curve_name, bird_id=bird_id, session=session,
                                          source=channel_drop_path, verbose=verbose)
        mean_curve_list.append(mean_curve_sel)

        std_curve_name = "std_curve_" + feat_type + str(index) + "_2"
        std_curve_sel = _load_numpy_data(data_name=std_curve_name, bird_id=bird_id, session=session,
                                         source=channel_drop_path, verbose=verbose)
        std_curve_list.append(std_curve_sel)

    mean_curve = np.concatenate(mean_curve_list, axis=0)
    std_curve = np.concatenate(std_curve_list, axis=0)

    return mean_curve, std_curve


def get_feature_dropping_results(bird_id='z007', session='day-2016-09-09', feat_type: str = 'pow', verbose=True):
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

    mean_curve_name = "mean_curve_" + feat_type
    mean_curve = _load_numpy_data(data_name=mean_curve_name, bird_id=bird_id, session=session, source=channel_drop_path,
                                  verbose=verbose)
    std_curve_name = "std_curve_" + feat_type

    std_curve = _load_numpy_data(data_name=std_curve_name, bird_id=bird_id, session=session, source=channel_drop_path,
                                 verbose=verbose)

    return mean_curve, std_curve


def plot_single_drop_curve(curve, err_bar, ch_range, color, top, bottom, ax=None):
    if ax is None:
        plt.plot(ch_range[::-1], curve, color=color, label=f" {top} - {bottom} Hz")  # Main Drop Curve
        plt.fill_between(ch_range[::-1], curve[:, 0] - err_bar[:, 0], curve[:, 0] + err_bar[:, 0],
                         color=color, alpha=0.2)

    else:
        ax.plot(ch_range[::-1], curve, color=color, label=f" {top} - {bottom} Hz")  # Main Drop Curve
        ax.fill_between(ch_range[::-1], curve[:, 0] - err_bar[:, 0], curve[:, 0] + err_bar[:, 0],
                        color=color, alpha=0.2)


def plot_featdrop_multi(drop_curve_list, std_list, Tops, Bottoms, chance_level, font=20, title_font=30,
                        title="Place Holder", verbose=False):
    """ Plots a single feature dropping cure

    :param drop_curve_list:
    :param Tops:
    :param Bottoms:
    :param chance_level:
    :param font:
    :param title_font:
    :param title:
    :param verbose:
    :return:
    """
    # fig= plt.figure(figsize=(15,15))
    plt.figure(figsize=(7, 7))  # Create Figure and Set Size

    colors = ['aqua', 'darkorange', 'cornflowerblue', 'tab:brown', 'tab:pink',
              'tab:gray', 'tab:green', 'xkcd:dark olive green',
              'xkcd:ugly yellow', 'xkcd:fire engine red', 'xkcd:radioactive green']

    num_channels = drop_curve_list.shape[1]  # Make x-axis based off the First Curve
    ch_range = np.arange(0, num_channels, 1)

    if verbose:
        print("Chance Level is: ", chance_level)

    # Main Dropping Curve

    patch_list = []

    for index, (curve, err_bar) in enumerate(zip(drop_curve_list, std_list)):
        if verbose:
            print('Making plot for curve: ', index)

        color = colors[index]
        plot_single_drop_curve(curve=curve, err_bar=err_bar, ch_range=ch_range, color=color,
                               top=Tops[index], bottom=Bottoms[index], ax=None)

        patch_list.append(mpatches.Patch(color=color, label=f' {Tops[index]} - {Bottoms[index]} Hz'))  # Set Patches

    # Plot Chance
    plt.plot(ch_range, chance_level * np.ones(ch_range.shape), '--k', linewidth=5)
    patch_list.append(mpatches.Patch(color='w', label=f'{round(chance_level,2)} Binomial Chance'))

    # Make Legend
    plt.legend(handles=patch_list, bbox_to_anchor=(1.05, .61), loc=2, borderaxespad=0.)

    # Axis Labels
    plt.title(title, fontsize=title_font)
    plt.xlabel('No. of Channels', fontsize=font)
    plt.ylabel('Accuracy', fontsize=font)

    # Format Annotatitng Ticks
    plt.tick_params(axis='both', which='major', labelsize=font)
    plt.tick_params(axis='both', which='minor', labelsize=font)
    plt.ylim(0, 1.0)
    plt.xlim(1, num_channels - 1)


def random_feature_drop_multi_narrow_chunk_both(power_data, phase_data, ClassObj, drop_temps, k_folds=5, seed=None,
                                                verbose=False):
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
    nested_dropping_curves = []  # np.zeros([])

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

        repeated_freq_curves = []
        test_list = list(np.arange(num_chans))
        random.seed(0)
        for index in range(5000):
            drop_order = random.sample(test_list, k=len(test_list))
            fold_frequency_curves = []
            for freq in range(num_freqs):
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
                ordered_index_cp = np.delete(ordered_index_cp, full_drop,
                                             axis=0)  # Remove features from other frequencies

                # 8.) Perform Nested Feature Dropping with K-Fold Cross Validation
                nested_drop_curve = cat.ordered_feature_dropping(train_set=ml_trials_train_freq,
                                                                 train_labels=ml_labels_train,
                                                                 test_set=ml_trials_test_freq,
                                                                 test_labels=ml_labels_test,
                                                                 ordered_index=ordered_index_cp, drop_type='channel',
                                                                 Class_Obj=ClassObj, order=drop_order, verbose=False)
                fold_frequency_curves.append(nested_drop_curve)  # For each Individual Frequency Band

            if verbose:
                if index % 100 == 0:
                    print('on loop' + str(index))

            repeated_freq_curves.append(fold_frequency_curves)  # Exhaustive Feature Dropping
        nested_dropping_curves.append(repeated_freq_curves)  # All of the Curves

    # 9.) Combine all curve arrays to one array
    all_drop_curves = np.array(nested_dropping_curves)  # (folds, frequencies, num_dropped, 1)

    # 10.) Calculate curve metrics
    fold_mean_curve = np.mean(all_drop_curves, axis=0)
    mean_curve = np.mean(fold_mean_curve, axis=0)
    # std_curve = np.std(all_drop_curves, axis=0, ddof=1)  # ddof parameter is set to 1 to return the sample std
    std_curve = scipy.stats.sem(fold_mean_curve, axis=0)

    return mean_curve, std_curve


def make_feature_dropping_report(bird_id='z007', session='day-2016-09-09'):
    warnings.filterwarnings("ignore", category=UserWarning)  # So that it doesn't print warnings until oblivion

    zdata = ImportData(bird_id=bird_id, session=session)

    # Get the Bird Specific Machine Learning Meta Data
    bad_channels = all_bad_channels[bird_id]
    drop_temps = all_drop_temps[bird_id]

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

    # Grab the Neural Activity Centered on Each event
    set_window = (-10, 0)

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
    mean_curve_pow, std_curve_pow = random_feature_drop_multi_narrow_chunk(event_data=chunk_events_balanced_pow,
                                                                           ClassObj=rand_obj, drop_temps=drop_temps,
                                                                           k_folds=5, seed=None, verbose=True)
    _save_numpy_data(data=mean_curve_pow, data_name="mean_curve_pow", bird_id=bird_id, session=session,
                     destination=channel_drop_path, make_parents=True, verbose=True)
    _save_numpy_data(data=std_curve_pow, data_name="std_curve_pow", bird_id=bird_id, session=session,
                     destination=channel_drop_path, make_parents=True, verbose=True)

    # Run Analysis on Only Phase
    mean_curve_phase, std_curve_phase = random_feature_drop_multi_narrow_chunk(event_data=chunk_events_balanced_phase,
                                                                               ClassObj=rand_obj, drop_temps=drop_temps,
                                                                               k_folds=5, seed=None, verbose=True)

    _save_numpy_data(data=mean_curve_phase, data_name="mean_curve_phase", bird_id=bird_id, session=session,
                     destination=channel_drop_path, make_parents=True, verbose=True)
    _save_numpy_data(data=std_curve_phase, data_name="std_curve_phase", bird_id=bird_id, session=session,
                     destination=channel_drop_path, make_parents=True, verbose=True)

    # Run Analysis on Both Features Independently
    mean_curve_both, std_curve_both = random_feature_drop_multi_narrow_chunk_both(power_data=chunk_events_balanced_pow,
                                                                                  phase_data=chunk_events_balanced_phase,
                                                                                  ClassObj=rand_obj,
                                                                                  drop_temps=drop_temps, k_folds=5,
                                                                                  seed=None, verbose=True)
    _save_numpy_data(data=mean_curve_both, data_name="mean_curve_both", bird_id=bird_id, session=session,
                     destination=channel_drop_path, make_parents=True, verbose=True)
    _save_numpy_data(data=std_curve_both, data_name="std_curve_both", bird_id=bird_id, session=session,
                     destination=channel_drop_path, make_parents=True, verbose=True)


##### Re-implementation for the paper

def random_feature_drop_sel_narrow_chunk(event_data, ClassObj, drop_temps, sel_freq, k_folds=5, seed=None,
                                         verbose=False):
    """ Runs the Random Channel Feature Dropping algorithm on a set of pre-processed data (defaults to 5K repeats)

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
        If True the funtion will print out useful information for user as it runs, defaults to False.

    Returns
    -------
    """

    # 1.) Make Array for Holding all of the feature dropping curves
    nested_dropping_curves = []  # np.zeros([])

    # 2.) Create INDEX of all instances of interests : create_discrete_index()
    label_identities, label_index = cat.create_discrete_index(event_data=event_data)
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
        # 4.1) Get template set/training : ml_selector(event_data, identity_index, label_index, sel_instances)
        sel_train = cat.ml_selector(event_data=event_data, identity_index=label_identities, label_index=label_index,
                                    sel_instances=X_train, )

        # 4.1) Get test set : ml_selector()
        sel_test = cat.ml_selector(event_data=event_data, identity_index=label_identities, label_index=label_index,
                                   sel_instances=X_test)

        # 5.) Use template/training set to make template : make_templates(event_data)
        templates = cat.make_templates(event_data=sel_train)

        # 5.2) Remove Template that aren't needed from train
        templates = np.delete(templates, drop_temps, axis=0)

        # 6.1) Use template/training INDEX and template to create Training Pearson Features : pearson_extraction()
        train_pearson_features = cat.pearson_extraction(event_data=sel_train, templates=templates)

        # 6.2) Use test INDEX and template to create Test Pearson Features : pearson_extraction()
        test_pearson_features = cat.pearson_extraction(event_data=sel_test, templates=templates)

        # 7.1) Reorganize Test Set into Machine Learning Format : ml_order_pearson()
        ml_trials_train, ml_labels_train = cat.ml_order(extracted_features_array=train_pearson_features)

        # 7.2) Get Ledger of the Features
        num_freqs, num_chans, num_temps = np.shape(train_pearson_features[0][0])  # Get the shape of the Feature data
        ordered_index = cat.make_feature_id_ledger(num_freqs=num_freqs, num_chans=num_chans, num_temps=num_temps)

        # 7.3) Reorganize Training Set into Machine Learning Format : ml_order_pearson()
        ml_trials_test, ml_labels_test = cat.ml_order(extracted_features_array=test_pearson_features)

        repeated_freq_curves = []
        test_list = list(np.arange(num_chans))
        random.seed(0)
        for index in range(5000):
            drop_order = random.sample(test_list, k=len(test_list))
            fold_frequency_curves = []
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
                ordered_index_cp = np.delete(ordered_index_cp, full_drop,
                                             axis=0)  # Remove features from other frequencies

                # 8.) Perform Nested Feature Dropping with K-Fold Cross Validation
                nested_drop_curve = cat.ordered_feature_dropping(train_set=ml_trials_train_freq,
                                                                 train_labels=ml_labels_train,
                                                                 test_set=ml_trials_test_freq,
                                                                 test_labels=ml_labels_test,
                                                                 ordered_index=ordered_index_cp, drop_type='channel',
                                                                 Class_Obj=ClassObj, order=drop_order, verbose=False)
                fold_frequency_curves.append(nested_drop_curve)  # For each Individual Frequency Band
            if verbose:
                if index % 100 == 0:
                    print('on loop' + str(index))

            repeated_freq_curves.append(fold_frequency_curves)  # Exhaustive Feature Dropping

        nested_dropping_curves.append(repeated_freq_curves)  # All of the Curves

    # 9.) Combine all curve arrays to one array
    all_drop_curves = np.array(nested_dropping_curves)  # (folds, 5K Repeats, frequencies, num_dropped, 1)

    # 10.) Calculate curve metrics
    fold_mean_curve = np.mean(all_drop_curves, axis=0)
    mean_curve = np.mean(fold_mean_curve, axis=0)
    std_curve = np.std(fold_mean_curve, axis=0, ddof=1)  # ddof parameter is set to 1 to return the sample std
    # std_curve = scipy.stats.sem(fold_mean_curve, axis=0)

    return mean_curve, std_curve


def random_feature_drop_sel_narrow_chunk_both(power_data, phase_data, ClassObj, drop_temps, sel_freq, k_folds=5,
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
    nested_dropping_curves = []  # np.zeros([])

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

        repeated_freq_curves = []
        test_list = list(np.arange(num_chans))
        random.seed(0)
        for index in range(5000):
            drop_order = random.sample(test_list, k=len(test_list))
            fold_frequency_curves = []
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
                ordered_index_cp = np.delete(ordered_index_cp, full_drop,
                                             axis=0)  # Remove features from other frequencies

                # 8.) Perform Nested Feature Dropping with K-Fold Cross Validation
                nested_drop_curve = cat.ordered_feature_dropping(train_set=ml_trials_train_freq,
                                                                 train_labels=ml_labels_train,
                                                                 test_set=ml_trials_test_freq,
                                                                 test_labels=ml_labels_test,
                                                                 ordered_index=ordered_index_cp, drop_type='channel',
                                                                 Class_Obj=ClassObj, order=drop_order, verbose=False)
                fold_frequency_curves.append(nested_drop_curve)  # For each Individual Frequency Band

            if verbose:
                if index % 100 == 0:
                    print('on loop' + str(index))

            repeated_freq_curves.append(fold_frequency_curves)  # Exhaustive Feature Dropping
        nested_dropping_curves.append(repeated_freq_curves)  # All of the Curves

    # 9.) Combine all curve arrays to one array
    all_drop_curves = np.array(nested_dropping_curves)  # (folds, frequencies, num_dropped, 1)

    # 10.) Calculate curve metrics
    fold_mean_curve = np.mean(all_drop_curves, axis=0)
    mean_curve = np.mean(fold_mean_curve, axis=0)
    std_curve = np.std(fold_mean_curve, axis=0, ddof=1)  # ddof parameter is set to 1 to return the sample std
    # std_curve = scipy.stats.sem(fold_mean_curve, axis=0)

    return mean_curve, std_curve


# Hard-coded from the results
best_bin_width = {"day-2016-06-03": [90, 185, 90, 85, 90],
                  "day-2016-06-05": [190, 125, 35, 160, 70],
                  "day-2016-09-10": [170, 75, 90, 25, 30],
                  "day-2016-09-11": [195, 170, 70, 20, 70],
                  "day-2016-06-19": [175, 170, 85, 50, 35],
                  "day-2016-06-21": [150, 170, 70, 60, 40]}

best_offset = {"day-2016-06-03": [-10, -5, 0, 0, 0],
               "day-2016-06-05": [-25, -5, 0, -20, 0],
               "day-2016-09-10": [-35, -5, -10, -10, 0],
               "day-2016-09-11": [-5, -5, -15, 0, -15],
               "day-2016-06-19": [-50, 0, -10, -5, -5],
               "day-2016-06-21": [-10, -15, -30, 0, 0]}

Best_Accuracy = {"day-2016-06-03": [0.8545, 0.8945, 0.87272, 0.8436, 0.7782],
                 "day-2016-06-05": [0.7022, 0.7288, 0.6044, 0.5822, 0.5333],
                 "day-2016-09-10": [0.9680, 0.9800, 0.9960, 0.9559, 0.9399],
                 "day-2016-09-11": [0.9565, 0.9710, 0.9420, 0.9014, 0.8753],
                 "day-2016-06-19": [0.8476, 0.9333, 0.8762, 0.8, 0.6857],
                 "day-2016-06-21": [0.6060, 0.7030, 0.7454, 0.7030, 0.5636]}


def make_best_feature_dropping_report(bird_id='z007', session='day-2016-09-09'):
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
    fc_lo = [4, 8, 25, 35, 50]
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
        mean_curve_pow, std_curve_pow = random_feature_drop_sel_narrow_chunk(event_data=chunk_events_balanced_pow,
                                                                             ClassObj=rand_obj, drop_temps=drop_temps,
                                                                             sel_freq=freq_num,
                                                                             k_folds=5, seed=None, verbose=True)
        _save_numpy_data(data=mean_curve_pow, data_name="mean_curve_pow" + str(freq_num) + "_2", bird_id=bird_id,
                         session=session, destination=channel_drop_path, make_parents=True, verbose=True)
        _save_numpy_data(data=std_curve_pow, data_name="std_curve_pow" + str(freq_num) + "_2", bird_id=bird_id,
                         session=session, destination=channel_drop_path, make_parents=True, verbose=True)

        # Run Analysis on Only Phase
        mean_curve_phase, std_curve_phase = random_feature_drop_sel_narrow_chunk(event_data=chunk_events_balanced_phase,
                                                                                 ClassObj=rand_obj,
                                                                                 drop_temps=drop_temps,
                                                                                 sel_freq=freq_num,
                                                                                 k_folds=5, seed=None, verbose=True)

        _save_numpy_data(data=mean_curve_phase, data_name="mean_curve_phase" + str(freq_num) + "_2", bird_id=bird_id,
                         session=session,
                         destination=channel_drop_path, make_parents=True, verbose=True)
        _save_numpy_data(data=std_curve_phase, data_name="std_curve_phase" + str(freq_num) + "_2", bird_id=bird_id,
                         session=session, destination=channel_drop_path, make_parents=True, verbose=True)

        # Run Analysis on Both Features Independently
        mean_curve_both, std_curve_both = random_feature_drop_sel_narrow_chunk_both(
            power_data=chunk_events_balanced_pow, phase_data=chunk_events_balanced_phase, ClassObj=rand_obj,
            drop_temps=drop_temps, sel_freq=freq_num, k_folds=5, seed=None, verbose=True)

        _save_numpy_data(data=mean_curve_both, data_name="mean_curve_both" + str(freq_num) + "_2", bird_id=bird_id,
                         session=session, destination=channel_drop_path, make_parents=True, verbose=True)
        _save_numpy_data(data=std_curve_both, data_name="std_curve_both" + str(freq_num) + "_2", bird_id=bird_id,
                         session=session, destination=channel_drop_path, make_parents=True, verbose=True)
