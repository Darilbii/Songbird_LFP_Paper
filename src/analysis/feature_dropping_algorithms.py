# from __future__ import absolute_import

import matplotlib.pyplot as plt
from itertools import count
import numpy as np
from scipy.stats import binom

import src.visualization.featdrop_vis as vis
from src.utils.organization import create_folder
from src.utils.paths import *

import BirdSongToolbox as tb
import BirdSongToolbox.Epoch_Analysis_Tools as bep
import BirdSongToolbox.feature_dropping_suite as fd

""" General Ground Rules with Algorithms (Under Development)

Save Results into a pandas dataframe to prevent having to repeat the same analysis when re-generating figures 
for presentation purposes
"""


# TODO: Reformat Algorithms to be more modular and not dependent on careful editing the same lines in different funcs


# Make Module Function to iterate over each Frequency Band

def run_narrowband_featdrop_strat(Days_Data, days_labels, days_onsets, label_instructions, Class_Obj, top, bottom,
                                  information_type: str, feature_type="Pearson", offset=0, tr_length=10, verbose=False):
    """ Narrow Bands one Frequency Band and runs a Channel Dropping Analysis (Dropping the worst Feature)

    Parameters:
    -----------
    Days_Data: class
        Instance of the Import_PrePd_Data() class.
    days_labels: list
        list of labels for all epochs for one day
        [Epoch] -> [Labels]
    days_onsets: list
        list of start and end times for all labels for one day
        [[Epochs]->[Start TIme] , [Epochs]->[End Time]]
    label_instructions: list
        list of labels and how they should be treated. If you use a nested list in the instructions the labels in
        this nested list will be treated as if they are the same label
    Class_Obj: class
        classifier object from the scikit-learn package

    feature_type="Pearson",
    offset=0,
    tr_length=10,

    top: int
        high frequency cutoff of the narrow band
    bottom: int
        low frequency cutood of the narrow band
    verbose = False

    Returns:
    --------
    test_drop: list
        list of accuracy values from the feature dropping code (values are floats)
    test_err_bar: list
        list of error values from the feature dropping code (values are floats)
    """

    assert information_type == 'amplitude' or information_type == 'phase' or information_type == 'both', \
        "output parameter can only be 'amplitude', 'phase', or 'both' not {output}"

    ##
    # Pre-Process Data
    Pipe = tb.Pipeline(Days_Data)

    # Run Pre-Process Steps
    Pipe.Define_Frequencies(([bottom], [top]))
    Pipe.Band_Pass_Filter(verbose=verbose)
    # Pipe_1.Re_Reference()

    if information_type == 'amplitude':
        Pipe.hilbert_amplitude()
    elif information_type == 'phase':
        Pipe.hilbert_phase()

    Pipe.Z_Score()

    Pipe.Pipe_end()

    print('check 1')

    # Prepare Data by grabbing the Epochs of First motifs
    days_data = bep.Full_Trial_LFP_Clipper(Neural=Pipe.Song_Neural,
                                           Sel_Motifs=Pipe.All_First_Motifs,
                                           Num_Freq=Pipe.Num_Freq,
                                           Num_Chan=Pipe.Num_Chan,
                                           Sn_Len=Pipe.Sn_Len,
                                           Gap_Len=Pipe.Gap_Len)

    print('check 2')

    test_drop, test_err_bar = bep.featdrop_module(dataset=days_data,
                                                  labels=days_labels,
                                                  onsets=days_onsets,
                                                  label_instructions=label_instructions,
                                                  Class_Obj=Class_Obj)
    return test_drop, test_err_bar


def run_multi_narrowband_analysis_strat(Days_Data, days_labels, days_onsets, label_instructions, Class_Obj,
                                        information_type: str, feature_type,
                                        offset, tr_length, tops, bottoms, chance, single=False, verbose=False):
    """ Runs through a list of narrow frequency bands and runs channel dropping analysis on them (Drops the Worst Feature)

    Parameters:
    -----------
    Days_Data: class
        Instance of the Import_PrePd_Data() class.
    days_labels: list
        list of labels for all epochs for one day
        [Epoch] -> [Labels]
    days_onsets: list
        list of start and end times for all labels for one day
        [[Epochs]->[Start TIme] , [Epochs]->[End Time]]
    label_instructions: list
        list of labels and how they should be treated. If you use a nested list in the instructions the labels in
        this nested list will be treated as if they are the same label
    Class_Obj: class
        classifier object from the scikit-learn package

    feature_type="Pearson",
    offset=0,
    tr_length=10

    tops: list
        list of high frequency cutoff of the narrow band
    bottoms: list
        list of low frequency cutood of the narrow band
    verbose = False

    Returns:
    --------

    """

    assert information_type == 'amplitude' or information_type == 'phase' or information_type == 'both', \
        "output parameter can only be 'amplitude', 'phase', or 'both' not {output}"

    drop_curves = []
    err_bars = []

    # Designate Where these types of Figures will be Save and What Hierarchical Structure it will use
    multi_strat_fig_dir = f"FeatureDropping/Stratified/{Days_Data.bird_id}/{Days_Data.date}/"

    # Create Path to Figure Location if it doesn't already Exist
    multi_strat_fig_path = create_folder(base_path=FIGURES_DIR, directory=multi_strat_fig_dir, rtn_path=True)

    # 1: Iterate over all of the Narrow Bands and Run Channel Dropping Analysis on them
    for index, top, bottom in zip(count(), tops, bottoms):
        drop_curve, err_bar = run_narrowband_featdrop_strat(Days_Data=Days_Data, days_labels=days_labels,
                                                            days_onsets=days_onsets,
                                                            label_instructions=label_instructions,
                                                            Class_Obj=Class_Obj, feature_type=feature_type,
                                                            offset=offset,
                                                            tr_length=tr_length, top=top, bottom=bottom,
                                                            information_type=information_type,
                                                            verbose=verbose)  # Loc: # 1: Iterate over all of ...

        drop_curves.append(drop_curve)  # Append each Narrow Band Result to a list
        err_bars.append(err_bar)  # Append each Error Bars for results to a list

        if single:
            # 2: Create and Save the Single Narrow Frequency Band Plots
            title_single = f"Channel Dropping Curve for {bottom}-{top} Hz ({feature_type})"

            # Create Path to Single Plots
            create_folder(base_path=multi_strat_fig_path, directory='/single')
            file_name_single = f"/single/featdrop_{feature_type}_channel_single_{bottom}-{top}_{label_instructions}_offset_{offset}_tr-len_{tr_length}"

            vis.plot_featdrop_single(drop_curve=drop_curves[index], chance_level=.5, font=20, title_font=30,
                                     title=title_single, single=True, verbose=verbose)

            plt.savefig(str(multi_strat_fig_path) + file_name_single + '.png', format='png', dpi=100)
            # plt.savefig(str(multi_strat_fig_path) + file_name_single + '.svg', format='svg', dpi=1200)
            plt.show()

    # 3: Create and Save the Multi-Narrow Frequency Band Plots

    title_multi = f"Channel Dropping Curve {information_type}"

    file_name_multi = f"/{Days_Data.bird_id}_{Days_Data.date}_featdrop_{feature_type}_{information_type}_offset_{offset}_tr-len_{tr_length}_channel_multi_{label_instructions}"

    vis.plot_featdrop_multi(drop_curve_list=drop_curves, std_list=err_bars, Tops=tops, Bottoms=bottoms,
                            chance_level=chance, font=20,
                            title_font=30, title=title_multi, verbose=False)

    plt.savefig(str(multi_strat_fig_path) + file_name_multi + '.png', format='png', dpi=100)
    # plt.savefig(str(multi_fig_path) + file_name_multi + '.svg', format='svg', dpi=1200)
    plt.show()

    return drop_curves, err_bars


# TODO: Make feature_type="Pearson" parameter

def run_narrowband_featdrop_rand(Days_Data, days_labels, days_onsets, label_instructions, Class_Obj, top, bottom,
                                 information_type: str, feature_type="Pearson",
                                 offset=0, tr_length=10, k_folds=5, verbose=False):
    """ Narrow Bands one Frequency Band and runs a Channel Dropping Analysis on it using a Random Feature Selection Strategy

    Parameters:
    -----------
    Days_Data: class
        Instance of the Import_PrePd_Data() class.
    days_labels: list
        list of labels for all epochs for one day
        [Epoch] -> [Labels]
    days_onsets: list
        list of start and end times for all labels for one day
        [[Epochs]->[Start TIme] , [Epochs]->[End Time]]
    label_instructions: list
        list of labels and how they should be treated. If you use a nested list in the instructions the labels in
        this nested list will be treated as if they are the same label
    Class_Obj: class
        classifier object from the scikit-learn package

    #feature_type="Pearson",
    information_type
    offset=0,
    tr_length=10,


    k_folds: int
        Number of Folds to Split between Template | Train/Test sets, defaults to 5,
    top: int
        high frequency cutoff of the narrow band
    bottom: int
        low frequency cutood of the narrow band
    verbose = False

    Returns:
    --------
    test_drop: list
        list of accuracy values from the feature dropping code (values are floats)
    test_err_bar: list
        list of error values from the feature dropping code (values are floats)
    """

    assert information_type == 'amplitude' or information_type == 'phase' or information_type == 'both', \
        "output parameter can only be 'amplitude', 'phase', or 'both' not {output}"

    # TODO: feature_type parameter will eventually be used for something
    if verbose:
        print(feature_type)
    ##
    # Pre-Process Data
    Pipe = tb.Pipeline(Days_Data)

    # Run Pre-Process Steps
    Pipe.Define_Frequencies(([bottom], [top]))
    Pipe.Band_Pass_Filter(verbose=verbose)

    if information_type == 'amplitude':
        Pipe.hilbert_amplitude()
    elif information_type == 'phase':
        Pipe.hilbert_phase()

    # Pipe_1.Re_Reference()
    Pipe.Z_Score()

    Pipe.Pipe_end()

    # print('check 1')

    # Prepare Data by grabbing the Epochs of First motifs
    days_data = bep.Full_Trial_LFP_Clipper(Neural=Pipe.Song_Neural,
                                           Sel_Motifs=Pipe.All_First_Motifs,
                                           Num_Freq=Pipe.Num_Freq,
                                           Num_Chan=Pipe.Num_Chan,
                                           Sn_Len=Pipe.Sn_Len,
                                           Gap_Len=Pipe.Gap_Len)

    # print('check 2')

    test_drop, test_err_bar = fd.random_feat_drop_analysis(full_trials=days_data, all_labels=days_labels,
                                                           starts=days_onsets, label_instructions=label_instructions,
                                                           Class_Obj=Class_Obj, offset=offset, tr_length=tr_length,
                                                           k_folds=k_folds, slide=None, step=False, seed=None,
                                                           verbose=verbose)

    return test_drop, test_err_bar


# Run Multi-Narrowband Analysis using new random Functions


def run_multi_narrowband_analysis_rand(Days_Data, days_labels, days_onsets, label_instructions, Class_Obj,
                                       information_type: str, feature_type, offset, tr_length, tops, bottoms,
                                       chance=None, single=False, verbose=False):
    """Iterate over narrow freq-bands running random channel dropping analysis and saves figures of analysis

    Parameters:
    -----------
    Days_Data: class
        Instance of the Import_PrePd_Data() class.
    days_labels: list
        list of labels for all epochs for one day
        [Epoch] -> [Labels]
    days_onsets: list
        list of start and end times for all labels for one day
        [[Epochs]->[Start TIme] , [Epochs]->[End Time]]
    label_instructions: list
        list of labels and how they should be treated. If you use a nested list in the instructions the labels in
        this nested list will be treated as if they are the same label
    Class_Obj: class
        classifier object from the scikit-learn package

    information_type: ['amplitude', 'phase', 'both']
    feature_type="Pearson",
    offset=0,
    tr_length=10

    tops: list
        list of high frequency cutoff of the narrow band
    bottoms: list
        list of low frequency cutood of the narrow band
    verbose = False

    Returns:
    --------
    drop_curves: list
        dropping results of the analysis (List of Lists)
    err_bars: list
        Standard Deviations of the results (List of Lists)


    """

    if not isinstance(chance, int) or isinstance(chance, float):
        num_samples = bep.extracted_labels_counter(all_labels=days_labels, starts=days_onsets[0],
                                                   label_instructions=label_instructions, offset=offset,
                                                   epoch_length=len(Days_Data.Song_Neural[0][:, 0]),
                                                   tr_length=tr_length)
        chance = get_chance(num_samples=num_samples, num_classes=len(label_instructions))

    drop_curves = []
    err_bars = []

    # Designate Where these types of Figures will be Save and What Hierarchical Structure it will use
    multi_rand_fig_dir = f"FeatureDropping/Random/{Days_Data.bird_id}/{Days_Data.date}/"

    # Create Path to Figure Location if it doesn't already Exist
    multi_rand_fig_path = create_folder(base_path=FIGURES_DIR, directory=multi_rand_fig_dir, rtn_path=True)

    # 1: Iterate over all of the Narrow Bands and Run Channel Dropping Analysis on them

    for index, top, bottom in zip(count(), tops, bottoms):
        drop_curve, err_bar = run_narrowband_featdrop_rand(Days_Data=Days_Data, days_labels=days_labels,
                                                           days_onsets=days_onsets,
                                                           label_instructions=label_instructions,
                                                           Class_Obj=Class_Obj, information_type=information_type,
                                                           feature_type=feature_type,
                                                           offset=offset,
                                                           tr_length=tr_length, top=top, bottom=bottom, verbose=verbose)
        # Loc: # 1: Iterate over all of ...

        drop_curves.append(drop_curve)  # Append each Narrow Band Result to a list
        err_bars.append(err_bar)  # Append each Error Bar for results to a list

        # NOTE: This Parameter may not need to be used
        if single:
            # 2: Create and Save the Single Narrow Frequency Band Plots
            title_single = f"Channel Dropping Curve for {bottom}-{top} Hz ({feature_type})"

            # Create Path to Single Plots
            create_folder(base_path=multi_rand_fig_path, directory='/single')
            file_name_single = f"/single/featdrop_{feature_type}_channel_single_{bottom}-{top}_{label_instructions}_offset_{offset}_tr-len_{tr_length}"

            vis.plot_featdrop_single(drop_curve=drop_curves[index], chance_level=chance, font=20, title_font=30,
                                     title=title_single, single=True, verbose=verbose)

            plt.savefig(str(multi_rand_fig_path) + file_name_single + '.png', format='png', dpi=100)
            # plt.savefig(str(multi_fig_path) + file_name_single + '.svg', format='svg', dpi=1200)
            plt.show()

    # 3: Create and Save the Multi-Narrow Frequency Band Plots

    title_multi = f"Channel Dropping Curve {information_type}"

    file_name_multi = f"/{Days_Data.bird_id}_{Days_Data.date}_featdrop_{feature_type}_{information_type}_offset_{offset}_tr-len_{tr_length}_channel_multi_{label_instructions}"

    vis.plot_featdrop_multi(drop_curve_list=drop_curves, std_list=err_bars, Tops=tops, Bottoms=bottoms,
                            chance_level=chance, font=20,
                            title_font=30, title=title_multi, verbose=False)

    plt.savefig(str(multi_rand_fig_path) + file_name_multi + '.png', format='png', dpi=100)
    # plt.savefig(str(multi_fig_path) + file_name_multi + '.svg', format='svg', dpi=1200)
    plt.show()

    return drop_curves, err_bars


# Creates a folder in the current directory called data
#####################################################


def get_chance(num_samples, num_classes, alpha=0.05, bon_correct=1):
    """ Calculate statistically significant classifier performance for data set with limited samples

    Parameters:
    ----------
    num_samples: array-like, (can be int)
        number of samples in data (assumed to be balanced)
    num_classes: int
        number of classes
    alpha: float
        significance level given by z/n or the ratio of tolerated false positives, defaults to 0.05
        (z: the number of observations correctly classified by chance, n: the number of all observations)
    bon_correct: int
        The Bonferroni Correction. Set equal to the number of 'Test' being
        for more information visit: https://en.wikipedia.org/wiki/Bonferroni_correction

    Returns:
    --------
    base: float
        Threshold for statistically significant classification rate, range [0, 1]
    """
    base = np.divide(binom.ppf(1 - (alpha / bon_correct), num_samples, 1. / num_classes), num_samples)
    return base


#####################################################


def make_report_multi_narrowband_analysis_rand(Days_Data, days_labels, days_onsets, label_instructions, Class_Obj,
                                               feature_type, offset, tr_length, tops, bottoms,
                                               single=False, verbose=False):
    # multi_fig_folder = f"FeatureDropping/Random/{Days_Data.bird_id}/{Days_Data.date}/"
    #
    # #TODO:
    # # Base path is always ~/Songbird-LFP-Paper/reports/
    # create_folder(base_path=FIGURES_DIR, directory=multi_fig_folder)

    num_samples = bep.extracted_labels_counter(all_labels=days_labels, starts=days_onsets[0],
                                               label_instructions=label_instructions,
                                               epoch_length=len(Days_Data.Song_Neural[0][:, 0]),
                                               offset=offset, tr_length=tr_length)

    chance = get_chance(num_samples=num_samples, num_classes=len(label_instructions))

    for signal_info in ['amplitude', 'phase', 'both']:
        all_drop_curves, all_err_bars = run_multi_narrowband_analysis_rand(Days_Data, days_labels, days_onsets,
                                                                           label_instructions, Class_Obj,
                                                                           information_type=signal_info,
                                                                           feature_type=feature_type, offset=offset,
                                                                           tr_length=tr_length, tops=tops,
                                                                           bottoms=bottoms, chance=chance,
                                                                           single=single, verbose=verbose)
