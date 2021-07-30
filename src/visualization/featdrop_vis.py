# Functions for Visualizing the Results of Feature Dropping Analysis

import matplotlib.pyplot as plt
import numpy as np
from itertools import cycle, count
import matplotlib.patches as mpatches



#TODO: Make quick line or function to make the title for the plots and name for its save file
#TODO: Fix Error Bars: Change them to standard deviation


def plot_featdrop_single(drop_curve, chance_level, font=20, title_font=30, title="Place Holder", single=True, verbose=False):
    """Plots a single feature dropping cure
    
    Parameters:
    -----------
    
    
    single: bool
        if True it will make a new figure within the function, defaults to True
    """

    Len_Test = len(drop_curve)

    Test1 = np.arange(0, Len_Test, 1)
    Test2 = np.arange(0, Len_Test + 1, 1)

    if verbose:
        print("Chance Level is: ", chance_level)
    # Test1 = Test1[::-1]

    # Main Dropping Curve

    # fig= plt.figure(figsize=(15,15))
    
    if single:
        plt.figure(figsize=(7, 7))  # Create Figure and Set Size
    plt.plot(Test1[::-1], drop_curve, color='black', label='10 ms')  # Main Drop Curve
    # plt.errorbar(Test1, Syll_DC,  yerr= Syll_StdERR, color= 'black', linestyle=' ') # Error Bars
    # black_patch2 = mpatches.Patch(color='black', label='Bin Width = 10 ms')  # Set Patches

    # Plot Chance
    plt.plot(Test2, chance_level * np.ones(Test2.shape), '--k', linewidth=5)

    # Axis Labels
    plt.title(title, fontsize=title_font)
    plt.xlabel('No. of Channels', fontsize=font)
    plt.ylabel('Accuracy', fontsize=font)

    # Format Annotatitng Ticks
    plt.tick_params(axis='both', which='major', labelsize=font)
    plt.tick_params(axis='both', which='minor', labelsize=font)
    plt.ylim(0, 1.0)
    # plt.xlim(0,17)
#     plt.show()



# The Below function is a Concept and isn't functional

def plot_and_save():
    
    title = f"Channel Dropping Curve for {top}-{bottom} Hz ({feat_type})"
    
    file_name = "featdrop_pearson_channel_single_bottom_top"

# def featdrop_module(dataset, labels, onsets, label_instructions, Class_Obj):
#     """
#     Parameters:
#     -----------
#     dataset: ndarray
#         Array that is structured to work with the SciKit-learn Package
#         (n_samples, n_features)
#             n_samples = Num of Instances Total
#             n_features = Num_Ch * Num_Freq)
#     labels: ndarray
#         1-d array of Labels of the Corresponding n_samples
#         ( n_samples   x   1 )
#     onsets: list
#         [[Epochs]->[Start TIme] , [Epochs]->[End Time]]
#     label_instructions: list
#         list of labels and how they should be treated. If you use a nested list in the instructions the labels in
#         this nested list will be treated as if they are the same label
#     Class_Obj: class
#         classifier object from the scikit-learn package
#
#     Returns:
#     --------
#     dropping_curve: list
#         list of accuracy values from the feature dropping code (values are floats)
#     err_bars: list
#         list of error values from the feature dropping code (values are floats)
#     """
#
#     # """Modular src to create a single"""
#
#     ## 2. Split into [Template] / [Train/Test] Sets
#     num_clippings = np.arange(len(labels))
#     train, test, _, _ = train_test_split(num_clippings, num_clippings, test_size=0.33, random_state=42)
#
#     print("train set:", train)
#     train_set, train_labels, train_starts = bep.Convienient_Selector(Features=dataset,
#                                                                  Labels=labels,
#                                                                  Starts=onsets[0],
#                                                                  Sel_index=train)
#
#     print("test set", test)
#     test_set, test_labels, test_starts = bep.Convienient_Selector(Features=dataset,
#                                                               Labels=labels,
#                                                               Starts=onsets[0],
#                                                               Sel_index=test)
#
#     ## 3. Create Pearson Template
#     _, hate, _, temps_int = bep.Classification_Prep_Pipeline(train_set,
#                                                          train_labels,
#                                                          train_starts,
#                                                          label_instructions,
#                                                          Offset=0,
#                                                          Tr_Length=10,
#                                                          Feature_Type='Pearson',
#                                                          Temps=None)  # ,
#     #                                                       Slide=Slide,
#     #                                                       Step=Step)
#
#     ## 4. Extract Pearson FIlters
#     ml_test_trials, ml_test_labels, test_ordered_index = bep.Classification_Prep_Pipeline(test_set,
#                                                                                       test_labels,
#                                                                                       test_starts,
#                                                                                       label_instructions,
#                                                                                       Offset=0,
#                                                                                       Tr_Length=10,
#                                                                                       Feature_Type='Pearson',
#                                                                                       Temps=temps_int)  # ,
#     #                                                                                   Slide=Slide,
#     #                                                                                   Step=Step)
#
#     ## 5. Run Feature Dropping
#     dropping_curve, err_bars = run_feature_dropping(Data_Set=ml_test_trials,
#                                                     Data_Labels=ml_test_labels,
#                                                     ordered_index=test_ordered_index,
#                                                     Class_Obj=Class_Obj,
#                                                     k_folds=2,
#                                                     verbose=True)
#     return dropping_curve, err_bars


def plot_featdrop_multi(drop_curve_list, std_list, Tops, Bottoms, chance_level, font=20, title_font=30, title="Place Holder", show_err=False, verbose=False):
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

    patch_list = []
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'tab:brown', 'tab:pink',
                    'tab:gray', 'tab:green', 'xkcd:dark olive green',
                    'xkcd:ugly yellow', 'xkcd:fire engine red', 'xkcd:radioactive green'])
    
    Len_Test = len(drop_curve_list[0]) # Make x-axis based off the First Curve

    ch_range = np.arange(0, Len_Test, 1)

    if verbose:
        print("Chance Level is: ", chance_level)
    # Test1 = Test1[::-1]

    # Main Dropping Curve

    # fig= plt.figure(figsize=(15,15))
    plt.figure(figsize=(7, 7))  # Create Figure and Set Size

    for index, curve, err_bar in zip(count(), drop_curve_list, std_list):
        if verbose:
            print('Making plot for curve: ', index)
            
        color = next(colors)
        plt.plot(ch_range[::-1], curve, color=color, label=' {:d} - {:d} Hz'.format(Tops[index], Bottoms[index]))  # Main Drop Curve
        # plt.errorbar(ch_range[::-1], curve, yerr= std_list, color='black', linestyle=' ') # Error Bars
        patch_list.append(mpatches.Patch(color=color, label=' {:d} - {:d} Hz'.format(Tops[index], Bottoms[index]))) # Set Patches

        if show_err:
            plt.fill_between(ch_range[::-1], np.asarray(curve) - np.asarray(err_bar), np.asarray(curve) + np.asarray(err_bar), color=color, alpha=0.2)

    # Plot Chance
    plt.plot(ch_range, chance_level * np.ones(ch_range.shape), '--k', linewidth=5)
    patch_list.append(mpatches.Patch(color='w', label=f'{round(chance_level,2)} Binomial Chance'))

    # Make Legend
    plt.legend(handles=patch_list, bbox_to_anchor=(.7, .3), loc=2, borderaxespad=0.)

    # Axis Labels
    plt.title(title, fontsize=title_font)
    plt.xlabel('No. of Channels', fontsize=font)
    plt.ylabel('Accuracy', fontsize=font)

    # Format Annotatitng Ticks
    plt.tick_params(axis='both', which='major', labelsize=font)
    plt.tick_params(axis='both', which='minor', labelsize=font)
    plt.ylim(0, 1.0)
