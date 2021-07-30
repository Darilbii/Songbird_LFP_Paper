from src.utils.paths import REPORTS_DIR
from src.analysis.chunk_feature_dropping_pearson import get_feature_dropping_results, \
    get_optimum_channel_dropping_results
from src.analysis.feature_dropping_algorithms import get_chance
from src.analysis.chunk_fix_channel_dropping import get_feature_dropping_corrections

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

# For the internal Function to get the data shape
import BirdSongToolbox.free_epoch_tools as fet
import src.analysis.ml_pipeline_utilities as mlpu
from src.analysis.ml_pipeline_utilities import all_bad_channels, all_drop_temps, all_label_instructions
from BirdSongToolbox.import_data import ImportData
import src.analysis.hilbert_based_pipeline as hbp


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
                        title="Place Holder", axis=None, show_legend=False, verbose=False):
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
    if not axis:
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
                               top=Tops[index], bottom=Bottoms[index], ax=axis)

        patch_list.append(mpatches.Patch(color=color, label=f'{Bottoms[index]} - {Tops[index]} Hz'))  # Set Patches

    if axis:
        pass
        # Plot Chance
        axis.plot(ch_range, chance_level * np.ones(ch_range.shape), '--k', linewidth=5)
        patch_list.append(mpatches.Patch(color='w', label=f'{round(chance_level,2)} Binomial Chance'))

        if show_legend:
            # Make Legend
            axis.legend(handles=patch_list, bbox_to_anchor=(1.05, .61), loc=2, borderaxespad=0.)

        #         # Axis Labels
        axis.set_title(title, fontsize=title_font)
        axis.set_xlabel('No. of Channels', fontsize=font)
        axis.set_ylabel('Accuracy', fontsize=font)

        #         # Format Annotatitng Ticks
        axis.tick_params(axis='both', which='major', labelsize=font)
        axis.tick_params(axis='both', which='minor', labelsize=font)
        axis.set_ylim(0, 1.0)
        axis.set_xlim(1, num_channels - 1)

    else:
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


def _get_the_chance(bird_id='z007', session='day-2016-09-09'):
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

    # Balance the sets

    chunk_events_balanced_pow = mlpu.balance_classes(chunk_events_power)

    data_shape = np.shape(chunk_events_balanced_pow)

    total_num_samples = data_shape[0] * data_shape[1]
    test_set = total_num_samples * .8

    binomial_chance = get_chance(num_samples=int(test_set),
                                 num_classes=data_shape[0],
                                 alpha=0.05,
                                 bon_correct=5)
    return round(binomial_chance, 2)


def make_channel_dropping_report(bird_id='z007', session='day-2016-09-09'):
    # Set the Frequency Bands to Be Used for Feature Extraction
    fc_lo = [4, 8, 25, 30, 50]
    fc_hi = [8, 12, 35, 50, 70]

    mean_curve_pow, std_curve_pow = get_feature_dropping_results(bird_id=bird_id, session=session,
                                                                 feat_type='pow', verbose=True)
    mean_curve_phase, std_curve_phase = get_feature_dropping_results(bird_id=bird_id, session=session,
                                                                     feat_type='phase', verbose=True)
    mean_curve_both, std_curve_both = get_feature_dropping_results(bird_id=bird_id, session=session,
                                                                   feat_type='both', verbose=True)
    binomial_chance = _get_the_chance(bird_id=bird_id, session=session)

    # Make the Figure

    fig, ax = plt.subplots(1, 3, sharex=False, sharey=True, figsize=(20, 10))  # For PSDs Summary

    plot_featdrop_multi(drop_curve_list=mean_curve_pow,
                        std_list=std_curve_pow,
                        Tops=fc_hi,
                        Bottoms=fc_lo,
                        chance_level=binomial_chance,
                        font=20,
                        title_font=30,
                        title="Power Only",
                        axis=ax[0],
                        verbose=False)

    plot_featdrop_multi(drop_curve_list=mean_curve_phase,
                        std_list=std_curve_phase,
                        Tops=fc_hi,
                        Bottoms=fc_lo,
                        chance_level=binomial_chance,
                        font=20,
                        title_font=30,
                        title="Phase Only",
                        axis=ax[1],
                        verbose=False)

    plot_featdrop_multi(drop_curve_list=mean_curve_both,
                        std_list=std_curve_both,
                        Tops=fc_hi,
                        Bottoms=fc_lo,
                        chance_level=binomial_chance,
                        font=20,
                        title_font=30,
                        title="Both",
                        axis=ax[2],
                        show_legend=True,
                        verbose=False)
    plt.suptitle(f"{bird_id} Channel Dropping Curve", size=35)

    # Create the Report
    report_type_folder = REPORTS_DIR / 'Phase_v_Power_Pearson'

    # Check if Folder Path Exists
    if not report_type_folder.exists():
        report_type_folder.mkdir(parents=True, exist_ok=True)

    # Save the ITC Figure By Itself
    fig_file_name = 'Phase_v_Power_' + bird_id + '_' + session + '.png'
    figure_location = report_type_folder / fig_file_name
    fig.savefig(figure_location, format='png')
    return


def fix_the_curves(mean_curve, std_curve, mean_correction, std_correction):
    for index, (mean, std) in enumerate(zip(mean_correction, std_correction)):
        mean_curve[index, 0] = mean
        std_curve[index, 0] = std
    return mean_curve, std_curve


def make_optimzed_channel_dropping_report(bird_id='z007', session='day-2016-09-09'):
    # Set the Frequency Bands to Be Used for Feature Extraction
    fc_lo = [4, 8, 25, 35, 50]
    fc_hi = [8, 12, 35, 50, 70]

    # Power
    mean_curve_pow, std_curve_pow = get_optimum_channel_dropping_results(bird_id=bird_id, session=session,
                                                                         feat_type='pow', verbose=True)
    # mean_correction_pow, std_correction_pow = get_feature_dropping_corrections(bird_id=bird_id, session=session,
    #                                                                            feat_type='pow', verbose=True)
    # mean_curve_pow, std_curve_pow = fix_the_curves(mean_curve=mean_curve_pow, std_curve=std_curve_pow,
    #                                                mean_correction=mean_correction_pow,
    #                                                std_correction=std_correction_pow)

    # Phase
    mean_curve_phase, std_curve_phase = get_optimum_channel_dropping_results(bird_id=bird_id, session=session,
                                                                             feat_type='phase', verbose=True)
    # mean_correction_phase, std_correction_phase = get_feature_dropping_corrections(bird_id=bird_id, session=session,
    #                                                                                feat_type='phase', verbose=True)
    # mean_curve_phase, std_curve_phase = fix_the_curves(mean_curve=mean_curve_phase, std_curve=std_curve_phase,
    #                                                    mean_correction=mean_correction_phase,
    #                                                    std_correction=std_correction_phase)

    # Both
    mean_curve_both, std_curve_both = get_optimum_channel_dropping_results(bird_id=bird_id, session=session,
                                                                           feat_type='both', verbose=True)
    # mean_correction_both, std_correction_both = get_feature_dropping_corrections(bird_id=bird_id, session=session,
    #                                                                            feat_type='both', verbose=True)
    # mean_curve_both, std_curve_both = fix_the_curves(mean_curve=mean_curve_both, std_curve=std_curve_both,
    #                                                mean_correction=mean_correction_both,
    #                                                std_correction=std_correction_both)

    binomial_chance = _get_the_chance(bird_id=bird_id, session=session)

    # Make the Figure

    fig, ax = plt.subplots(1, 3, sharex=False, sharey=True, figsize=(20, 10))  # For PSDs Summary

    plot_featdrop_multi(drop_curve_list=mean_curve_pow,
                        std_list=std_curve_pow,
                        Tops=fc_hi,
                        Bottoms=fc_lo,
                        chance_level=binomial_chance,
                        font=20,
                        title_font=30,
                        title="Power Only",
                        axis=ax[0],
                        verbose=False)

    plot_featdrop_multi(drop_curve_list=mean_curve_phase,
                        std_list=std_curve_phase,
                        Tops=fc_hi,
                        Bottoms=fc_lo,
                        chance_level=binomial_chance,
                        font=20,
                        title_font=30,
                        title="Phase Only",
                        axis=ax[1],
                        verbose=False)

    plot_featdrop_multi(drop_curve_list=mean_curve_both,
                        std_list=std_curve_both,
                        Tops=fc_hi,
                        Bottoms=fc_lo,
                        chance_level=binomial_chance,
                        font=20,
                        title_font=30,
                        title="Both",
                        axis=ax[2],
                        show_legend=True,
                        verbose=False)
    plt.suptitle(f"{bird_id} Channel Dropping Curve", size=35)

    # Create the Report
    report_type_folder = REPORTS_DIR / 'Phase_v_Power_Pearson'

    # Check if Folder Path Exists
    if not report_type_folder.exists():
        report_type_folder.mkdir(parents=True, exist_ok=True)

    # Save the ITC Figure By Itself
    fig_file_name = 'Phase_v_Power_' + bird_id + '_' + session + '_2.png'
    figure_location = report_type_folder / fig_file_name
    fig.savefig(figure_location, format='png')
    return
