""" Make A Report of the Average ITC Analysis to find Consitent Frequency Bands """
from src.utils.paths import REPORTS_DIR
from src.analysis.context_utility import birds_context_obj
import src.analysis.hilbert_based_pipeline as hbp
from src.visualization.time_series_vis import plot_behavior
from src.analysis.ml_pipeline_utilities import all_chan_map, all_plot_maps, all_axis_orders, all_bad_channels

import BirdSongToolbox.free_epoch_tools as fet
from BirdSongToolbox.import_data import ImportData
from BirdSongToolbox.context_hand_labeling import label_focus_context, first_context_func
from BirdSongToolbox.behave.behave_utils import event_array_maker_chunk, get_events_rasters, repeat_events

import numpy as np
import scipy
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib.backends.backend_pdf import PdfPages
import pycircstat


def plot_z_itc(freqs, itc_z, color='black', ax=None):
    mean_z = np.mean(itc_z, axis=-1)
    sem = scipy.stats.sem(itc_z, axis=-1)

    if ax is None:
        plt.plot(freqs, mean_z, color=color)
        plt.fill_between(freqs, mean_z - sem, mean_z + sem, color=color, alpha=0.2)
        plt.axhline(y=2, color='red', linestyle='--')
        plt.axhline(y=1, color='blue', linestyle='--')

    else:
        ax.plot(freqs, mean_z, color=color)
        ax.fill_between(freqs, mean_z - sem, mean_z + sem, color=color, alpha=0.2)
        ax.axhline(y=2, color='red', linestyle='--')
        ax.axhline(y=1, color='blue', linestyle='--')


def plot_itc(ersp, event_times, fc_lo, fc_hi, cmap=None, **kwargs):
    """

    :param cmap:
    :param ersp:
    :param event_times:
    :param kwargs: Check the Seaborn Options (Lots of control here)
    :return:
    """

    if cmap is None:
        cmap = 'RdBu_r'

    ax = sns.heatmap(ersp, xticklabels=event_times, yticklabels=(fc_lo + fc_hi) / 2, cmap=cmap, **kwargs)

    ax.invert_yaxis()
    for ind, label in enumerate(ax.get_xticklabels()):
        if ind % 100 == 0:
            label.set_visible(True)
        else:
            label.set_visible(False)
    for ind, label in enumerate(ax.get_yticklabels()):
        if ind % 5 == 0:
            label.set_visible(True)
        else:
            label.set_visible(False)
    if ax is None:
        plt.show()


def get_itc_z(bird_id='z007', session='day-2016-09-09'):
    # The Order here doesn't matter so removing bad channels here won't hurt the analysis

    zdata = ImportData(bird_id=bird_id, session=session)

    chunk_labels_list, chunk_onsets_list = fet.get_chunk_handlabels(handlabels_list=zdata.song_handlabels)

    # Pre-Process the Data
    fc_lo = np.arange(3, 249, 2)
    fc_hi = np.arange(5, 251, 2)

    proc_data = hbp.itc_phase_chunk(neural_chunks=zdata.song_neural,
                                    fs=1000,
                                    l_freqs=fc_lo,
                                    h_freqs=fc_hi,
                                    verbose=True)

    # Helper Function to create the properly initialized context class
    testclass = birds_context_obj(bird_id=bird_id)

    # Get the Context Array for the Day's Data
    test_context = testclass.get_all_context_index_arrays(chunk_labels_list)

    # Select Labels Using Flexible Context Selection
    first_syll = label_focus_context(focus=1,
                                     labels=chunk_labels_list,
                                     starts=chunk_onsets_list[0],
                                     contexts=test_context,
                                     context_func=first_context_func)

    # Set the Context Windows

    first_window = (-500, 800)

    # Clip around Events of Interest
    all_firsts = fet.get_event_related_nd_chunk(chunk_data=proc_data, chunk_indices=first_syll,
                                                fs=1000, window=first_window)

    # Correct The Shape of the Data
    all_firsts = fet.event_shape_correction(all_firsts, original_dim=3)

    #     test_itc = pycircstat.resultant_vector_length(np.asarray(all_firsts), axis=0)
    test_itc_p, test_itc_z = pycircstat.rayleigh(np.asarray(all_firsts), axis=0)

    mean_z = np.mean(test_itc_z[:, :, 400:501], axis=-1)  # (freqs, channels)

    return mean_z


def make_itc_z_report(bird_id='z007', session='day-2016-09-09'):
    zdata = ImportData(bird_id=bird_id, session=session)

    bad_channels = all_bad_channels[bird_id]

    chunk_labels_list, chunk_onsets_list = fet.get_chunk_handlabels(handlabels_list=zdata.song_handlabels)

    # Pre-Process the Data
    fc_lo = np.arange(3, 249, 2)
    fc_hi = np.arange(5, 251, 2)

    proc_data = hbp.itc_phase_chunk(neural_chunks=zdata.song_neural, fs=1000, l_freqs=fc_lo, h_freqs=fc_hi,
                                    bad_channels=bad_channels, verbose=True)

    # Helper Function to create the properly initialized context class
    testclass = birds_context_obj(bird_id=bird_id)

    # Get the Context Array for the Day's Data
    test_context = testclass.get_all_context_index_arrays(chunk_labels_list)

    # Select Labels Using Flexible Context Selection
    first_syll = label_focus_context(focus=1, labels=chunk_labels_list, starts=chunk_onsets_list[0],
                                     contexts=test_context, context_func=first_context_func)

    # Set the Context Windows
    first_window = (-500, 800)

    # Clip around Events of Interest
    all_firsts = fet.get_event_related_nd_chunk(chunk_data=proc_data, chunk_indices=first_syll,
                                                fs=1000, window=first_window)

    # Correct The Shape of the Data
    all_firsts = fet.event_shape_correction(all_firsts, original_dim=3)

    test_itc = pycircstat.resultant_vector_length(np.asarray(all_firsts), axis=0)
    test_itc_p, test_itc_z = pycircstat.rayleigh(np.asarray(all_firsts), axis=0)

    # Create the Event Times
    first_event_times = fet.make_event_times_axis(first_window, fs=1000)

    # Create timeseries representing the labeled Events For all Chunks
    event_array_test2 = event_array_maker_chunk(labels_list=chunk_labels_list, onsets_list=chunk_onsets_list)

    # START PLOTTING RESULTS
    sel_chan_map = all_chan_map[bird_id]
    sel_plot_maps = all_plot_maps[bird_id]
    sel_axis_orders = all_axis_orders[bird_id]

    fig1, ax1 = plt.subplots(figsize=(20, 8))  # For PSDs Summary
    fig4, ax4 = plt.subplots(figsize=(20, 8))  # For PSDs Summary

    if test_itc.shape[1] > 16:
        fig2, ax2 = plt.subplots(6, 6, sharex=True, sharey=True, figsize=(50, 50))  # For PSDs Summary
        fig3, ax3 = plt.subplots(6, 6, sharex=True, sharey=True, figsize=(50, 50))  # For PSDs

    else:
        fig2, ax2 = plt.subplots(4, 4, sharex=True, sharey=True, figsize=(30, 30))  # For PSDs
        fig3, ax3 = plt.subplots(4, 4, sharex=True, sharey=True, figsize=(30, 30))  # For PSDs

    ax_2 = [ax_inst for ax_inst in ax2.flat]
    ax_3 = [ax_inst for ax_inst in ax3.flat]

    # fig 1: Make Behavior Raster for First
    first_events = get_events_rasters(data=event_array_test2, indices=first_syll, fs=1000, window=first_window)
    fill_events_first = repeat_events(first_events)

    # Plot the Behavior

    plot_behavior(fill_events_context=fill_events_first, context_event_times=first_event_times,
                  context_events=first_events, show_x=True, ax=ax1)

    # fig 2: Plot the ITC z-scores for each channel
    cbar_ax = fig2.add_axes([0.97, 0.11, 0.02, 0.78])
    z_max = np.max(test_itc_z)
    for i in range(test_itc.shape[1]):
        sel_chan = sel_plot_maps[i]
        sel_axis = sel_axis_orders[i]
        plot_itc(ersp=test_itc_z[:, sel_chan, :], event_times=first_event_times, fc_lo=fc_lo, fc_hi=fc_hi,
                 cmap='cubehelix', ax=ax_2[sel_axis], vmin=0, vmax=z_max, cbar_ax=cbar_ax)
        ax_2[sel_axis].set(title=f"Mean Z-Score in CH {sel_chan_map[i]}")  # Add the Channel number to the Z-Score Plots

    # fig 3: Plot the mean z-score values for each channel
    freqz = (fc_lo + fc_hi) / 2
    for i in range(test_itc.shape[1]):
        sel_chan = sel_plot_maps[i]
        sel_axis = sel_axis_orders[i]

        plot_z_itc(freqs=freqz, itc_z=test_itc_z[:, sel_chan, 400:501], ax=ax_3[sel_axis])
        ax_3[sel_axis].set_xlim(4, 250)
        ax_3[sel_axis].set(title=f"Mean Z-Score in CH {sel_chan_map[i]}")  # Add the Channel number to the Z-Score Plots

    # fig 4: Plot the Mean z-score across all channels
    mean_z = np.mean(test_itc_z[:, :, 400:501], axis=-1)
    plot_z_itc(freqs=freqz, itc_z=mean_z, ax=ax4)
    ax4.set_xlim(4, 250)

    # Create the Report
    report_name = 'ITC_Z_Score_' + bird_id + '_' + session + '_report.pdf'
    report_type_folder = REPORTS_DIR / 'Chunk_ITC_Z_Score'

    # Check if Folder Path Exists
    if not report_type_folder.exists():
        report_type_folder.mkdir(parents=True, exist_ok=True)

    report_location = report_type_folder / report_name

    # Create the PdfPages object to which we will save the pages:
    # The with statement makes sure that the PdfPages object is closed properly at
    # the end of the block, even if an Exception occurs.

    with PdfPages(report_location) as pdf:
        pdf.attach_note(f"Number of bouts = {first_events.shape[0]}", positionRect=[-100, -100, 0, 0])
        pdf.savefig(fig1)
        pdf.savefig(fig3)
        pdf.savefig(fig4)

    # Save the ITC Figure By Itself
    fig_file_name = 'ITC_Z-Score_' + bird_id + '_' + session + '_First_.png'
    figure_location = report_type_folder / fig_file_name
    fig2.savefig(figure_location, format='png')

