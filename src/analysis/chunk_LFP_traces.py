""" Make A Report of the Average ITC Analysis to find Consitent Frequency Bands """
from src.utils.paths import REPORTS_DIR
from src.analysis.context_utility import birds_context_obj
import src.analysis.hilbert_based_pipeline as hbp
from src.analysis.ml_pipeline_utilities import all_chan_map, all_plot_maps, all_axis_orders, all_bad_channels

import BirdSongToolbox.free_epoch_tools as fet
from BirdSongToolbox.import_data import ImportData
from BirdSongToolbox.context_hand_labeling import label_focus_context, first_context_func

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.backends.backend_pdf import PdfPages



def run_lfp_trace_analysis(bird_id='z007', session='day-2016-09-09'):

    zdata = ImportData(bird_id=bird_id, session=session)
    bad_channels = all_bad_channels[bird_id]

    # Reshape Handlabels into Useful Format
    chunk_labels_list, chunk_onsets_list = fet.get_chunk_handlabels(handlabels_list=zdata.song_handlabels)

    fc_lo = [4, 8, 25, 30, 50]
    fc_hi = [8, 12, 35, 50, 70]

    proc_data = hbp.feature_extraction_chunk(neural_chunks=zdata.song_neural, fs=1000, l_freqs=fc_lo,  h_freqs=fc_hi,
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

    all_firsts = fet.event_shape_correction(all_firsts, original_dim=3)

    all_firsts = np.asarray(all_firsts)

    # Create the Event Times
    first_event_times = fet.make_event_times_axis(first_window, fs=1000)

     # all_firsts, first_event_times

    # START PLOTTING RESULTS
    sel_chan_map = all_chan_map[bird_id]
    sel_plot_maps = all_plot_maps[bird_id]
    sel_axis_orders = all_axis_orders[bird_id]

    all_figs = []

    for index, (freq_low, freq_high) in enumerate(zip(fc_lo, fc_hi)):
        if all_firsts.shape[2] > 16:
            fig, ax = plt.subplots(6, 6, sharex=False, sharey=False, figsize=(80, 50))
        else:
            fig, ax = plt.subplots(4, 4, sharex=False, sharey=False, figsize=(80, 50))

        ax = [ax_inst for ax_inst in ax.flat]

        for ch in range(all_firsts.shape[2]):
            sel_chan = sel_plot_maps[ch]
            sel_axis = sel_axis_orders[ch]
            ax[sel_axis].plot(first_event_times, np.transpose(all_firsts[:, index, sel_chan, :]))
            ax[sel_axis].axvline(x=0, color='black', linestyle='--', lw=3)
            #     ax[ch].set_xlim(-100, 100)
            #     ax[ch].set_xlim(-200,400)
            ax[sel_axis].set_xlim(-400, 800)
            ax[sel_axis].set(
                title=f"Trace on CH {sel_chan_map[ch]}")  # Add the Channel number to the Z-Score Plots
        fig.suptitle(f' Traces for {freq_low} to {freq_high} Hz', fontsize=30)
        all_figs.append(fig)

        # Create the Report
        report_name = 'Chunk_LFP_Traces' + bird_id + '_' + session + '_report.pdf'
        report_type_folder = REPORTS_DIR / 'Chunk_LFP_Traces'

        # Check if Folder Path Exists
        if not report_type_folder.exists():
            report_type_folder.mkdir(parents=True, exist_ok=True)

        report_location = report_type_folder / report_name

        # Create the PdfPages object to which we will save the pages:
        # The with statement makes sure that the PdfPages object is closed properly at
        # the end of the block, even if an Exception occurs.

        with PdfPages(report_location) as pdf:
            # pdf.attach_note(f"Number of bouts = {first_events.shape[0]}", positionRect=[-100, -100, 0, 0])
            for fig in all_figs:
                pdf.savefig(fig)


        # # Save the ITC Figure By Itself
        # fig_file_name = 'ITC_Z-Score_' + bird_id + '_' + session + '_First_.png'
        # figure_location = report_type_folder / fig_file_name
        # fig2.savefig(figure_location, format='png')
