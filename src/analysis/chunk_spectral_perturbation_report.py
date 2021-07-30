import src.analysis.hilbert_based_pipeline as hbp
from src.visualization.time_series_vis import plot_pretty_ersp
from src.analysis.context_utility import birds_context_obj
from src.utils.paths import REPORTS_DIR

import BirdSongToolbox.free_epoch_tools as fet
from BirdSongToolbox.import_data import ImportData
from BirdSongToolbox.context_hand_labeling import ContextLabels
from BirdSongToolbox.context_hand_labeling import label_focus_context, first_context_func, last_context_func, \
    mid_context_func
from BirdSongToolbox.behave.behave_utils import event_array_maker_chunk, get_events_rasters, repeat_events

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.backends.backend_pdf import PdfPages

# # Set the Colormap
# cmap2 = matplotlib.colors.ListedColormap(
#     ['black', 'red', 'orange', 'yellow', 'saddlebrown', 'blue', 'green', 'white', 'pink', 'purple'])
# cmap2.set_over('cyan')
# cmap2.set_under('cyan')
# bounds = [.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5]
# norm = matplotlib.colors.BoundaryNorm(bounds, cmap2.N)

# Hard Code the ploting order Using the Probe Index (4x4 Subplot Plotting)
# Order to Plot on a 4x4 (Probe Channel Name)
all_4x4_openephys = {'z007': [2, 10, 18, 26, 4, 12, 20, 28, 5, 13, 21, 29, 7, 15, 23, 31,
                              1, 9, 17, 25, 3, 11, 19, 27, 6, 14, 22, 30, 8, 16, 24, 32],
                     'z020': [6, 5, 9, 15, 3, 4, 10, 16, 1, 7, 13, 14, 2, 8, 12, 11],
                     'z017': [1, 7, 13, 14, 3, 4, 10, 16, 2, 8, 12, 11, 6, 5, 9, 15]}

# Order to Plot on a 4x4 (OpenEphys Designation)
all_4x4_mappings = {'z007': [3, 11, 19, 27, 5, 13, 21, 29, 6, 14, 22, 30, 8, 16, 24, 32,
                             2, 10, 18, 26, 4, 12, 20, 28, 7, 15, 23, 31, 9, 17, 25, 33],
                    'z020': [13, 12, 6, 5, 11, 15, 1, 7, 8, 14, 0, 4, 10, 9, 3, 2],
                    'z017': [8, 14, 0, 4, 11, 15, 1, 7, 10, 9, 3, 2, 13, 12, 6, 5]}


def plot_behavior_test(fill_events_context, context_event_times, context_events, ax=None):
    # Setup the Colorbar
    cmap2 = matplotlib.colors.ListedColormap(
        ['black', 'red', 'orange', 'yellow', 'saddlebrown', 'blue', 'green', 'white', 'pink', 'purple'])
    cmap2.set_over('cyan')
    cmap2.set_under('cyan')
    bounds = [.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5]
    norm = matplotlib.colors.BoundaryNorm(bounds, cmap2.N)

    # PlotBehavior Raster
    num_events = context_events.shape[0]
    max_len = fill_events_context.shape[0]
    bin_width = max_len / num_events
    y_labels = np.arange(0, num_events, 5, dtype=int)
    y_steps = np.linspace(0, y_labels[-1] * bin_width, len(y_labels), dtype=int)
    y_steps[1:] = y_steps[1:] - int(bin_width / 2)

    if ax is None:
        plt.imshow(fill_events_context, cmap=cmap2, Norm=norm, aspect="auto")
        plt.yticks(ticks=y_steps[1:], labels=y_labels[1:])
        plt.ylim(0, max_len)

    else:
        ax.imshow(fill_events_context, cmap=cmap2, Norm=norm, aspect="auto")
        ax.set_yticks(y_steps[1:])
        ax.set_yticklabels(y_labels[1:])
        ax.set_ylim(0, max_len)
        ax.set_xticks([])


def plot_16_channels(mean_activity, fill_events_context, context_event_times, context_events, mapping, chan_id,
                     title: str, repeat=False):
    if repeat:
        second = 16
    else:
        second = 0

    fig, ax_2d = plt.subplots(5, 4, sharex=True, sharey=False, figsize=(60, 40))

    v_min = 0
    v_max = 2

    ax = [ax_inst for ax_inst in ax_2d.flat]

    fig.suptitle(title, size=60)

    # Make A Single ColorBar
    cbar_ax = fig.add_axes([0.97, 0.11, 0.02, 0.78])

    for i in range(4):
        # Plot Behavior Raster
        plot_behavior_test(fill_events_context=fill_events_context, context_event_times=context_event_times,
                           context_events=context_events, ax=ax[i])

    for i, j in enumerate(np.arange(4, 20)):
        plot_pretty_ersp(mean_activity[:, mapping[i + second], :], context_event_times, vmin=v_min, vmax=v_max,
                         ax=ax[j],
                         cbar_ax=cbar_ax)
        ax[j].set(xlabel='Time (ms)', ylabel='Frequency (Hz)', title=f"Power in CH {chan_id[i + second]}")

    return fig


def make_spectral_report(bird_id='z007', session='day-2016-09-09'):
    zdata = ImportData(bird_id=bird_id, session=session)

    chunk_labels_list, chunk_onsets_list = fet.get_chunk_handlabels(handlabels_list=zdata.song_handlabels)

    # Set the Bandpass filters
    fc_lo = np.arange(3, 249, 2)
    fc_hi = np.arange(5, 251, 2)

    # Pre-Process the Data
    proc_data = hbp.spectral_perturbation_chunk(neural_chunks=zdata.song_neural,
                                                fs=1000,
                                                l_freqs=fc_lo,
                                                h_freqs=fc_hi,
                                                verbose=True)

    # Create instance of the Context Labels Class

    # bout_states = {8: 'not', 'I': 'not', 'C': 'not', 1: 'bout', 2: 'bout', 3: 'bout', 4: 'bout', 5: 'bout', 6: 'bout',
    #                7: 'bout',
    #                "BUFFER": "not", "X": "not"}
    # bout_transitions = {'not': 1, 'bout': 8}
    # bout_syll_length = 5
    testclass = birds_context_obj(bird_id=bird_id)
        # ContextLabels(bout_states, bout_transitions, full_bout_length=bout_syll_length)

    # Get the Context Array for the Day's Data

    test_context = testclass.get_all_context_index_arrays(chunk_labels_list)

    # Select Labels Using Flexible Context Selection
    first_syll = label_focus_context(focus=1, labels=chunk_labels_list, starts=chunk_onsets_list[0],
                                     contexts=test_context,
                                     context_func=first_context_func)
    last_syll = label_focus_context(focus=1, labels=chunk_labels_list, starts=chunk_onsets_list[0],
                                    contexts=test_context,
                                    context_func=last_context_func)
    mid_syll = label_focus_context(focus=1, labels=chunk_labels_list, starts=chunk_onsets_list[0],
                                   contexts=test_context,
                                   context_func=mid_context_func)

    # Set the Context Windows
    first_window = (-500, 800)
    last_window = (-100, 1000)
    mid_window = (-1000, 1800)

    # Clip around Events of Interest
    all_firsts = fet.get_event_related_nd_chunk(chunk_data=proc_data, chunk_indices=first_syll, fs=1000,
                                                window=first_window)
    all_lasts = fet.get_event_related_nd_chunk(chunk_data=proc_data, chunk_indices=last_syll, fs=1000,
                                               window=last_window)
    all_mids = fet.get_event_related_nd_chunk(chunk_data=proc_data, chunk_indices=mid_syll, fs=1000, window=mid_window)

    # Correct The Shape of the Data
    all_firsts = fet.event_shape_correction(all_firsts, original_dim=3)
    all_lasts = fet.event_shape_correction(all_lasts, original_dim=3)
    all_mids = fet.event_shape_correction(all_mids, original_dim=3)

    # Take Mean Across the Instances (Examplars)
    mean_first = np.mean(all_firsts, axis=0)
    mean_last = np.mean(all_lasts, axis=0)
    mean_mid = np.mean(all_mids, axis=0)

    # Create timeseries representing the labeled Events For all Chunks
    event_array_test2 = event_array_maker_chunk(labels_list=chunk_labels_list, onsets_list=chunk_onsets_list)

    ##
    # Make Behavior Rasters
    # Make Behavior Raster for First
    first_events = get_events_rasters(data=event_array_test2, indices=first_syll, fs=1000, window=first_window)
    fill_events_first = repeat_events(first_events)

    # Make Behavior Raster for Last
    last_events = get_events_rasters(data=event_array_test2, indices=last_syll, fs=1000, window=last_window)
    fill_events_last = repeat_events(last_events)

    # Make Behavior Raster formid
    mid_events = get_events_rasters(data=event_array_test2, indices=mid_syll, fs=1000, window=mid_window)
    fill_events_mid = repeat_events(mid_events)
    ##

    # Create the Event Times
    first_event_times = fet.make_event_times_axis(first_window, fs=1000)
    last_event_times = fet.make_event_times_axis(last_window, fs=1000)
    mid_event_times = fet.make_event_times_axis(mid_window, fs=1000)

    # Select the plotting mapping
    sel_mapping = all_4x4_mappings[bird_id]
    sel_chan_id = all_4x4_openephys[bird_id]

    # figs = []
    # fig_file_names = []

    # Create Directory to save the Figures
    # report_name = 'Spectral_Perturbation_' + bird_id + '_' + session + '_report.pdf'
    report_type_folder = REPORTS_DIR / 'Chunk_Spectral_Perturbation' / bird_id

    # Check if Folder Path Exists
    if not report_type_folder.exists():
        report_type_folder.mkdir(parents=True, exist_ok=True)

    # Plot First Behavior
    fig = plot_16_channels(mean_activity=mean_first, fill_events_context=fill_events_first,
                           context_event_times=first_event_times, context_events=first_events,
                           mapping=sel_mapping, chan_id=sel_chan_id, title='Start of All Bouts', repeat=False)
    fig_file_name = 'Spectral_Perturbation_' + bird_id + '_' + session + '_first_1.png'
    figure_location = report_type_folder / fig_file_name
    fig.savefig(figure_location, format='png')

    if mean_first.shape[1] > 16:
        fig = plot_16_channels(mean_activity=mean_first, fill_events_context=fill_events_first,
                               context_event_times=first_event_times, context_events=first_events,
                               mapping=sel_mapping, chan_id=sel_chan_id, title='Start of All Bouts', repeat=True)
        fig_file_name = 'Spectral_Perturbation_' + bird_id + '_' + session + '_first_2.png'
        figure_location = report_type_folder / fig_file_name
        fig.savefig(figure_location, format='png')

    # Plot Last Behavior
    fig = plot_16_channels(mean_activity=mean_last, fill_events_context=fill_events_last,
                           context_event_times=last_event_times, context_events=last_events,
                           mapping=sel_mapping, chan_id=sel_chan_id, title='End of All Bouts', repeat=False)
    fig_file_name = 'Spectral_Perturbation_' + bird_id + '_' + session + '_last_1.png'
    figure_location = report_type_folder / fig_file_name
    fig.savefig(figure_location, format='png')

    if mean_last.shape[1] > 16:
        fig = plot_16_channels(mean_activity=mean_last, fill_events_context=fill_events_last,
                               context_event_times=last_event_times, context_events=last_events,
                               mapping=sel_mapping, chan_id=sel_chan_id, title='End of All Bouts', repeat=True)
        fig_file_name = 'Spectral_Perturbation_' + bird_id + '_' + session + '_last_2.png'
        figure_location = report_type_folder / fig_file_name
        fig.savefig(figure_location, format='png')

    # Plot Middle Behavior
    fig = plot_16_channels(mean_activity=mean_mid, fill_events_context=fill_events_mid,
                           context_event_times=mid_event_times, context_events=mid_events,
                           mapping=sel_mapping, chan_id=sel_chan_id, title='During All Bouts', repeat=False)
    fig_file_name = 'Spectral_Perturbation_' + bird_id + '_' + session + '_mid_1.png'
    figure_location = report_type_folder / fig_file_name
    fig.savefig(figure_location, format='png')

    if mean_mid.shape[1] > 16:
        fig = plot_16_channels(mean_activity=mean_mid, fill_events_context=fill_events_mid,
                               context_event_times=mid_event_times, context_events=mid_events,
                               mapping=sel_mapping, chan_id=sel_chan_id, title='During All Bouts', repeat=True)
        fig_file_name = 'Spectral_Perturbation_' + bird_id + '_' + session + '_mid_2.png'
        figure_location = report_type_folder / fig_file_name
        fig.savefig(figure_location, format='png')

    # report_name = 'Spectral_Perturbation_' + bird_id + '_' + session + '_report.pdf'
    # report_type_folder = REPORTS_DIR / 'Chunk_Spectral_Perturbation' / bird_id

    # # Check if Folder Path Exists
    # if not report_type_folder.exists():
    #     report_type_folder.mkdir(parents=True, exist_ok=True)

    # for fig, figure_name in zip(figs, fig_file_names):
    #     figure_location = report_type_folder / figure_name
    #
    #     fig.savefig(figure_location, format='png')
    #     # fig.savefig(figure_location, dpi=300, papertype=None, format='png')

    # # Create the PdfPages object to which we will save the pages:
    # # The with statement makes sure that the PdfPages object is closed properly at
    # # the end of the block, even if an Exception occurs.
    # with PdfPages(report_location) as pdf:
    #     pdf.attach_note(f"Number of Bouts = {len(first_events)}", positionRect=[-100, -100, 0, 0])
    #
    #     for fig in figs:
    #         pdf.savefig(fig)
