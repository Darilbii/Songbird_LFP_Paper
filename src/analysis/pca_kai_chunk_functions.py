from src.analysis.ml_pipeline_utilities import all_chan_map, all_plot_maps, all_axis_orders, all_bad_channels
from src.utils.paths import REPORTS_DIR
from src.analysis.ml_pipeline_utilities import balance_classes

import BirdSongToolbox.free_epoch_tools as fet
from BirdSongToolbox.preprocess import common_average_reference
from BirdSongToolbox.import_data import ImportData

import numpy as np
import scipy
import matplotlib.pyplot as plt
from mne.time_frequency import psd_array_multitaper





def long_silence_finder(silence, all_labels, all_starts, all_ends, window):
    """ Checks if the Duration of the Silence Label is longer than the window and sets start equal to the middle of event

    Parameters
    ----------
    silence : str or int
        User defined Label to focus on
    all_labels : list
        List of all Labels corresponding to each Chunk in Full_Trials
        [Epochs]->[Labels]
    all_starts : list
        List of all Start Times corresponding to each Chunk in Full_Trials
        [Epochs]->[Start Time]
    all_ends : list
        List of all End Times corresponding to each Chunk in Full_Trials
        [Epochs]->[End Time]
    window : tuple | shape (start, end)
            Window (in ms) around event onsets, window components must be integer values

    Returns
    -------
    label_index : list
        List of all start frames of every instances of the label of focus
        [Num_Trials]->[Num_Exs]
    """

    label_index = []
    fs = 30  # Originally Sammpling is 30Khz

    window_len = len(np.arange(window[0], window[1])) * fs  # Length of the Window

    for starts, ends, labels in zip(all_starts, all_ends, all_labels):
        mid_starts = [start + ((end - start) / 2) for start, end, label in zip(starts, ends, labels) if
                      label == silence and (end - start) > window_len]
        label_index.append(mid_starts)
    return label_index


# # TODO: Update this To only be dependent on one source
# def balance_classes(neural_data):
#     """ Takes a List of Instances of the Time Series and Balances out all classes to be equal size
#     (Approach 1: All Classes Set to be Equal)
#
#     Parameters
#     ----------
#     neural_data : list | (classes, instances, channels, samples)
#         Neural Data to be used in PCA-PSD Analysis
#
#     Returns
#     -------
#     balanced_data : list | (classes, instances, channels, samples)
#         Randomly Rebalanced Neural Data to be used in PCA-PSD Analysis (All Sets are equal length)
#     """
#
#     balanced_data = neural_data  # Shallow Copy
#     group_sizes = [len(events) for events in neural_data]  # Number of Instances per Class
#
#     minimum = min(np.unique(group_sizes))  # Size of Smallest Class
#     focus_mask = [index for index, value in enumerate(group_sizes) if value > minimum]  # Index of Larger Classes
#
#     for needs_help in focus_mask:
#         big = len(neural_data[needs_help])
#         selected = random.sample(range(0, big), minimum)  # Select the instances to Use
#         balanced_data[needs_help] = neural_data[needs_help][selected]  # Reduce Instances to Those Selected
#
#     return balanced_data


def stack_instances(neural_data):
    """ Concatenates the  Neural Data to be used to calculate the power spectrum

    Parameters
    ----------
    neural_data : list | (classes, instances, channels, samples)
        Balanced Neural Data to be used in PCA-PSD Analysis (All Sets are equal length)

    Returns
    -------
    stacked_events : ndarray | (classes * instances, channels, samples)
        reshaped array of the Balanced Neural Data
    """

    holder = []

    for instances in neural_data:
        holder.append(instances)

    stacked_events = np.concatenate(holder, axis=0)

    return np.asarray(stacked_events)


from sklearn.decomposition import PCA
from scipy.signal import welch
from matplotlib.backends.backend_pdf import PdfPages


def run_pca_analysis_for_one_channel_repeatedly(Pxx_norm, Num_Epochs, ax1, ax2, ax3, ax4, channel):
    # Run PCA
    pca = PCA(n_components=.95)
    pca.fit(Pxx_norm[:, channel, :200])

    # Plot the Explained Variance with Number of Components
    ax1.plot(np.cumsum(pca.explained_variance_ratio_))
    ax1.set_xlabel('# PCs')
    ax1.set_ylabel('explained variance')
    ax1.set_title(f"Number of Principle Components Ch: {channel}")

    # Plot the first 3 Principle Components (PSCS)
    for i in range(3):
        ax2.plot(pca.components_[i, :], label=str(i + 1))
    ax2.legend()
    ax2.set_title(f"Ch: {channel} PSCs projected into PSD (x=Hz)")

    # Plot the Data based off the first two Principle Compoenents
    #     fig, ax = plt.subplots(1, 1)
    x_pca = pca.transform(Pxx_norm[:, channel, :200])
    ax3.scatter(x_pca[:Num_Epochs, 0], x_pca[:Num_Epochs, 1], alpha=0.2, color='red', label='Active')
    ax3.scatter(x_pca[Num_Epochs:, 0], x_pca[Num_Epochs:, 1], alpha=0.2, color='blue', label='Inactive')
    ax3.axis('equal')
    ax3.set(xlabel='component 1', ylabel='component 2',
            title=f"principal components for CH {channel}")  # ,xlim=(-5, 5), ylim=(-3, 3.1))
    ax3.legend()

    ax4.scatter(x_pca[:Num_Epochs, 1], x_pca[:Num_Epochs, 2], alpha=0.2, color='red', label='Active')
    ax4.scatter(x_pca[Num_Epochs:, 1], x_pca[Num_Epochs:, 2], alpha=0.2, color='blue', label='Inactive')
    ax4.axis('equal')
    ax4.set(xlabel='component 2', ylabel='component 3',
            title=f"principal components for CH {channel}")  # ,xlim=(-5, 5), ylim=(-3, 3.1))
    ax4.legend()


def plot_psds_for_one_channel_repeatedly(Pxx_concat, freqs, num_epochs, ax0, channel):
    # Plot the PSDs of the Trials
    ax0.semilogy(freqs, np.transpose(Pxx_concat[1:num_epochs, channel, :]), color='blue', alpha=.5)
    ax0.semilogy(freqs, np.transpose(Pxx_concat[num_epochs:-2, channel, :]), color='red', alpha=.5)
    ax0.semilogy(freqs, np.transpose(Pxx_concat[-1, channel, :]), color='blue', label='Vocally Active', alpha=.5)
    ax0.semilogy(freqs, np.transpose(Pxx_concat[0, channel, :]), color='red', label='Vocally Inactive', alpha=.5)
    ax0.set_title(f"PSDs for Ch: {channel}")
    ax0.set_xlim(0, 200)
    ax0.legend(loc='best')


def plot_summary_psd(Pxx_concat, freqs, num_trials, ax=None):
    mean_1 = np.mean(Pxx_concat[:num_trials, :], axis=0)
    mean_2 = np.mean(Pxx_concat[num_trials:, :], axis=0)
    # std_1 = np.std(Pxx_concat[:num_trials, :], axis=0)
    # std_2 = np.std(Pxx_concat[num_trials:, :], axis=0)
    err_1 = scipy.stats.sem(Pxx_concat[:num_trials, :], axis=0)
    err_2 = scipy.stats.sem(Pxx_concat[num_trials:, :], axis=0)

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))

    ax.set_yscale("log")  # log first
    ax.plot(freqs, np.transpose(mean_1), color='blue', label='Vocally Active', alpha=.5)
    ax.fill_between(freqs, mean_1 - err_1, mean_1 + err_1,
                    color='blue', alpha=0.2)
    ax.plot(freqs, np.transpose(mean_2), color='red', label='Vocally Inactive', alpha=.5)
    ax.fill_between(freqs, mean_2 - err_2, mean_2 + err_2,
                    color='red', alpha=0.2)
    # plt.set_title(f"PSDs for Ch: {index}")
    ax.legend(loc='best')
    ax.set_xlim(0, 200)
    ax.set_ylim(bottom=0.1)

    if ax is None:
        plt.show()


def make_axis_index(row, col):
    index = []
    for i in range(row):
        for j in range(col):
            index.append([i, j])
    return index


def make_multi_pca_prep(bird_id='z007', session='day-2016-09-09'):
    bad_channels = all_bad_channels[bird_id]  # Hard Code the Bad Channels

    z_data = ImportData(bird_id=bird_id, session=session)

    chunk_labels_list, chunk_onsets_list = fet.get_chunk_handlabels(handlabels_list=z_data.song_handlabels)

    set_window = (-500, 500)  # Set the Duration of the activity periods

    # Get Silence Periods
    silent_periods = long_silence_finder(silence=8, all_labels=chunk_labels_list, all_starts=chunk_onsets_list[0],
                                         all_ends=chunk_onsets_list[1], window=(-500, 500))
    # Find the Start of the First Syllables
    spec_events = fet.label_extractor(all_labels=chunk_labels_list, starts=chunk_onsets_list[0],
                                      label_instructions=[1])
    # Append Vocally Inactive the Vocally Active
    spec_events.append(silent_periods)

    # 1. Common Average Reference
    car_data = common_average_reference(z_data.song_neural, bad_channels=bad_channels)

    # Grab the Neural Activity Centered on Each event
    chunk_events = fet.event_clipper_nd(data=car_data, label_events=spec_events, fs=1000, window=set_window)

    # Determine the Number of Each event
    label_sets = [1, 8]

    balanced_events = balance_classes(neural_data=chunk_events)

    # Multitaper
    Pxx_song, freqs = psd_array_multitaper(np.asarray(balanced_events), sfreq=1000, fmin=0, fmax=200, bandwidth=15)

    Pxx_concat = stack_instances(Pxx_song)

    Pxx_norm = np.log(Pxx_concat / np.mean(Pxx_concat, axis=0)[None, :, :])

    return Pxx_norm, Pxx_concat, freqs


def make_welch_pca_prep(bird_id='z007', session='day-2016-09-09', nperseg=500):
    bad_channels = all_bad_channels[bird_id]  # Hard Code the Bad Channels

    z_data = ImportData(bird_id=bird_id, session=session)

    chunk_labels_list, chunk_onsets_list = fet.get_chunk_handlabels(handlabels_list=z_data.song_handlabels)

    set_window = (-500, 500)  # Set the Duration of the activity periods

    # Get Silence Periods
    silent_periods = long_silence_finder(silence=8, all_labels=chunk_labels_list, all_starts=chunk_onsets_list[0],
                                         all_ends=chunk_onsets_list[1], window=(-500, 500))
    # Find the Start of the First Syllables
    spec_events = fet.label_extractor(all_labels=chunk_labels_list, starts=chunk_onsets_list[0],
                                      label_instructions=[1])
    # Append Vocally Inactive the Vocally Active
    spec_events.append(silent_periods)

    # 1. Common Average Reference
    car_data = common_average_reference(z_data.song_neural, bad_channels=bad_channels)

    # Grab the Neural Activity Centered on Each event
    chunk_events = fet.event_clipper_nd(data=car_data, label_events=spec_events, fs=1000, window=set_window)

    # Determine the Number of Each event
    label_sets = [1, 8]

    balanced_events = balance_classes(neural_data=chunk_events)

    # Multitaper
    freqs, Pxx_song = welch(np.asarray(balanced_events), fs=1000, window='hann', nperseg=nperseg, scaling='spectrum',
                            axis=-1)

    Pxx_concat = stack_instances(Pxx_song)

    Pxx_norm = np.log(Pxx_concat / np.mean(Pxx_concat, axis=0)[None, :, :])

    return Pxx_norm, Pxx_concat, freqs


def make_multi_pca_report(bird_id: str, session: str):
    sel_chan_map = all_chan_map[bird_id]
    sel_plot_maps = all_plot_maps[bird_id]
    sel_axis_orders = all_axis_orders[bird_id]

    Pxx_norm, Pxx_concat, freqs = make_multi_pca_prep(bird_id=bird_id, session=session)

    if Pxx_norm.shape[1] > 16:
        fig, ax = plt.subplots(6, 6, sharex=True, sharey=True, figsize=(50, 50))  # For PSDs Summary
        fig0, ax0 = plt.subplots(6, 6, sharex=True, sharey=True, figsize=(50, 50))  # For PSDs
        # Run PCA and Plot Principle Components in a Huge ass Plot
        fig1, ax1 = plt.subplots(6, 6, sharex=True, sharey=True, figsize=(50, 50))
        # fig1.subplots_adjust(hspace=0.4, wspace=0.4)
        fig2, ax2 = plt.subplots(6, 6, sharey=True, figsize=(60, 40))
        fig3, ax3 = plt.subplots(6, 6, figsize=(50, 50))
        fig4, ax4 = plt.subplots(6, 6, figsize=(50, 50))
        # axis = make_axis_index(6, 6)

    else:
        fig, ax = plt.subplots(4, 4, sharex=True, sharey=True, figsize=(30, 30))  # For PSDs
        fig0, ax0 = plt.subplots(4, 4, sharex=True, sharey=True, figsize=(30, 30))  # For PSDs
        # Run PCA and Plot Principle Components in a Huge ass Plot
        fig1, ax1 = plt.subplots(4, 4, sharex=True, sharey=True, figsize=(30, 30))
        # fig1.subplots_adjust(hspace=0.4, wspace=0.4)
        fig2, ax2 = plt.subplots(4, 4, sharey=True, figsize=(40, 20))
        fig3, ax3 = plt.subplots(4, 4, figsize=(30, 30))
        fig4, ax4 = plt.subplots(4, 4, figsize=(30, 30))
        # axis = make_axis_index(4, 4)

    ax = [ax_inst for ax_inst in ax.flat]
    ax_0 = [ax_inst for ax_inst in ax0.flat]
    ax_1 = [ax_inst for ax_inst in ax1.flat]
    ax_2 = [ax_inst for ax_inst in ax2.flat]
    ax_3 = [ax_inst for ax_inst in ax3.flat]
    ax_4 = [ax_inst for ax_inst in ax4.flat]


    fig1.suptitle(f"Number of Active/Inactive Periods = {Pxx_norm.shape[0] / 2}")

    for i in range(Pxx_norm.shape[1]):
        sel_chan = sel_plot_maps[i]
        sel_axis = sel_axis_orders[i]
        plot_summary_psd(Pxx_concat[:, sel_chan, :], freqs, num_trials=int(Pxx_norm.shape[0] / 2), ax=ax[sel_axis])
        ax[sel_axis].set(title=f"Power in CH {sel_chan_map[i]}")  # Add the Channel number to the PSD Plots

        plot_psds_for_one_channel_repeatedly(Pxx_concat, freqs, int(Pxx_norm.shape[0] / 2), ax_0[sel_axis],
                                             channel=sel_chan)
        ax_0[sel_axis].set(title=f"Power in CH {sel_chan_map[i]}")  # Add the Channel number to the PSD Plots

        run_pca_analysis_for_one_channel_repeatedly(Pxx_norm, int(Pxx_norm.shape[0] / 2), ax_1[sel_axis],
                                                    ax_2[sel_axis], ax_3[sel_axis], ax_4[sel_axis], channel=sel_chan)

    fig1.text(.1, .1, f"The Number of Events for this day was {Pxx_norm.shape[0] / 2}")

    report_name = 'PCA_Multitaper_' + bird_id + '_' + session + '_report.pdf'
    report_type_folder = REPORTS_DIR / 'Chunk_PCA_Multitaper'

    # Check if Folder Path Exists
    if not report_type_folder.exists():
        report_type_folder.mkdir(parents=True, exist_ok=True)

    report_location = report_type_folder / report_name

    # Create the PdfPages object to which we will save the pages:
    # The with statement makes sure that the PdfPages object is closed properly at
    # the end of the block, even if an Exception occurs.
    with PdfPages(report_location) as pdf:
        pdf.attach_note(f"Number of epochs = {Pxx_norm.shape[0] / 2}", positionRect=[-100, -100, 0, 0])
        pdf.savefig(fig0)
        pdf.savefig(fig)
        pdf.savefig(fig1)
        pdf.savefig(fig2)
        pdf.savefig(fig3)
        pdf.savefig(fig4)


def make_welch_pca_report(bird_id: str, session: str, nperseg=1000):
    sel_chan_map = all_chan_map[bird_id]
    sel_plot_maps = all_plot_maps[bird_id]
    sel_axis_orders = all_axis_orders[bird_id]

    Pxx_norm, Pxx_concat, freqs = make_welch_pca_prep(bird_id=bird_id, session=session, nperseg=nperseg)

    if Pxx_norm.shape[1] > 16:
        fig, ax = plt.subplots(6, 6, sharex=True, sharey=True, figsize=(50, 50))  # For PSDs Summary
        fig0, ax0 = plt.subplots(6, 6, sharex=True, sharey=True, figsize=(50, 50))  # For PSDs
        # Run PCA and Plot Principle Components in a Huge ass Plot
        fig1, ax1 = plt.subplots(6, 6, sharex=True, sharey=True, figsize=(50, 50))
        # fig1.subplots_adjust(hspace=0.4, wspace=0.4)
        fig2, ax2 = plt.subplots(6, 6, sharey=True, figsize=(60, 40))
        fig3, ax3 = plt.subplots(6, 6, figsize=(50, 50))
        fig4, ax4 = plt.subplots(6, 6, figsize=(50, 50))
        # axis = make_axis_index(6, 6)

    else:
        fig, ax = plt.subplots(4, 4, sharex=True, sharey=True, figsize=(30, 30))  # For PSDs
        fig0, ax0 = plt.subplots(4, 4, sharex=True, sharey=True, figsize=(30, 30))  # For PSDs
        # Run PCA and Plot Principle Components in a Huge ass Plot
        fig1, ax1 = plt.subplots(4, 4, sharex=True, sharey=True, figsize=(30, 30))
        # fig1.subplots_adjust(hspace=0.4, wspace=0.4)
        fig2, ax2 = plt.subplots(4, 4, sharey=True, figsize=(40, 20))
        fig3, ax3 = plt.subplots(4, 4, figsize=(30, 30))
        fig4, ax4 = plt.subplots(4, 4, figsize=(30, 30))
        # axis = make_axis_index(4, 4)

    ax = [ax_inst for ax_inst in ax.flat]
    ax_0 = [ax_inst for ax_inst in ax0.flat]
    ax_1 = [ax_inst for ax_inst in ax1.flat]
    ax_2 = [ax_inst for ax_inst in ax2.flat]
    ax_3 = [ax_inst for ax_inst in ax3.flat]
    ax_4 = [ax_inst for ax_inst in ax4.flat]

    fig1.suptitle(f"Number of Active/Inactive Periods = {Pxx_norm.shape[0] / 2}")

    for i in range(Pxx_norm.shape[1]):
        sel_chan = sel_plot_maps[i]
        sel_axis = sel_axis_orders[i]
        plot_summary_psd(Pxx_concat[:, sel_chan, :], freqs, num_trials=int(Pxx_norm.shape[0] / 2), ax=ax[sel_axis])
        ax[sel_axis].set(title=f"Power in CH {sel_chan_map[i]}")  # Add the Channel number to the PSD Plots

        plot_psds_for_one_channel_repeatedly(Pxx_concat, freqs, int(Pxx_norm.shape[0] / 2), ax_0[sel_axis],
                                             channel=sel_chan)
        ax_0[sel_axis].set_ylim(bottom=0.1)
        ax_0[sel_axis].set(title=f"Power in CH {sel_chan_map[i]}")  # Add the Channel number to the PSD Plots

        run_pca_analysis_for_one_channel_repeatedly(Pxx_norm, int(Pxx_norm.shape[0] / 2), ax_1[sel_axis],
                                                    ax_2[sel_axis], ax_3[sel_axis], ax_4[sel_axis], channel=sel_chan)

    fig1.text(.1, .1, f"The Number of Events for this day was {Pxx_norm.shape[0] / 2}")

    report_name = 'PCA_Welch_' + bird_id + '_' + session + '_' + str(nperseg) + '_report.pdf'
    report_type_folder = REPORTS_DIR / 'Chunk_PCA_Welch'

    # Check if Folder Path Exists
    if not report_type_folder.exists():
        report_type_folder.mkdir(parents=True, exist_ok=True)

    report_location = report_type_folder / report_name

    # Create the PdfPages object to which we will save the pages:
    # The with statement makes sure that the PdfPages object is closed properly at
    # the end of the block, even if an Exception occurs.
    with PdfPages(report_location) as pdf:
        pdf.attach_note(f"Number of epochs = {Pxx_norm.shape[0] / 2}", positionRect=[-100, -100, 0, 0])
        pdf.savefig(fig0)
        pdf.savefig(fig)
        pdf.savefig(fig1)
        pdf.savefig(fig2)
        pdf.savefig(fig3)
        pdf.savefig(fig4)

