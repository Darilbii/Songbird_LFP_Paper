import BirdSongToolbox as tb
from src.utils.paths import REPORTS_DIR

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.signal import welch
from matplotlib.backends.backend_pdf import PdfPages


def common_average_reference_PreP_data(neural_data, epoch_index: list = None, bad_channels: list = None):
    """ Convert PreP Data to ndarray and Common Average References data

    Parameters
    ----------
    neural_data : list
        PreP_Data Format for Neural Data. User Designated Neural data during Song Trials
        [Number of Trials]-> [Trial Length (Samples @ User Designated Sample Rate) x Ch]
    epoch_index : list, optional
        List of Epochs to select for analysis
    bad_channels : list, optional
        list of Channels To Exclude from Common Average Reference

    Returns
    -------
    data_common_avg_ref : ndarray
        An array object of the Common Averaged Referenced Data, shape (Epochs, Samples, Channels)
    """
    # Convert to a ndarray (Deep Copy), shape: (Epochs, Samples, Channels)
    data_common_avg_ref = np.array(neural_data)

    # Only Select the Epochs (All_First_Epochs)
    if epoch_index is not None:
        data_common_avg_ref = data_common_avg_ref[epoch_index, :, :]

    # Exclude Noisy Channels from CAR if list of bad channels given
    channels_include = list(range(data_common_avg_ref.shape[2]))
    if bad_channels is not None:
        channels_include = np.delete(channels_include, bad_channels)

    # Common Average Reference
    data_common_avg_ref = data_common_avg_ref - np.mean(data_common_avg_ref[:, :, channels_include], axis=2)[:, :, None]

    return data_common_avg_ref


def preprocess_for_pca(song_neural, silence_neural, epoch_index: list, bad_channels: list):
    """ Convert PreP Data to ndarray, Select only First Motifs, and Common Average Reference

    :param song_neural:
    :param silence_neural:
    :param epoch_index:
    :return:
    """
    # Convert to a ndarray, shape: (Epochs, Samples, Channels)
    song_data = np.asarray(song_neural)
    silence_data = np.asarray(silence_neural)

    # Only Select the Epochs (All_First_Epochs)
    song_data = song_data[epoch_index, :, :]
    silence_data = silence_data[epoch_index, :, :]

    channels_include = list(range(song_data.shape[2]))
    if len(bad_channels) > 0:
        channels_include = np.delete(channels_include, bad_channels)

    # Common Average Reference
    song_data = song_data - np.mean(song_data[:, :, channels_include], axis=2)[:, :, None]
    silence_data = silence_data - np.mean(silence_data[:, :, channels_include], axis=2)[:, :, None]

    return song_data, silence_data


def plot_every_channels_psd(Pxx_song, Pxx_silence, freqs):
    # Visualize one comparison Between the Vocally Active and Vocally inactive Trials
    Num_channels = Pxx_song.shape[2]

    for i in range(Num_channels):
        fig, ax = plt.subplots()
        ax.plot(freqs[:200], np.mean(Pxx_song[:, :200, i], axis=0), label='song')
        ax.plot(freqs[:200], np.mean(Pxx_silence[:, :200, i], axis=0), label='silence')
        ax.set_xticks([0, 50, 100, 150, 200])  # Order Matters when using the set_xscale to 'log
        #     ax.set_xscale('log')
        ax.set_xlim((0, 200))

        ax.set_xticklabels([0, 50, 100, 150, 200])
        ax.set_yscale('log')

        ax.legend()
        fig.show


def normalize_psds(Pxx_song, Pxx_silence):
    # Normalize using Kai Miller's Method (divide by frequency element mean and then take the log)

    # Normalize Together
    Pxx_concat = np.concatenate((Pxx_song, Pxx_silence), axis=0)
    Pxx_norm = np.log(Pxx_concat / np.mean(Pxx_concat, axis=0)[None, :, :])
    return Pxx_norm


def run_pca_analysis_for_one_channel(Pxx_norm):
    # Run PCA
    pca = PCA(n_components=.95)
    pca.fit(Pxx_norm[:, :200, 3])

    # Plot the Explained Variance with Number of Components
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('number of components')
    plt.ylabel('cumulative explained Variance')

    # Plot the first 3 Principle Components (PSCS)
    for i in range(3):
        plt.plot(pca.components_[i, :], label=str(i + 1))
    plt.legend()

    # Plot the Data based off the first two Principle Compoenents
    fig, ax = plt.subplots(1, 1)
    X_pca = pca.transform(Pxx_norm[:, :200, 3])
    ax.scatter(X_pca[:29, 0], X_pca[:29, 1], alpha=0.2, color='red')
    ax.scatter(X_pca[29:, 0], X_pca[29:, 1], alpha=0.2, color='blue')
    ax.axis('equal')
    ax.set(xlabel='component 1', ylabel='component 2',
           title='principal components')  # ,xlim=(-5, 5), ylim=(-3, 3.1))
    fig.show()


def make_axis_index(row, col):
    index = []
    for i in range(row):
        for j in range(col):
            index.append([i, j])
    return index


def run_pca_analysis_for_one_channel_repeatedly2(Pxx_norm, Num_Epochs, ax1, ax2, ax3, index):
    # Run PCA
    pca = PCA(n_components=.95)
    pca.fit(Pxx_norm[:, :200, index])

    # Plot the Explained Variance with Number of Components
    ax1.plot(np.cumsum(pca.explained_variance_ratio_))
    ax1.set_xlabel('# PCs')
    ax1.set_ylabel('explained variance')
    ax1.set_title(f"Number of Principle Components Ch: {index}")

    # Plot the first 3 Principle Components (PSCS)
    for i in range(3):
        ax2.plot(pca.components_[i, :], label=str(i + 1))
    ax2.legend()
    ax2.set_title(f"Ch: {index} PSCs projected into PSD (x=Hz)")

    # Plot the Data based off the first two Principle Compoenents
    #     fig, ax = plt.subplots(1, 1)
    X_pca = pca.transform(Pxx_norm[:, :200, index])
    ax3.scatter(X_pca[:Num_Epochs, 0], X_pca[:Num_Epochs, 1], alpha=0.2, color='red', label='Active')
    ax3.scatter(X_pca[Num_Epochs:, 0], X_pca[Num_Epochs:, 1], alpha=0.2, color='blue', label='Inactive')
    ax3.axis('equal')
    ax3.set(xlabel='component 1', ylabel='component 2',
            title=f"principal components for CH {index}")  # ,xlim=(-5, 5), ylim=(-3, 3.1))
    ax3.legend()


def main(bird_id, session):
    # Import Data
    z_data = tb.Import_PrePd_Data(bird_id, session)

    ## Make Instance of Pre-Processing Class to get class objects
    z_processed = tb.Pipeline(z_data)

    ##############Quick Fix###############
    # TODO: Make the Bad_channels zero-indexed
    # Quick Fix for bad Channels:
    bad_channels = []
    if len(z_processed.Bad_Channels) > 0:
        for i in z_processed.Bad_Channels:
            bad_channels.append(i - 1)
    ##############Quick Fix###############

    song_data, silence_data = preprocess_for_pca(song_neural=z_data.Song_Neural,
                                                 silence_neural=z_data.Silence_Neural,
                                                 epoch_index=z_data.All_First_Motifs,
                                                 bad_channels=bad_channels)

    # Note for nperseg: The frequency of the kth bin is given by, k × (sampling_rate / N)

    freqs, Pxx_song = welch(song_data, fs=1000, window='hann', nperseg=1000,
                            scaling='spectrum', axis=1)

    freqs, Pxx_silence = welch(silence_data, fs=1000, window='hann', nperseg=1000,
                               scaling='spectrum', axis=1)

    Pxx_norm = normalize_psds(Pxx_song, Pxx_silence)

    if Pxx_norm.shape[2] > 16:
        # Run PCA and Plot Principle Components in a Huge ass Plot
        fig1, ax1 = plt.subplots(6, 6, sharex=True, sharey=True, figsize=(50, 50))
        # fig1.subplots_adjust(hspace=0.4, wspace=0.4)
        fig2, ax2 = plt.subplots(6, 6, sharey=True, figsize=(60, 40))
        fig3, ax3 = plt.subplots(6, 6, figsize=(50, 50))
        axis = make_axis_index(6, 6)

    else:
        # Run PCA and Plot Principle Components in a Huge ass Plot
        fig1, ax1 = plt.subplots(4, 4, sharex=True, sharey=True, figsize=(30, 30))
        # fig1.subplots_adjust(hspace=0.4, wspace=0.4)
        fig2, ax2 = plt.subplots(4, 4, sharey=True, figsize=(40, 20))
        fig3, ax3 = plt.subplots(4, 4, figsize=(30, 30))
        axis = make_axis_index(4, 4)
    fig1.suptitle(f"Number of epochs = {len(z_data.All_First_Motifs)}")

    for i in range(Pxx_norm.shape[2]):
        run_pca_analysis_for_one_channel_repeatedly2(Pxx_norm, len(z_data.All_First_Motifs),
                                                     ax1[axis[i][0], axis[i][1]],
                                                     ax2[axis[i][0], axis[i][1]], ax3[axis[i][0], axis[i][1]], index=i)

    fig1.text(.1, .1, f"The Number of epochs for this day was {len(z_data.All_First_Motifs)}")

    report_name = 'PCA_' + bird_id + '_' + session + '_report_fixed.pdf'
    report_location = REPORTS_DIR / report_name

    # Create the PdfPages object to which we will save the pages:
    # The with statement makes sure that the PdfPages object is closed properly at
    # the end of the block, even if an Exception occurs.
    with PdfPages(report_location) as pdf:
        pdf.attach_note(f"Number of epochs = {len(z_data.All_First_Motifs)}", positionRect=[-100, -100, 0, 0])
        pdf.savefig(fig1)
        pdf.savefig(fig2)
        pdf.savefig(fig3)


def plot_psd_comparison_for_one_channel_repeatedly(Pxx_song, Pxx_silence, freqs, ax1, index):
    # Plot the Psd of Vocally Active vs. Inactive

    ax1.plot(freqs[:200], np.mean(Pxx_song[:, :200, index], axis=0), label='song')
    ax1.plot(freqs[:200], np.mean(Pxx_silence[:, :200, index], axis=0), label='silence')
    ax1.set_xticks([0, 50, 100, 150, 200])  # Order Matters when using the set_xscale to 'log

    ax1.set_xticklabels([0, 50, 100, 150, 200])
    ax1.set_yscale('log')
    ax1.set_title(f"Ch: {index} PSD (x=Hz)")

    ax1.legend()


def sanity_check_active_v_inactive_epoch_psd_welch(bird_id, session):
    # Import Data
    z_data = tb.Import_PrePd_Data(bird_id, session)

    ## Make Instance of Pre-Processing Class to get class objects
    z_processed = tb.Pipeline(z_data)

    ##############Quick Fix###############
    # TODO: Make the Bad_channels zero-indexed
    # Quick Fix for bad Channels:
    bad_channels = []
    if len(z_processed.Bad_Channels) > 0:
        for i in z_processed.Bad_Channels:
            bad_channels.append(i - 1)
    ##############Quick Fix###############

    song_data = common_average_reference_PreP_data(neural_data=z_data.Song_Neural,
                                                   epoch_index=z_processed.All_First_Motifs,
                                                   bad_channels=bad_channels)

    silence_data = common_average_reference_PreP_data(neural_data=z_data.Silence_Neural,
                                                      bad_channels=bad_channels)

    # Note for nperseg: The frequency of the kth bin is given by, k × (sampling_rate / N)

    freqs, Pxx_song = welch(song_data, fs=1000, window='hann', nperseg=1000,
                            scaling='spectrum', axis=1)

    freqs, Pxx_silence = welch(silence_data, fs=1000, window='hann', nperseg=1000,
                               scaling='spectrum', axis=1)

    if Pxx_song.shape[2] > 16:
        fig1, ax1 = plt.subplots(6, 6, sharey=True, figsize=(60, 40))
        axis = make_axis_index(6, 6)

    else:
        fig1, ax1 = plt.subplots(4, 4, sharey=True, figsize=(40, 20))
        axis = make_axis_index(4, 4)

    for i in range(Pxx_song.shape[2]):
        plot_psd_comparison_for_one_channel_repeatedly(Pxx_song, Pxx_silence, freqs, ax1[axis[i][0], axis[i][1]],
                                                       index=i)

    fig1.text(.1, .1, f"The Number of epochs for this day was {len(z_data.All_First_Motifs)}")

    report_name = 'PSD_Welch_' + bird_id + '_' + session + '_report.pdf'
    report_type_folder = REPORTS_DIR / 'PSD_Sanity_Check'

    # Check if Folder Path Exists
    if not report_type_folder.exists():
        report_type_folder.mkdir(parents=True, exist_ok=True)

    report_location = report_type_folder / report_name

    # Create the PdfPages object to which we will save the pages:
    # The with statement makes sure that the PdfPages object is closed properly at
    # the end of the block, even if an Exception occurs.
    with PdfPages(report_location) as pdf:
        pdf.attach_note(f"Number of epochs = {len(z_data.All_First_Motifs)}", positionRect=[-100, -100, 0, 0])
        pdf.savefig(fig1)



