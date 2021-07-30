from src.utils.paths import REPORTS_DIR

import BirdSongToolbox as tb
import BirdSongToolbox.Epoch_Analysis_Tools as bep
import BirdSongToolbox.feature_dropping_suite as fd

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def inspect_pressure_audio_for_old_label(old_audio, label_index, buffer, sn_len, bad_labels: list = [], alpha=.5,
                                         show=False):
    """ For each channel plot all of its unfiltered traces for one type of Lable (First Type of Label)"""
    true_start = int(buffer / 2) - sn_len
    window = sn_len * 3

    firsts2 = np.zeros((window * 30, len(label_index)))

    for index, i in enumerate(label_index):
        if i in bad_labels:
            pass
        else:
            firsts2[:, index] = old_audio[i][true_start * 30:(true_start + window) * 30, 0]
            firsts2[:, index] = firsts2[:, index] / np.max(firsts2[:, index])
    print(true_start)
    print(true_start + window)
    plt.plot(firsts2, alpha=alpha)
    plt.axvline(x=sn_len * 30, color='red')
    plt.axvline(x=sn_len * 2 * 30, color='red')

    if show:
        plt.show()


def inspect_each_ch_per_old_label(old_data, old_audio, label_index, buffer, sn_len, bad_labels: list = [], alpha=.5,
                                  pdf=False):
    """ For each channel plot all of its unfiltered traces for one type of Lable (First Type of Label)"""

    true_start = int(buffer / 2) - sn_len
    window = sn_len * 3  # The Window is 1 Motif Length Before and After the End of the Labeled Motif

    firsts2 = np.zeros((window * 30, len(label_index)))

    for index, i in enumerate(label_index):
        firsts2[:, index] = old_audio[i][true_start * 30:(true_start + window) * 30, 0]
        firsts2[:, index] = firsts2[:, index] / np.max(firsts2[:, index])

    sel_trials = np.zeros((window, len(label_index)))

    # For Each Channel
    for j in range(16):
        fig, ax = plt.subplots(2, 1)
        # For each Epoch
        for index, i in enumerate(label_index):
            #     plt.plot(z020_day_1.Song_Neural[i][:,0])
            if i in bad_labels:
                pass
            else:
                sel_trials[:, index] = old_data[i][true_start:(true_start + window), j]  # Grab one Channels Data
        ax[0].plot(firsts2, alpha=alpha)
        ax[0].axvline(x=sn_len * 30, color='red')
        ax[0].axvline(x=sn_len * 2 * 30, color='red')
        ax[1].plot(sel_trials)
        ax[0].set_title(f"Channel # {j}")
        ax[1].axvline(x=sn_len, color='red')
        ax[1].axvline(x=sn_len * 2, color='red')

        if pdf:
            pdf.savefig(fig)
            fig.close()
        else:
            fig.show()


# Plot all Trials for One Type of Song
def inspect_ch_similarity_per_old_label(old_data, old_audio, buffer, sn_len, label_index, bad_chans: list = [],
                                        pdf=False):
    """ For each instance of a type of Old label plot all of the Channels overlapping with each other"""

    true_start = int(buffer / 2) - sn_len
    window = sn_len * 3  # The Window is 1 Motif Length Before and After the End of the Labeled Motif

    firsts2 = np.zeros((window * 30, len(label_index)))

    for index, i in enumerate(label_index):
        firsts2[:, index] = old_audio[i][true_start * 30:(true_start + window) * 30, 0]
        firsts2[:, index] = firsts2[:, index] / np.max(firsts2[:, index])

    num_chans = len(old_data[0][0, :])  # The Number of Recording Channels
    sel_trials = np.zeros((window, num_chans))

    # For each Epoch
    for index, i in enumerate(label_index):
        fig, ax = plt.subplots(2, 1)
        for j in range(num_chans):
            if j in bad_chans:
                pass
            else:
                sel_trials[:, j] = old_data[i][true_start:(true_start + window), j]  # Grab one Channels Data
        ax[0].plot(firsts2[:, index])
        ax[0].set_title(f"Old Epoch: {i}")
        ax[0].axvline(x=sn_len * 30, color='red')
        ax[0].axvline(x=sn_len * 2 * 30, color='red')
        ax[1].plot(sel_trials)
        ax[1].axvline(x=sn_len, color='red')
        ax[1].axvline(x=sn_len * 2, color='red')

        if pdf:
            pdf.savefig(fig)
            fig.close()
        else:
            fig.show()


def main():
    data = tb.Import_PrePd_Data('z007', 'day-2016-09-07')

    bad_channels = []
    bad_trials = []

    report_location = REPORTS_DIR / 'test_pdf.pdf'

    # Create the PdfPages object to which we will save the pages:
    # The with statement makes sure that the PdfPages object is closed properly at
    # the end of the block, even if an Exception occurs.
    with PdfPages(report_location) as pdf:
        inspect_pressure_audio_for_old_label(old_audio=data.Song_Audio,
                                             buffer=data.Gap_Len,
                                             sn_len=data.Sn_Len,
                                             label_index=data.All_First_Motifs,
                                             alpha=.3,
                                             show=False)
        pdf.savefig()
        plt.close()

        # Plot all Trials for One Type of Song
        inspect_each_ch_per_old_label(old_data=data.Song_Neural,
                                      old_audio=data.Song_Audio,
                                      buffer=data.Gap_Len,
                                      sn_len=data.Sn_Len,
                                      label_index=data.All_First_Motifs,
                                      bad_labels=bad_trials,
                                      alpha=.5,
                                      pdf=pdf)

        inspect_ch_similarity_per_old_label(old_data=data.Song_Neural,
                                            old_audio=data.Song_Audio,
                                            buffer=data.Gap_Len,
                                            sn_len=data.Sn_Len,
                                            label_index=data.All_First_Motifs,
                                            bad_chans=bad_channels,
                                            pdf=pdf)
