from BirdSongToolbox.file_utility_functions import _load_numpy_data, _save_pckl_data, _save_json_data
from src.utils.paths import REPORTS_DIR
import BirdSongToolbox as tb

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Find the Epochs that overlap
# Catalogue the Duration of the Overlap
# Save an array of the samples that overlap
# Make a report showing the epoch overlap


def range_overlapping(x: list, y: list):
    """ Determine if two Epochs Overlap in Time
    Parameters
    ----------
    x : list
        all values in range x
    y : list
        all values in range y
    """
    if x.start == x.stop or y.start == y.stop:
        return False
    return ((x.start < y.stop and x.stop > y.start) or
            (x.stop > y.start and y.stop > x.start))


def overlap(x, y):
    if not range_overlapping(x, y):
        return set()
    return set(range(max(x.start, y.start), min(x.stop, y.stop) + 1))


def find_epochs_that_overlap(epoch_times, epoch_index):
    """ Check Each Epoch and Determine if they Overlap with each other

    Parameters
    ----------
    epoch_times : array
        (Motifs, 2) # For Kai and Also for Neural Networks
        [Start Sample, End Sample]
    epoch_index : array
        Array of all of the First Epochs for a Specific day

    Returns
    -------
    overlapping_epoch : array
        array of each epoch that overlaps, shape = (Number of Overlaps, 2)
    """
    overlapping_epoch = []
    for index, epoch in enumerate(epoch_index):
        focus_range = range(int(epoch_times[epoch, 0]), int(epoch_times[epoch, 1] + 1), 1)
        for j in epoch_index[index + 1:]:
            #         print(j)
            test_range = range(int(epoch_times[j, 0]), int(epoch_times[j, 1] + 1), 1)
            if range_overlapping(focus_range, test_range):
                print(f"{epoch} overlaps with {j}")
                overlapping_epoch.append([epoch, j])

    return np.array(overlapping_epoch)

def get_overlapping_samples(overlapping_epochs, epoch_times):
    """

    Parameters
    ----------
    overlapping_epochs : array
        array of each epoch that overlaps, shape = (Number of Overlaps, 2)
    epoch_times : array
        (Motifs, 2) # For Kai and Also for Neural Networks
        [Start Sample, End Sample]

    Returns
    -------
    overlapping_samples : list
        liat of the samples that overlap as designated by the overlapping_epoch parameter
    """
    # Python program to illustrate the intersection
    # of two lists
    def intersection(lst1, lst2):
        # Use of hybrid method
        temp = set(lst2)
        lst3 = [value for value in lst1 if value in temp]
        return lst3

    overlapping_samples = []
    for ep1, ep2 in overlapping_epochs:
        overlapping_samples.append(intersection(range(int(epoch_times[ep1, 0]), int(epoch_times[ep1, 1])),
                                                range(int(epoch_times[ep2, 0]), int(epoch_times[ep2, 1]))))
    return overlapping_samples


def plot_the_overlapping_epochs(audio_norm, epoch1, epoch2, epoch_times, epoch_index, ax):
    finder = list(epoch_index)
    index_1 = finder.index(epoch1)
    index_2 = finder.index(epoch2)

    Epoch_length = audio_norm.shape[0]  # Length of the Song Duration

    epoch1_start = epoch_times[epoch1, 0]
    epoch2_start = epoch_times[epoch2, 0]

    if epoch1_start < epoch2_start:
        epoch1_range = np.arange(0, Epoch_length)
        epoch2_range = np.arange(epoch2_start - epoch1_start, epoch2_start - epoch1_start + Epoch_length)
    else:
        epoch2_range = np.arange(0, Epoch_length)
        epoch1_range = np.arange(epoch1_start - epoch2_start, epoch1_start - epoch2_start + Epoch_length)
        print(epoch1_start - epoch2_start)

    ax.plot(epoch1_range, audio_norm[:, index_1], alpha=.5, label=f"Epoch: {epoch1}")
    ax.plot(epoch2_range, audio_norm[:, index_2], alpha=.5, label=f"Epoch: {epoch2}")
    ax.legend(fontsize= 30)


def inspect_epochs_audio(old_audio: list, label_index: np.ndarray):
    """ Convert all of the Epochs Pressure Waveform to a Viewable Fashion

    Parameters
    ----------
    old_audio : list
        audio from the PreP data Format
    label_index : array
        Index to subselect instances of audio
    """

    audio_norm = np.zeros((len(old_audio[0]), len(label_index)))

    for index, i in enumerate(label_index):
        audio_norm[:, index] = old_audio[i][:, 0]
        audio_norm[:, index] = audio_norm[:, index] / np.max(audio_norm[:, index])
    return audio_norm

def make_overlap_report(bird_id, session):
    # Import epoch_times
    epoch_times = _load_numpy_data(data_name='EpochTimes', bird_id=bird_id, session=session)

    # Import Data
    bird_data = tb.Import_PrePd_Data(bird_id, session)

    # Find the Epochs that Overlap
    overlapping_epochs = find_epochs_that_overlap(epoch_times=epoch_times, epoch_index=bird_data.All_First_Motifs)

    # Save the overlapping Epochs Identities
    # _save_json_data(data=list(overlapping_epochs),data_name='Overlap_Identities', bird_id=bird_id, session=session)

    # Get the Samples that Overlap
    overlapping_samples = get_overlapping_samples(overlapping_epochs=overlapping_epochs, epoch_times=epoch_times)

    # Save the overlap parameters
    _save_pckl_data(data=[overlapping_epochs, overlapping_samples], data_name='Overlap_Data', bird_id=bird_id,
                    session=session)

    # Normalize Pressure Waveforms
    Audio = inspect_epochs_audio(old_audio=bird_data.Song_Audio, label_index=bird_data.All_First_Motifs)

    # Create Figure for Report
    fig, ax = plt.subplots(overlapping_epochs.shape[0], figsize=(60, 40))

    for index, (ep1, ep2) in enumerate(overlapping_epochs):
        plot_the_overlapping_epochs(Audio, ep1, ep2, epoch_times, bird_data.All_First_Motifs, ax[index])

    report_name = 'Epoch_Overlap_' + bird_id + '_' + session + '_report.pdf'
    report_type_folder = REPORTS_DIR / 'Epoch_Sanity_Check'

    # Check if Folder Path Exists
    if not report_type_folder.exists():
        report_type_folder.mkdir(parents=True, exist_ok=True)

    report_location = report_type_folder / report_name

    # Create the PdfPages object to which we will save the pages:
    # The with statement makes sure that the PdfPages object is closed properly at
    # the end of the block, even if an Exception occurs.
    with PdfPages(report_location) as pdf:
        pdf.savefig(fig)


