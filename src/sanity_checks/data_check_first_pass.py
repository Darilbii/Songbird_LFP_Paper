""" Get Information about the Old 2 Second Buffer Epochs"""

import BirdSongToolbox.GetBirdData as gbd
from BirdSongToolbox.file_utility_functions import _save_numpy_data, _save_json_data

from collections import Counter
from pathlib import Path

import numpy as np
import os
import h5py


def _get_true_starts(kwe_data, kwik_data, verbose=False):
    """ Get the start times of the automated Motif labels within the entire recording

    Parameters
    ----------
    kwe_data : dict
        dictionary of the events in the KWE file
        Keys:
            'motif_st': [# of Motifs]
            'motif_rec_num': [# of Motifs]
    kwik_data : dict
        data needed to render spiking activity fromm the Kwik file
        keys: 'recordingStarts', 'time_samples', 'clusters

    Returns
    -------
    times : array
        array of the absolute start of the labels
    '"""

    times = np.zeros((kwe_data['motif_st'].shape[0]))

    for motif in range(kwe_data['motif_st'].shape[0]):

        # Get start time for motif and recording start
        motif_start_time = kwe_data['motif_st'][motif]  # Start Time of Motif in its Specific Recording
        motif_rec_start = kwik_data['recordingStarts'][kwe_data['motif_rec_num'][motif]]  # Start Sample of Recording
        start_time = int(motif_start_time + motif_rec_start)  # The True Start time within the Entire Recording
        times[motif] = start_time  # Add the Start time to the array

        if verbose:
            # Print out info about motif
            print('On Motif ', (motif + 1), '/', kwe_data['motif_st'].shape[0])

    return times



def _check_for_repeats(times):
    """ Check if any of Zeke's Labels are Repeats

    Parameters
    ----------
    times : array
        array of the absolute start of the labels

    Returns
    -------
    removals : array
        array of the automated motifs that are repeats (Not including the one that will be used)
    """

    repeats = [item for item, count in Counter(times).items() if count > 1]  # Find which times are repeated

    removals = []
    for i in repeats:
        repeat_motifs = np.where(times == i)[0]  # Find which Motifs are repeats of this time stamp
        removals.append(repeat_motifs[1:])  # Record all but one of the repeats

    removals = np.squeeze(np.asarray(removals))  # Convert to a array and remove the extra dimension

    return removals


def _get_rec_duration(kwd_file):
    """Calculate the duration of the Recording (in Samples) using the KWD File"""

    Duration = 0
    for i in kwd_file['recordings'].keys():
        Duration = Duration + kwd_file['recordings'][str(i)]['data'].shape[0]
    return Duration


def make_session_report(duration, times, repeats):
    """ Makes a simple report of the results of the first pass check of the data

    Returns
    -------
    report : dict
        keys: {duration: int (Recording Duration),
                num_motifs: int (Number of Motifs),
                num_repeats: int (Number of Repeats)}
    """
    report = {}
    report['duration'] = duration
    report['num_motifs'] = len(times)
    report['num_repeats'] = len(repeats)

    return report


def main_for_1day(bird_id, session, motif_dur, before_t, after_t):
    """ First Pass Audio Check

    Metrics it gets:
        The Start and End of the Epochs
        The Absolute Start Times of the Motifs
        report: {duration: int (Recording Duration),
                num_motifs: int (Number of Motifs),
                num_repeats: int (Number of Repeats)}


    :param session:
    :param bird_id:
    :param motif_dur:
    :param before_t:
    :param after_t:
    :return:


    Returns
    -------
    epoch_times : array
        (Motifs, 2) # For Kai and Also for Neural Networks
        [Start Sample, End Sample]
    times : array
        array of the absolute start of the labels
    report : dict
        keys: {duration: int (Recording Duration),
                num_motifs: int (Number of Motifs),
                num_repeats: int (Number of Repeats)}

    """
    # Folder where birds are
    experiment_folder_path = '/net/expData/birdSong/ss_data'

    bird_folder_path = os.path.join(experiment_folder_path, bird_id)  # Folder for the bird

    bird_path = Path(bird_folder_path)
    folders = [x for x in bird_path.iterdir() if x.is_dir()]  # Get all of the Folders in that Day

    dayFolder = os.path.join(bird_folder_path, session)  # Folder for Session

    # [3] Select Kwik File and get its Data
    kwik_file = gbd.get_data_path(day_folder=dayFolder, file_type='.kwik')  # Ask User to select Kwik File
    kwik_file_path = os.path.join(dayFolder, kwik_file)  # Get Path to Selected Kwik File
    kwik_data = gbd.read_kwik_data(kwik_path=kwik_file_path, verbose=True)  # Make Dict of Data from Kwik File

    # [4] Select the Kwe file
    kwe = gbd.get_data_path(day_folder=dayFolder, file_type='.kwe')  # Select KWE File
    kwe_file_path = os.path.join(dayFolder, kwe)  # Get Path to Selected KWE File
    kwe_data = gbd.read_kwe_file(kwe_path=kwe_file_path, verbose=False)  # Read KWE Data into Dict

    # [5] Select the Kwd file
    kwd = gbd.get_data_path(day_folder=dayFolder, file_type='.kwd')  # Select Kwd File
    kwd_file = h5py.File(os.path.join(dayFolder, kwd), 'r')  # Read Kwd Data into Dict

    # Showing where data is coming from
    print('Getting Data from ', kwd)

    song_len_ms = motif_dur + before_t + after_t  # Calculate the Duration of the Grabbed Data
    SamplingRate = 30000

    # []
    epoch_times = gbd.get_epoch_times(kwe_data, kwik_data, song_len_ms, before_t)
    _save_numpy_data(data=epoch_times, data_name="EpochTimes", bird_id=bird_id, session=session)

    # []
    true_times = _get_true_starts(kwe_data, kwik_data, verbose=False)
    _save_numpy_data(data=true_times, data_name="AbsoluteTimes", bird_id=bird_id, session=session)

    # []
    need_to_ignore = _check_for_repeats(true_times)
    _save_numpy_data(data=need_to_ignore, data_name="Repeats", bird_id=bird_id, session=session)

    # []
    duration = _get_rec_duration(kwd_file)

    report = make_session_report(duration=duration, times=true_times, repeats=need_to_ignore)

    # Save the report to a json
    _save_json_data(data=report, data_name="Report", bird_id=bird_id, session=session)
    # file_name = 'Report' + '_' + bird_id + '_' + session + '.json'
    # destination = gbd.INTERMEDIATE_DATA_PATH / file_name
    # file_object = open(destination, "w", encoding="utf8")
    # json.dump(report, file_object)
    # file_object.close()

    # print('Saving First Pass Report to', destination)




def main(bird_id, motif_dur, before_t, after_t):
    """ First Pass Audio Check

    Metrics it gets:
        The Start and End of the Epochs
        The Absolute Start Times of the Motifs
        report: {duration: int (Recording Duration),
                num_motifs: int (Number of Motifs),
                num_repeats: int (Number of Repeats)}


    :param bird_id:
    :param motif_dur:
    :param before_t:
    :param after_t:
    :return:


    Returns
    -------
    epoch_times : array
        (Motifs, 2) # For Kai and Also for Neural Networks
        [Start Sample, End Sample]
    times : array
        array of the absolute start of the labels
    report : dict
        keys: {duration: int (Recording Duration),
                num_motifs: int (Number of Motifs),
                num_repeats: int (Number of Repeats)}

    """
    # Folder where birds are
    experiment_folder_path = '/net/expData/birdSong/ss_data'

    bird_folder_path = os.path.join(experiment_folder_path, bird_id)  # Folder for the bird

    bird_path = Path(bird_folder_path)
    folders = [x for x in bird_path.iterdir() if x.is_dir()]  # Get all of the Folders in that Day
    days = [folder.stem for folder in folders if 'day' in folder.stem]  # Get all of the actual Days frpm the Bird

    for session in days:
        dayFolder = os.path.join(bird_folder_path, session)  # Folder for Session

        # [3] Select Kwik File and get its Data
        kwik_file = gbd.get_data_path(day_folder=dayFolder, file_type='.kwik')  # Ask User to select Kwik File
        kwik_file_path = os.path.join(dayFolder, kwik_file)  # Get Path to Selected Kwik File
        kwik_data = gbd.read_kwik_data(kwik_path=kwik_file_path, verbose=True)  # Make Dict of Data from Kwik File

        # [4] Select the Kwe file
        kwe = gbd.get_data_path(day_folder=dayFolder, file_type='.kwe')  # Select KWE File
        kwe_file_path = os.path.join(dayFolder, kwe)  # Get Path to Selected KWE File
        kwe_data = gbd.read_kwe_file(kwe_path=kwe_file_path, verbose=False)  # Read KWE Data into Dict

        # [5] Select the Kwd file
        kwd = gbd.get_data_path(day_folder=dayFolder, file_type='.kwd')  # Select Kwd File
        kwd_file = h5py.File(os.path.join(dayFolder, kwd), 'r')  # Read Kwd Data into Dict

        # Showing where data is coming from
        print('Getting Data from ', kwd)

        song_len_ms = motif_dur + before_t + after_t  # Calculate the Duration of the Grabbed Data
        SamplingRate = 30000

        # []
        epoch_times = gbd.get_epoch_times(kwe_data, kwik_data, song_len_ms, before_t)
        _save_numpy_data(data=epoch_times, data_name="EpochTimes", bird_id=bird_id, session=session)

        # []
        true_times = _get_true_starts(kwe_data, kwik_data, verbose=False)
        _save_numpy_data(data=true_times, data_name="AbsoluteTimes", bird_id=bird_id, session=session)

        # []
        need_to_ignore = _check_for_repeats(true_times)
        _save_numpy_data(data=need_to_ignore, data_name="Repeats", bird_id=bird_id, session=session)

        # []
        duration = _get_rec_duration(kwd_file)

        report = make_session_report(duration=duration, times=true_times, repeats=need_to_ignore)

        # Save the report to a json
        _save_json_data(data=report, data_name="Report", bird_id=bird_id, session=session)
        # file_name = 'Report' + '_' + bird_id + '_' + session + '.json'
        # destination = gbd.INTERMEDIATE_DATA_PATH / file_name
        # file_object = open(destination, "w", encoding="utf8")
        # json.dump(report, file_object)
        # file_object.close()

        # print('Saving First Pass Report to', destination)


