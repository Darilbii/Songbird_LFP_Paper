""" Make A Directory of the Chunk Audio data as .wav and empty textgrid functions"""

# For Script
from BirdSongToolbox.file_utility_functions import _load_pckl_data, _load_numpy_data, _load_json_data
from pathlib import Path
import scipy.io.wavfile
import numpy as np

from praatio import tgio

# Testing of the make_wav_directory Function


def make_wav_directory(directory_name, data, sample_rate: int = 30000):
    """ makes a directory of .wav files for all of the chunks during one day

    Parameters
    ----------
    directory_name : path obj
        Path to the Directory the .wav files will be saved
    data : list, shape [chunk]->(audio_samples,)
        Data to be saved
    sample_rate : int
        Sampling rate to write the .wav file
    """
    assert directory_name.exists(), "destination directory doesn't exist"

    for index, chunk in enumerate(data):
        chunk_wav_name = str(index) + '.wav'  # Name of the .wav file
        data_file_path = directory_name / chunk_wav_name  # Full Path to the .wav file
        wav_data = np.asarray(chunk, np.float32)  # Make Copy of Data at Highest Resolution for the wav file
        wav_data = wav_data / np.max(np.abs(wav_data))  # Make the audio data between -1 and 1

        with open(data_file_path, "w") as file_object:
            scipy.io.wavfile.write(filename=data_file_path, rate=sample_rate, data=wav_data)  # Save the .wav file


def make_days_wav_directory(bird_id, session, activity_type='song', sample_rate: int = 30000):
    """ Make the Directory of .wav files for the Chunk Data

    Parameters
    ----------
    activity_type = str
        Activity Type can either be 'song' or 'silence'

    """
    assert activity_type in ['song', 'silence']

    # buff_chunks_audio : list, shape = [Chunk]->(Samples)
    if activity_type == 'song':
        sel_audio = _load_pckl_data(data_name="Large_Epochs_Audio", bird_id=bird_id, session=session)
    elif activity_type == 'silence':
        sel_audio = _load_pckl_data(data_name="Large_Epochs_Audio_Silence", bird_id=bird_id, session=session)

    # Create Destination Path for .wav files
    chunk_wav_folder = Path('/net/expData/birdSong/ss_data_chunk_wav')  # Root Directory of all chunk .wav files
    assert chunk_wav_folder.parent.exists(), "Only the Root .wav Directory should be recursively created. There is likely a typo in you path"

    wav_path = chunk_wav_folder / bird_id / session / activity_type  # directory created by function

    if not wav_path.exists():
        wav_path.mkdir(parents=True, exist_ok=True)  # Recursively creates parent folders

    make_wav_directory(directory_name=wav_path, data=sel_audio, sample_rate=sample_rate)  # make .wav files


def make_chunk_textgrid(chunk_audio, buffer=10):
    """ Makes a Text Grid with a Empty Interval Tier for Hand Labeling the Chunks

    Parameters
    ----------
    chunk_audio : list, shape = [Chunk]->(channels, Samples)
        Audio Data that is Bandpass Filtered between 300 and 10000 Hz, list of 2darrays
    buffer : int
        Duration of Time at the Start and End of the .wav file that won't be labeled (Neural Filter Buffer)

    Returns
    -------
    textgrid_list : list
        list of Textgrid instances corresponding the the Chunks in chunk_audio, with the 'labels' interval tier

    """

    textgrid_list = []
    for chunk_index, chunk in enumerate(chunk_audio):
        tg = tgio.Textgrid()  # Make a Empty TextGrid

        # Make Empty Interval Tier for the Chunk's Hand Labeling
        chunk_duration = round(len(chunk_audio[chunk_index]) / 30000, 6)  # Convert to Format Praatio expects

        # Make List of the Filter Buffer in the Tier
        buffers = [(0, buffer, 'BUFFER'), (chunk_duration - buffer, chunk_duration, 'BUFFER')]  # Label the Filter Buffer

        labelTier = tgio.IntervalTier('labels', buffers, 0, chunk_duration)  # Make Empty Interval Tier named 'labels'
        tg.addTier(labelTier)  # Add the 'labels' Interval Tier to the Empty TextGrid
        textgrid_list.append(tg)  # Append the TextGrid to the list

    return textgrid_list


def add_kwe_to_textgrid(textgrid_list, chunk_ledger, chunk_times, kwe_times, chunk_audio):
    """Creates and Adds Point Tier of the KWE Onsets

    Parameters
    ----------
    textgrid_list : list
        list of Textgrid instances corresponding the the Chunks in chunk_audio
    chunk_ledger : list
        list of each automated motif that occurs within each Epoch
        [Chunk (Epoch)] -> [Motifs in Epoch (Chunk)]
    chunk_times : list, shape = [Chunk]->(absolute start, absolute end)
        Absolute Start and End of the Chunks
    kwe_times : ndarray, shape = (num_kwe_motifs,)
        Absolute Onsets from the KWE File
    chunk_audio : list, shape = [Chunk]->(channels, Samples)
        Audio Data that is Bandpass Filtered between 300 and 10000 Hz, list of 2darrays

    Returns
    -------
    texgrid_list : list
        list of Textgrid instances corresponding the the Chunks in chunk_audio, with the kwe tier added
    """

    for chunk_index, chunk in enumerate(chunk_ledger):
        chunk_events = []
        for epoch in chunk:
            relative_event = kwe_times[epoch] - chunk_times[chunk_index][0]  # Relative KWE event in the Chunk
            kwe_event = relative_event / 30000  # Convert to Format Praatio expects
            chunk_events.append((kwe_event, 'KWE'))  # Add Epoch Relative Start to Chunk
        chunk_duration = round(len(chunk_audio[chunk_index]) / 30000, 6)  # Convert to Format Praatio expects
        kweTier = tgio.PointTier('KWE', chunk_events, 0, chunk_duration)  # Store the KWE Events to a Point Tier
        textgrid_list[chunk_index].addTier(kweTier)  # Add the 'KWE' Point Tier to the TextGrid

    return textgrid_list


def save_chunks_textgrids(textgrid_list, directory_path):
    """ Saves the Textgrid in the specified directory
    """

    for index, textgrid in enumerate(textgrid_list):
        textgrid_name = str(index) + ".TextGrid"
        textgrid_path = directory_path / textgrid_name
        textgrid.save(textgrid_path)


def transfer_old_handlabels(bird_id, session, chunk_textgrids, chunk_ledger, epoch_times, chunk_times, chunk_audio):
    """ Add Additional Interval Tier with the Previous Hand Labels

    Parameters
    ----------
    textgrid_list : list
        list of Textgrid instances corresponding the the Chunks in chunk_audio
    chunk_ledger : list
        list of each automated motif that occurs within each Epoch
        [Chunk (Epoch)] -> [Motifs in Epoch (Chunk)]
    epoch_times : ndarray, shape = (epochs(kwe), 2)
        The Absolute Start and End of all of the Epochs(KWE) including those not handlabeled
    chunk_times : list, shape = [Chunk]->(absolute start, absolute end)
        Absolute Start and End of the Chunks
    chunk_audio : list, shape = [Chunk]->(channels, Samples)
        Audio Data that is Bandpass Filtered between 300 and 10000 Hz, list of 2darrays

    Returns
    -------
    chunk_textgrids : list
        list of Textgrid instances with a tier for each hand labeled epoch

    """

    textgrid_base_folder = Path('/home/debrown/praat_hand_labels_v1/')
    sel_textgrid_dir = textgrid_base_folder / bird_id / session / 'song'

    all_textgrids = [x for x in sel_textgrid_dir.iterdir() if x.suffix == '.TextGrid']  # list of all textgrid Path()

    # Make Dictionary of all of the Paths to the Textgrids (Keys are there 0-indexed index)
    textgrid_dict = {}
    for textgrid in all_textgrids:
        textgrid_index = int(textgrid.stem) - 1  # 0-indexed index of the epoch
        textgrid_dict[textgrid_index] = textgrid  # store key=0-index and value=path of the epoch

    # Add Interval Tiers for the Labeled Epochs
    for chunk_index, chunk in enumerate(chunk_ledger):
        for epoch in chunk:
            if epoch in textgrid_dict.keys():
                # Determine the Relative Start of the Epoch in the Chunk
                relative_event = epoch_times[epoch, 0] - chunk_times[chunk_index][
                    0]  # Relative Start of Epoch in the Chunk
                relative_event = relative_event / 30000  # Convert to Format Praat Likes

                # Get the Handlabels for the Epoch
                TextGrid_Obj = tgio.openTextgrid(textgrid_dict[epoch])  # Open the TextGrid
                try:
                    LabelTier = TextGrid_Obj.tierDict['labels']  # Get the Tier with the Handlabels
                except:
                    assert len(
                        TextGrid_Obj.tierNameList) == 1, "the TextGrid Doesn't have a tier named 'labels' and there are more than one tiers"
                    LabelTier = TextGrid_Obj.tierDict[TextGrid_Obj.tierNameList[0]]  # The Tier was named different

                # Make a Update version of the Epochs Handlabels to append the Chunk's TextGrid
                chunk_duration = round(len(chunk_audio[chunk_index]) / 30000, 6)  # Convert to Format Praatio expects
                tier_name = 'epoch_' + str(epoch)
                TransTier = LabelTier.new(name=tier_name,
                                          maxTimestamp=chunk_duration)  # Make Copy of Interval Tier with Chunk Duration
                shiftedTier = TransTier.editTimestamps(offset=relative_event)  # Shift Interval Tier labels
                chunk_textgrids[chunk_index].addTier(shiftedTier)  # Add Shifted Tier to the TextGrid

    return chunk_textgrids


def make_chunk_textgrids(bird_id, session, activity_type='song', include_old=False, buffer=10):


    # Import all of the variables needed for the Bird/Session Pair
    if activity_type == 'song':
        chunk_times = _load_pckl_data(data_name="Large_Epochs_Times", bird_id=bird_id, session=session)
        chunk_ledger = _load_pckl_data(data_name="Epochs_Ledger", bird_id=bird_id, session=session)
        chunk_audio = _load_pckl_data(data_name="Large_Epochs_Audio", bird_id=bird_id, session=session)
        times = _load_numpy_data(data_name='AbsoluteTimes', bird_id=bird_id, session=session)
        epoch_times = _load_numpy_data(data_name='EpochTimes', bird_id=bird_id, session=session)

    elif activity_type =='silence':
        chunk_audio = _load_pckl_data(data_name="Large_Epochs_Audio_Silence", bird_id=bird_id, session=session)

    # Make the TextGrid and Populate it with Tiers
    textgrid_list = make_chunk_textgrid(chunk_audio, buffer=buffer)  # Make the TextGrids

    if activity_type == 'song':
        textgrid_list = add_kwe_to_textgrid(textgrid_list, chunk_ledger, chunk_times, times, chunk_audio)  # Add KWETier

    if include_old:
        textgrid_list = transfer_old_handlabels(bird_id, session, textgrid_list,
                                                chunk_ledger, epoch_times, chunk_times, chunk_audio)  # Add Old Labels

    # Save the TextGrids in the .wav directory
    chunk_wav_folder = Path('/net/expData/birdSong/ss_data_chunk_wav')  # Root Directory of all chunk .wav files
    assert chunk_wav_folder.parent.exists(), "The Directory Doesn't Exist. There is likely a typo in you path"

    wav_path = chunk_wav_folder / bird_id / session / activity_type  # .wav director
    save_chunks_textgrids(textgrid_list, wav_path)




