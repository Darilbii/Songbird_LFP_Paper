import BirdSongToolbox.GetBirdData as gbd

import BirdSongToolbox.epoch_raw_data as erd

from BirdSongToolbox.file_utility_functions import _save_pckl_data

import h5py
import os
from src.sanity_checks.data_check_first_pass import _get_true_starts


def main_for_1day(bird_id, session, neural_chans: list, audio_chan: list,  verbose=False):
    """ Epoch (Chunk) Raw Chronic Data

    Parameters
    ----------
    bird_id : str
        subject id
    session : str
        the date of the recording
    neural_chans : list
        list of the channels(columns) of the .kwd file are neural channels
    audio_chan : list
        list of the channel(s)[column(s)] of the .kwd file that are audio channels
    verbose : bool
        If True the Function prints out useful statements, defaults to False

    Returns
    -------
    buff_chunks_neural : list, shape = [Chunk]->(channels, Samples)
        Neural Data that is Low-Pass Filter at 400 Hz and Downsampled to 1 KHz, list of 2darrays
    chunk_index_test : list, shape = [Chunk]->(absolute start, absolute end)
        List of the Absolute Start and End of Each Chunk for that Recordings Day
    chunk_ledger : list, shape = [Chunk]->(first epoch, ..., last epoch)
        Ledger of which epochs occur in each Chunk, Chunks that only contain one Epoch have a length of 1
    buff_chunks_audio : list, shape = [Chunk]->(Samples)
        Audio Data that is Band-Pass Filtered between

    Saves
    -----
    buff_chunks_neural : pckl
        Large_Epochs_Neural.pckl
    chunk_index_test : pckl
        Large_Epochs_Times.pckl
    chunk_ledger : pckl
        Epochs_Ledger.pckl
    buff_chunks_audio : pckl
        Large_Epochs_Audio.pckl

    """
    # Folder where birds are
    experiment_folder_path = '/net/expData/birdSong/ss_data'

    bird_folder_path = os.path.join(experiment_folder_path, bird_id)  # Folder for the bird

    day_folder = os.path.join(bird_folder_path, session)  # Folder for Session

    # [3] Select Kwik File and get its Data
    kwik_file = gbd.get_data_path(day_folder=day_folder, file_type='.kwik')  # Ask User to select Kwik File
    kwik_file_path = os.path.join(day_folder, kwik_file)  # Get Path to Selected Kwik File
    kwik_data = gbd.read_kwik_data(kwik_path=kwik_file_path, verbose=True)  # Make Dict of Data from Kwik File

    # [4] Select the Kwe file
    kwe = gbd.get_data_path(day_folder=day_folder, file_type='.kwe')  # Select KWE File
    kwe_file_path = os.path.join(day_folder, kwe)  # Get Path to Selected KWE File
    kwe_data = gbd.read_kwe_file(kwe_path=kwe_file_path, verbose=False)  # Read KWE Data into Dict

    # [5] Select the Kwd file
    kwd = gbd.get_data_path(day_folder=day_folder, file_type='.kwd')  # Select Kwd File
    kwd_file = h5py.File(os.path.join(day_folder, kwd), 'r')  # Read Kwd Data into Dict

    # Showing where data is coming from
    print('Getting Data from ', kwd)

    # TODO: Clean Up Documentation Below

    # [6] Get the Absolute Start Times for the Automated Labels
    times = _get_true_starts(kwe_data=kwe_data, kwik_data=kwik_data)

    # [7] Determine How to Chunk (Epoch) the Data using the Automated Labels
    chunks, chunk_ledger = erd.determine_chunks_for_epochs(times)

    # [8.1] Chunk the Neural data with a buffer, low pass filter then Downsample
    buff_chunks_neural, chunk_index_test = erd.epoch_lfp_ds_data(kwd_file=kwd_file, kwe_data=kwe_data, chunks=chunks,
                                                                 neural_chans=neural_chans, verbose=verbose)

    # [8.2] Save the Neural Epochs(Chunks) to pickle
    _save_pckl_data(data=buff_chunks_neural, data_name='Large_Epochs_Neural', bird_id=bird_id, session=session,
                    make_parents=True)

    # [8.3] Save the Register of the Absolute Start and End of Each Chunk
    _save_pckl_data(data=chunk_index_test, data_name="Large_Epochs_Times", bird_id=bird_id, session=session,
                    make_parents=True)

    # [8.4] Save the Ledger of which Labels occur in which Epochs
    _save_pckl_data(data=chunk_ledger, data_name="Epochs_Ledger", bird_id=bird_id, session=session, make_parents=True)

    # [8.5] Save the Neural Epochs to .mat
    # TODO: Use info from https://docs.scipy.org/doc/scipy/reference/tutorial/io.html

    # [9.1] Chunk the Audio data with a buffer, and Band Pass Filter noise
    buff_chunks_audio = erd.epoch_bpf_audio(kwd_file=kwd_file, kwe_data=kwe_data, chunks=chunks, audio_chan=audio_chan,
                                            verbose=verbose)
    # [9.2] Save the Chunk Audio
    _save_pckl_data(data=buff_chunks_audio, data_name="Large_Epochs_Audio", bird_id=bird_id, session=session)

