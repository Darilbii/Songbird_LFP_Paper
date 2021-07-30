""" Chunk and Pre-process the Raw Data for Silent (Vocally Inactive Times)"""

from src.sanity_checks.data_check_first_pass import _get_true_starts

import BirdSongToolbox.GetBirdData as gbd
import BirdSongToolbox.epoch_raw_data as erd
from BirdSongToolbox.file_utility_functions import _save_pckl_data
from pathlib import Path
import scipy.io as sio

import h5py
import os


def get_silence_ledger(bird_id, session):
    """ get the silence ledger


    Parameters
    ----------
    bird_id : str

    session : str

    Returns
    -------
    mat_file_filt : ndarray
        Ledger of the Silent Epochs [Start, Stop, CWE]
    """

    prepd_ss_data_folder = Path('/net/expData/birdSong/ss_data_Processed')

    # Define Path to the User Designated Pre-Processed Data
    # desig_data_type = epoch_type + '_' + data_type  # Full Designated Data Name, ex: 'Song'+'LFP_DS'->'Song_LFP_DS'
    desig_data_type = 'All_Silence_Epoch_Times'
    spec_file_name = desig_data_type + '.mat'  # Name of Matlab Data with Specified Data
    data_file_path = prepd_ss_data_folder / bird_id / session / spec_file_name

    # Import the Data
    data_file_path.resolve()
    mat_file = sio.loadmat(str(data_file_path))  # Open and Import the specified Matlab File

    # Data is named different than file name
    data_iwant = 'All_Silences_epoch_times'
    mat_file_filt = mat_file[data_iwant]  # make the data easier to work with in python

    return mat_file_filt


def conv_silence_to_kwe_data(silence_ledger):
    kwe_data = dict()
    kwe_data['motif_st'] = silence_ledger[:,0]
    kwe_data['motif_rec_num'] = silence_ledger[:,2]
    return kwe_data


def main_for_1day_silence(bird_id, session, neural_chans: list, audio_chan: list, filter_buffer: int = 10,
                          data_buffer: int = 10,  verbose=False):
    """ Epoch (Chunk) Raw Chronic Data [For Vocally Inactive Times]

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
    filter_buffer : int, optional
        Time buffer in secs to be sacrificed for filtering, defaults to 10 secs
    data_buffer : int, optional
        Time buffer around time of interest to chunk data, defaults to 30 secs
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
    silence_ledger = get_silence_ledger(bird_id, session)  # Get Silent Epoch KWE Information #***#
    kwe_data = conv_silence_to_kwe_data(silence_ledger)  # Read Silent Epoch Data into KWE Dict

    # [5] Select the Kwd file
    kwd = gbd.get_data_path(day_folder=day_folder, file_type='.kwd')  # Select Kwd File
    kwd_file = h5py.File(os.path.join(day_folder, kwd), 'r')  # Read Kwd Data into Dict

    # Showing where data is coming from
    print('Getting Data from ', kwd)

    # TODO: Clean Up Documentation Below

    # [6] Get the Absolute Start Times for the Automated Labels
    times = _get_true_starts(kwe_data=kwe_data, kwik_data=kwik_data)

    # [7] Determine How to Chunk (Epoch) the Data using the Automated Labels
    chunks, chunk_ledger = erd.determine_chunks_for_epochs(times, search_buffer=10)

    # [8.1] Chunk the Neural data with a buffer, low pass filter then Downsample
    buff_chunks_neural, chunk_index_test = erd.epoch_lfp_ds_data(kwd_file=kwd_file, kwe_data=kwe_data, chunks=chunks,
                                                                 neural_chans=neural_chans, filter_buffer=filter_buffer,
                                                                 data_buffer=data_buffer, verbose=verbose)

    # [8.2] Save the Neural Epochs(Chunks) to pickle
    _save_pckl_data(data=buff_chunks_neural, data_name='Large_Epochs_Neural_Silence', bird_id=bird_id, session=session,
                    make_parents=True)

    # [8.3] Save the Register of the Absolute Start and End of Each Chunk
    _save_pckl_data(data=chunk_index_test, data_name="Large_Epochs_Times_Silence", bird_id=bird_id, session=session,
                    make_parents=True)

    # [8.4] Save the Ledger of which Labels occur in which Epochs
    _save_pckl_data(data=chunk_ledger, data_name="Epochs_Ledger_Silence", bird_id=bird_id, session=session,
                    make_parents=True)

    # [8.5] Save the Neural Epochs to .mat
    # TODO: Use info from https://docs.scipy.org/doc/scipy/reference/tutorial/io.html

    # [9.1] Chunk the Audio data with a buffer, and Band Pass Filter noise
    buff_chunks_audio = erd.epoch_bpf_audio(kwd_file=kwd_file, kwe_data=kwe_data, chunks=chunks, audio_chan=audio_chan,
                                            filter_buffer=filter_buffer, data_buffer=data_buffer, verbose=verbose)
    # [9.2] Save the Chunk Audio
    _save_pckl_data(data=buff_chunks_audio, data_name="Large_Epochs_Audio_Silence", bird_id=bird_id, session=session)
