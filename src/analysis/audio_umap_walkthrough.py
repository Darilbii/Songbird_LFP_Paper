from config import general_config, umap_config, spec_config
from utils import plot_melspec, pad_zeros, sung_with_female

import pickle as pkl
from BirdSongToolbox.import_data import ImportData
import librosa, librosa.display
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import noisereduce as nr

import umap
import umap.plot

bird_id, days = pkl.load(open('zf_list.pkl', 'rb'))

print('Bird:', general_config['bird'])  # 'Bird': 'z007'
rec_days = days[bird_id.index(general_config['bird'])]  # Basically np.where()
print('Recording days:', rec_days)

if spec_config['fft_center']:  # immported from the config file
    pad = spec_config['window'] / 2
else:
    pad = 0

########################################################################################################################
call_tracker = []
call_melspec = []

# list of bird vocalizations
bird_voc = {
    'z007': [1, 2, 3, 4, 5, 6, 'I', 'C'],
    'z017': [1, 2, 3, 4, 5, 6, 7, 'C'],
    'z020': [1, 2, 3, 4, 'I', 'C']
}

bird_voc_ledger = {}  # key - natural no. indicating the vocalization event, value - 'day_chunk_(voc)label_(voc)instance'
event_count = 1
bird_voc_event = {}
for i in bird_voc[general_config['bird']]:  # Counter for each labeled vocalization {vocalization: instance (1-indexed)}
    bird_voc_event[i] = 1
########################################################################################################################


call_event = 1  # event tracker for call

for rec_ind in range(len(rec_days)):  # For each Dayi
    data = ImportData(bird_id=general_config['bird'], session=rec_days[rec_ind], location=general_config['data_path'])
    audio = data.song_audio
    hand_labels = data.song_handlabels
    print("\nBird:", general_config['bird'], "Day:", rec_days[rec_ind], "No. of chunks:", len(audio))

    for chunk_ind in range(len(audio)):  # For Each Chunk

        lbl = hand_labels[chunk_ind]['labels'][0]
        time = hand_labels[chunk_ind]['labels'][1]

        # ADDITIONAL LEDGER WITH LABELS FOR VOCALIZATION
        # -----------------------------------------------------------------------------------------
        for i in range(len(lbl)):
            if lbl[i] in bird_voc[general_config['bird']]:
                bird_voc_ledger[event_count] = '{}_chunk{}_{}_{}'.format(rec_days[rec_ind], chunk_ind, lbl[i],
                                                                         bird_voc_event[lbl[i]])
                event_count += 1
                bird_voc_event[lbl[i]] += 1

        # FILTER AUDIO
        # -----------------------------------------------------------------------------------------
        if general_config['noisereduce']:
            noise_ind = len(lbl) - 1 - lbl[::-1].index(8)
            noise_buffer = audio[chunk_ind][time[0][noise_ind]:time[1][noise_ind]]
            audio[chunk_ind] = nr.reduce_noise(audio_clip=audio[chunk_ind], noise_clip=noise_buffer, verbose=False)

        # COMPUTE SPECTROGRAM
        # -----------------------------------------------------------------------------------------
        # spectrogram
        S_amp = librosa.stft(audio[chunk_ind], n_fft=spec_config['n_fft'], hop_length=spec_config['stride']
                             , win_length=spec_config['window'], window=spec_config['window_type']
                             , center=spec_config['fft_center'])
        S_pow = np.abs(S_amp) ** 2

        # compute mel spectrogram of from band-limited spectrogram
        M_pow = librosa.feature.melspectrogram(S=S_pow, sr=spec_config['sr'], n_mels=spec_config['n_mels'],
                                               fmin=spec_config['f_min'], fmax=spec_config['f_max'])
        print("Spectrogram shapes:", S_pow.shape, M_pow.shape)

        # TRANSLATE TIME FROM SAMPLING RATE TO SPECTROGRAM SCALE
        # -----------------------------------------------------------------------------------------
        # window the time periods
        shifted_time = [[], []]
        shifted_time[0] = [int((time[0][k] - spec_config['window'] + (2 * pad)) / spec_config['stride']) + 1 for k in
                           range(1, len(time[0]))]
        shifted_time[0].insert(0, 0)
        shifted_time[1] = [int((time[1][k] - spec_config['window'] + (2 * pad)) / spec_config['stride']) + 1 for k in
                           range(len(time[1]))]
        print("Shifted last time stamp:", shifted_time[1][-1])

        assert shifted_time[1][-1] == M_pow.shape[1]

        # plot mel spec
        #         plot_melspec(M_pow, shifted_time, lbl)

        # retrieve mel spectrograms of calls
        #         current_ledger = {key: value for key, value in bird_voc_ledger.items() if '{}_chunk{}'.format(rec_days[rec_ind], chunk_ind) in value}
        for i in range(len(lbl)):
            if lbl[i] == 'C':
                if not sung_with_female([time[0][i], time[1][i]], hand_labels[chunk_ind]['female'][1]):
                    assert M_pow[:, shifted_time[0][i]:shifted_time[1][i]].shape[1] > 0
                    call_melspec.append(np.log(M_pow[:, shifted_time[0][i]:shifted_time[1][i]]))
                    call_tracker.append(list(bird_voc_ledger.keys())[list(bird_voc_ledger.values()).index
                    ('{}_chunk{}_C_{}'.format(rec_days[rec_ind], chunk_ind, call_event))])
                call_event += 1

########################################################################################################################
