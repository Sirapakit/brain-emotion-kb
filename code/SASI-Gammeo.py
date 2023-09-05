import numpy as np
import pandas as pd 
from scipy.signal import welch

SUBJECT_ID = 'S08'
GAME_ID = 'G4'
LABEL = 1
DATA_PATH = '../database'

data = pd.read_csv(DATA_PATH + '/' + SUBJECT_ID + '/' + SUBJECT_ID + GAME_ID + 'AllRawChannels.csv')

# Grab data 6 channles
data_AF3 = data['AF3']
data_AF4 = data['AF4']
data_F3 = data['F3']
data_F4 = data['F4']
data_F7 = data['F7']
data_F8 = data['F8']
data_6_channels = np.vstack((data_AF3, data_AF4, data_F3, data_F4, data_F7, data_F8))
print(f'Grabbing data with shape = {data_6_channels.shape}')
# np.save(f'./tools/EEG_{SUBJECT_ID}{GAME_ID}', data_6_channels)
del data, data_AF3, data_AF4, data_F3, data_F4, data_F7, data_F8

# Step 2: Calculate power spectral density using Welch's method
fs = 128
def calculate_power_spectral_density(eeg_signal):
    f, Pxx = welch(eeg_signal, fs=fs, nperseg=fs, noverlap=fs/2)
    return f, Pxx
# print(f'Calculate Power Spectral Densitty is {calculate_power_spectral_density(data_6_channels[0])}')

# Step 3: Calculate SASI for one subject window per window
def calculate_sasi_for_subject(subject_eeg_data):
    sasi_values = []

    for window_start in range(0, subject_eeg_data.shape[1], 128 * 4 // 2):
        window_end = window_start + 128 * 4
        if window_end <= subject_eeg_data.shape[1]:
            windowed_data = subject_eeg_data[:, window_start:window_end]
            
            channel_sasi = calculate_sasi_for_windowed_data(windowed_data)
            sasi_values.append(channel_sasi)

    return np.array(sasi_values)

def calculate_sasi_for_windowed_data(windowed_data):
    sasi_values = []

    for channel in range(windowed_data.shape[0]):
        f, Pxx = calculate_power_spectral_density(windowed_data[channel])

        central_band_idx = np.logical_and(f >= 8, f <= 13)
        max_freq_idx = np.argmax(Pxx[central_band_idx])
        max_freq = f[central_band_idx][max_freq_idx]

        lower_band = np.logical_and(f >= max_freq - 6, f <= max_freq - 4)
        higher_band = np.logical_and(f >= max_freq + 2, f <= max_freq + 26)

        lower_power = np.trapz(Pxx[lower_band], f[lower_band])
        higher_power = np.trapz(Pxx[higher_band], f[higher_band])

        sasi = (higher_power - lower_power) / (higher_power + lower_power)
        sasi_values.append(sasi)

    return np.array(sasi_values)

print(f'Calculating SASI for {SUBJECT_ID}{GAME_ID}')
sasi_results_for_subject = calculate_sasi_for_subject(data_6_channels)

print("SASI Results for Subject:")
print(sasi_results_for_subject.shape)
# np.save(f'./tools/SASI_{SUBJECT_ID}{GAME_ID}', sasi_results_for_subject)

# Add labels
labels = np.full((sasi_results_for_subject.shape[0], 1), LABEL)
print(f'Shape Labels is {labels.shape}')

data_with_labels = np.hstack((sasi_results_for_subject, labels))
np.save(f'../database/valence-SASI/SASI_{SUBJECT_ID}{GAME_ID}-labels.npy', data_with_labels)
print("Shape of data_with_labels:", data_with_labels.shape)