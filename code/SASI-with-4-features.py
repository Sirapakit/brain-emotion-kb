import numpy as np
import pandas as pd
from scipy.signal import welch, filtfilt

SUBJECT_ID = 'S08'
GAME_ID = 'G4'
LABEL = 1
DATA_PATH = '../database'

# Load EEG data
data = pd.read_csv(DATA_PATH + '/' + SUBJECT_ID + '/' + SUBJECT_ID + GAME_ID + 'AllRawChannels.csv')

# Grab data 6 channels
data_AF3 = data['AF3']
data_AF4 = data['AF4']
data_F3 = data['F3']
data_F4 = data['F4']
data_F7 = data['F7']
data_F8 = data['F8']
data_6_channels = np.vstack((data_AF3, data_AF4, data_F3, data_F4, data_F7, data_F8))
print(f'Grabbing data with shape = {data_6_channels.shape}')
del data, data_AF3, data_AF4, data_F3, data_F4, data_F7, data_F8

# Load filters
FILTER_PATH = './filter_coefficients.csv'
filters = pd.read_csv(FILTER_PATH)

filter_band_1 = filters['Filter 1']
filter_band_2 = filters['Filter 2']
filter_band_3 = filters['Filter 3']
filter_band_4 = filters['Filter 4']

NUM_BANDS = 4
SAMPLING_FREQ = 128
WINDOW_LENGTH = 4
DATA_CHN = 6
SHIFTED_SEC = 2
NUM_HJORTH_PARAM = 3 * 4

# Apply filter and calculate filtered data
data_filtered_array = np.zeros(((DATA_CHN*NUM_BANDS), data_6_channels.shape[1]))
for i in range(DATA_CHN):
    data_filtered_array[i*NUM_BANDS+0][:]  = filtfilt(filter_band_1, 1, data_6_channels[i])
    data_filtered_array[i*NUM_BANDS+1][:]  = filtfilt(filter_band_2, 1, data_6_channels[i])
    data_filtered_array[i*NUM_BANDS+2][:]  = filtfilt(filter_band_3, 1, data_6_channels[i])
    data_filtered_array[i*NUM_BANDS+3][:]  = filtfilt(filter_band_4, 1, data_6_channels[i])
print(f'Filtered data shape: {data_filtered_array.shape}')

# Should have 78 feature + 1 label = 79 rows

# Define parameters
fs = 128
WINDOW_LENGTH = 4
OVERLAPPING = 2

# Step 2: Calculate power spectral density using Welch's method
def calculate_power_spectral_density(eeg_signal):
    f, Pxx = welch(eeg_signal, fs=fs, nperseg=fs, noverlap=fs//2)
    return f, Pxx

# Step 3: Calculate SASI for one subject window per window
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

# Define functions for energy, Hjorth parameters, and entropy
def calculate_energy(eeg_signal):
    energy = np.sum(np.square(eeg_signal))
    return energy

def calculate_hjorth_parameters(eeg_signal):
    return np.var(eeg_signal)

def calculate_entropy(eeg_signal):
    normalized_squared_D = np.square(eeg_signal) / np.sum(np.square(eeg_signal))
    entropy = -np.sum(normalized_squared_D * np.log2(normalized_squared_D))
    return entropy

# Calculate features for each window
features_windowed_array = []
for window_start in range(0, data_filtered_array.shape[1], fs * WINDOW_LENGTH // 2):
    window_end = window_start + fs * WINDOW_LENGTH
    if window_end <= data_filtered_array.shape[1]:
        windowed_data = data_filtered_array[:, window_start:window_end]

        energy_values = np.apply_along_axis(calculate_energy, axis=1, arr=windowed_data)
        hjorth_parameters = np.apply_along_axis(calculate_hjorth_parameters, axis=1, arr=windowed_data)
        entropy_values = np.apply_along_axis(calculate_entropy, axis=1, arr=windowed_data)
        
        window_features = np.hstack((energy_values, hjorth_parameters, entropy_values))
        features_windowed_array.append(window_features)

features_windowed_array = np.array(features_windowed_array)

# Calculate sasi_values
sasi_values = calculate_sasi_for_subject(data_filtered_array)

# Reshape and stack the arrays
energy_hjorth_entropy = features_windowed_array.reshape(features_windowed_array.shape[0], -1)  # Flatten the features
combined_features = np.hstack((energy_hjorth_entropy, sasi_values))
print(f'Combined Features Shape: {combined_features.shape}')

# Save features
labels = np.full((features_windowed_array.shape[0], 1), LABEL)
features_with_labels = np.hstack((features_windowed_array, labels))
# np.save(f'../database/features/{SUBJECT_ID}{GAME_ID}_features_windowed.npy', features_with_labels)
np.save(f'./{SUBJECT_ID}{GAME_ID}_features_windowed.npy', features_with_labels)
print("Shape of Features with Labels:", features_with_labels.shape)
