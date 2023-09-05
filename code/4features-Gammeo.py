import pandas as pd 
import numpy as np 
from scipy import signal

SUBJECT_ID = 'S15'
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
# data_6_channels = np.vstack((data_AF3-data_AF4, data_F3-data_F4, data_F7-data_F8)) # for bipolar channels
print(f'Grabbing data with shape = {data_6_channels.shape}')
del data, data_AF3, data_AF4, data_F3, data_F4, data_F7, data_F8

# Grab filter
FILTER_PATH = './filter_coefficients.csv'
filters = pd.read_csv(FILTER_PATH)

filter_band_1 = filters['Filter 1']
filter_band_2 = filters['Filter 2']
filter_band_3 = filters['Filter 3']
filter_band_4 = filters['Filter 4']

NUM_BANDS = 4
SAMPLING_FREQ = 128
WINDOW_LENGTH = 4
DATA_CHN = 6 # Check this
SHIFTED_SEC = 2
NUM_HJORTH_PARAM = 3 * 4 # 3 Param in each bands 

# Filter Bank Convolution  (((data_chn*num_bands)+1, data_len))
data_filtered_array = np.zeros(((DATA_CHN*NUM_BANDS)+1, data_6_channels.shape[1]))
for i in range(DATA_CHN):
    data_filtered_array[i*NUM_BANDS+0][:]  = signal.filtfilt(filter_band_1, 1, data_6_channels[i])
    data_filtered_array[i*NUM_BANDS+1][:]  = signal.filtfilt(filter_band_2, 1, data_6_channels[i])      
    data_filtered_array[i*NUM_BANDS+2][:]  = signal.filtfilt(filter_band_3, 1, data_6_channels[i])
    data_filtered_array[i*NUM_BANDS+3][:]  = signal.filtfilt(filter_band_4, 1, data_6_channels[i])
print(data_filtered_array.shape)

# Add event channel
data_filtered_array[-1, :] = LABEL
print(f"Shape after Convolve is {data_filtered_array.shape}")
print(f'Count {LABEL}: {np.count_nonzero(data_filtered_array[-1]=={LABEL})}')


# Calculate Energy Features per each band
print(f"--------Calculating Energy {SUBJECT_ID}{GAME_ID}--------")
energy_array = np.zeros((int(data_filtered_array.shape[0] - 1), int(data_filtered_array.shape[1]/(SAMPLING_FREQ*SHIFTED_SEC))))

for feat in range(data_filtered_array.shape[0] - 1):
    # - 1: dont need event till the last features 
    START = 0
    END = SAMPLING_FREQ * WINDOW_LENGTH
    COUNT = 0
    OVERLAPPING = int(0.5 * END)

    if (feat != (NUM_BANDS+NUM_HJORTH_PARAM) * DATA_CHN):  
        while (END <= data_filtered_array.shape[1]):
            new_sub_array = np.zeros((1, int(END)))
            new_sub_array = data_filtered_array[feat][:][START: END]
            energy_one_band = np.sum(np.power(new_sub_array, 2))
            energy_array[feat][COUNT] = energy_one_band
            START += OVERLAPPING
            END += OVERLAPPING
            COUNT += 1
            # print(f'Feat {feat} done at chn {COUNT}!')

# Add label : No need till last features
# energy_array[-1, :] = LABEL
print(f'Shape of Energy array is {energy_array.shape}')
print(f'Count {LABEL}: {np.count_nonzero(energy_array[-1]=={LABEL})}')

####################################################
# Calculate 1st Hjorth Features per each band
print(f"--------Calculating First Hjorth Parameter {SUBJECT_ID}{GAME_ID}--------")
first_hjorth_array = np.zeros((int(data_filtered_array.shape[0] - 1), int(data_filtered_array.shape[1]/(SAMPLING_FREQ*SHIFTED_SEC))))

for feat in range(data_filtered_array.shape[0] - 1):
    # - 1: dont need event till the last features 
    START = 0
    END = SAMPLING_FREQ * WINDOW_LENGTH
    COUNT = 0
    OVERLAPPING = int(0.5 * END)

    if (feat != (NUM_BANDS+NUM_HJORTH_PARAM) * DATA_CHN):  
        while (END <= data_filtered_array.shape[1]):
            new_sub_array = np.zeros((1, int(END)))
            new_sub_array = data_filtered_array[feat][:][START: END]
            first_hjorth_one_band = np.var(new_sub_array)
            energy_array[feat][COUNT] = first_hjorth_one_band
            START += OVERLAPPING
            END += OVERLAPPING
            COUNT += 1
            # print(f'Feat {feat} done at chn {COUNT}!')

# Add label : No need till last features
# energy_array[-1, :] = LABEL
print(f'Shape of first_hjorth array is {first_hjorth_array.shape}')
print(f'Count {LABEL}: {np.count_nonzero(first_hjorth_array[-1]=={LABEL})}')

##############################################################
# Calculate 2nd Hjorth Features per each band
# def hjorth_mobility(signal):
#     diff = np.diff(signal)
#     return np.sqrt(np.var(diff) / np.var(signal))

# Calculate Entropy per each band
def calculate_entropy(D):
    normalized_squared_D = np.square(D) / np.sum(np.square(D))
    entropy = -np.sum(normalized_squared_D * np.log2(normalized_squared_D))
    return entropy

print(f"--------Calculating Second Hjorth Parameter {SUBJECT_ID}{GAME_ID}--------")
second_hjorth_array = np.zeros((int(data_filtered_array.shape[0] - 1), int(data_filtered_array.shape[1]/(SAMPLING_FREQ*SHIFTED_SEC))))

for feat in range(data_filtered_array.shape[0] - 1):
    # - 1: dont need event till the last features 
    START = 0
    END = SAMPLING_FREQ * WINDOW_LENGTH
    COUNT = 0
    OVERLAPPING = int(0.5 * END)

    if (feat != (NUM_BANDS+NUM_HJORTH_PARAM) * DATA_CHN):  
        while (END <= data_filtered_array.shape[1]):
            new_sub_array = np.zeros((1, int(END)))
            new_sub_array = data_filtered_array[feat][:][START: END]
            second_hjorth_one_band = calculate_entropy(new_sub_array)
            energy_array[feat][COUNT] = second_hjorth_one_band
            START += OVERLAPPING
            END += OVERLAPPING
            COUNT += 1
            # print(f'Feat {feat} done at chn {COUNT}!')

# Add label : No need till last features
# energy_array[-1, :] = LABEL
print(f'Shape of second_hjorth array is {second_hjorth_array.shape}')
print(f'Count {LABEL}: {np.count_nonzero(second_hjorth_array[-1]=={LABEL})}')

##############################################################
# Calculate 3rd Hjorth Features per each band
# def hjorth_complexity(signal):
#     diff1 = np.diff(signal)
#     diff2 = np.diff(diff1)
#     return np.sqrt(np.var(diff2) / np.var(diff1))

# print(f"--------Calculating Third Hjorth Parameter {SUBJECT_ID}{GAME_ID}--------")
# third_hjorth_array = np.zeros((int(data_filtered_array.shape[0] - 1), int(data_filtered_array.shape[1]/(SAMPLING_FREQ*SHIFTED_SEC))))

# for feat in range(data_filtered_array.shape[0] - 1):
#     # - 1: dont need event till the last features
#     START = 0
#     END = SAMPLING_FREQ * WINDOW_LENGTH
#     COUNT = 0
#     OVERLAPPING = int(0.5 * END)

#     if (feat != (NUM_BANDS+NUM_HJORTH_PARAM) * DATA_CHN):  
#         while (END <= data_filtered_array.shape[1]):
#             new_sub_array = np.zeros((1, int(END)))
#             new_sub_array = data_filtered_array[feat][:][START: END]
#             third_hjorth_one_band = hjorth_complexity(new_sub_array)
#             energy_array[feat][COUNT] = third_hjorth_one_band
#             START += OVERLAPPING
#             END += OVERLAPPING
#             COUNT += 1
#             # print(f'Feat {feat} done at chn {COUNT}!')

# # Add label : No need till last features
# # third_hjorth_array[-1, :] = LABEL
# print(f'Shape of third_hjorth array is {third_hjorth_array.shape}')
# print(f'Count {LABEL}: {np.count_nonzero(third_hjorth_array[-1]=={LABEL})}')

# Create label array 
label_array = np.repeat(LABEL, first_hjorth_array.shape[1])

# feature_array = np.vstack((energy_array, first_hjorth_array, second_hjorth_array, third_hjorth_array, label_array))
feature_array = np.vstack((energy_array, first_hjorth_array, second_hjorth_array, label_array))
print(f'Shape of feature array is {feature_array.shape}')


FILENAME = "../database/arousal-feature/3features-" + SUBJECT_ID + "-" + GAME_ID + "-with-labels.npy"
np.save(FILENAME, feature_array)
print(f'Count 0: {np.count_nonzero(feature_array[-1]==0)}')
print(f'Count 1: {np.count_nonzero(feature_array[-1]==1)}')
# Energy Array --> 1 loop
# First Hjorth ---> second loop 