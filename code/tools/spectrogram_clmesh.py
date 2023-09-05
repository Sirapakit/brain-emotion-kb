from scipy.signal import spectrogram
import matplotlib.pyplot as plt
import numpy as np

# Replace this with your actual EEG data
eeg_data_1 = np.load('./EEG_S03G4.npy')
eeg_data_2 = np.load('./EEG_S03G1.npy')

f, t, Sxx_1 = spectrogram(eeg_data_1, fs=128, nperseg=128, noverlap=128//2)
Sxx_1 = 10 * np.log10(Sxx_1)

f, t, Sxx_2 = spectrogram(eeg_data_2, fs=128, nperseg=128, noverlap=128//2)
Sxx_2 = 10 * np.log10(Sxx_2)

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 6))

# Plot for the first EEG data
pcm_1 = axes[0].pcolormesh(t, f, Sxx_1[0], shading='auto', cmap='seismic')
axes[0].set_ylabel('Frequency [Hz]')
axes[0].set_xlabel('Time [sec]')
axes[0].set_title('Spectrogram of EEG Signals - Subject 1')
axes[0].set_ylim(2, 30)


# Plot for the second EEG data
pcm_2 = axes[1].pcolormesh(t, f, Sxx_2[0], shading='auto', cmap='seismic')
axes[1].set_ylabel('Frequency [Hz]')
axes[1].set_xlabel('Time [sec]')
axes[1].set_title('Spectrogram of EEG Signals - Subject 2')
axes[1].set_ylim(2, 30)

# Add a colorbar to the entire figure
cbar = fig.colorbar(pcm_2, ax=axes, label='Power/Frequency (dB/Hz)')

plt.tight_layout()
plt.show()
