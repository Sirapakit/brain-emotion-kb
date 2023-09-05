import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import cwt

# Sample rate and time points
sampling_rate = 128  # Hz
duration = 10  # seconds
num_samples = sampling_rate * duration
time = np.linspace(0, duration, num_samples)

# Create a sample signal (you can replace this with your own signal)
frequency = 20  # Hz
signal = np.sin(2 * np.pi * frequency * time)

# Import Daubechies wavelet from pywt
import pywt

# Define the parameters for the Daubechies wavelet
wavelet_name = 'db4'
wavelet = pywt.Wavelet(wavelet_name)

# Perform continuous wavelet transform using db4 wavelet
scales = np.arange(1, 160)
coeffs = cwt(signal, wavelet, scales)

# Extract the coefficients corresponding to each frequency band
bands = [(4, 8), (8, 16), (16, 32), (32, 64), (64, np.inf)]  # Frequency bands in Hz
band_coeffs = []

for band in bands:
    min_freq, max_freq = band
    min_scale = sampling_rate / max_freq
    max_scale = sampling_rate / min_freq
    relevant_scales = np.where((min_scale <= scales) & (scales <= max_scale))
    band_coeff = np.sum(np.abs(coeffs[relevant_scales, :]), axis=0)
    band_coeffs.append(band_coeff)

# Plot the extracted bands
plt.figure(figsize=(10, 6))

for i, band_coeff in enumerate(band_coeffs):
    plt.plot(time, band_coeff, label=f'Band {i+1}: {bands[i][0]}-{bands[i][1]} Hz')

plt.xlabel('Time (s)')
plt.ylabel('Magnitude')
plt.title(f'Extracted Frequency Bands using {wavelet_name.capitalize()} Wavelet')
plt.legend()
plt.grid(True)
plt.show()
