total_samples = 38252
sampling_rate = 128
seconds = 4
window_length_samples = sampling_rate * seconds  # 4 seconds with 128 Hz sampling rate
overlap_samples = window_length_samples // 2  # 50% overlapping

number_of_windows = (total_samples - window_length_samples) // overlap_samples + 1

print("Number of windows:", number_of_windows)
