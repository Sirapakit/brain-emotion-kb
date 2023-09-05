import numpy as np
import scipy.signal as signal
import pandas as pd 

# Sampling rate
fs = 128.0  # Hz

# Define filter parameters
filter_order = 159  # You can adjust this based on your requirements

# Define filter cutoff frequencies
cutoffs = np.array([0.5, 4, 8, 14, 30])  # Band edges in Hz

# Normalize cutoff frequencies
normalized_cutoffs = cutoffs / (0.5 * fs)

# Design filters
filters = []
for i in range(len(normalized_cutoffs) - 1):
    band = [normalized_cutoffs[i], normalized_cutoffs[i + 1]]
    b = signal.firwin(filter_order, band, pass_zero=False, fs=fs)
    filters.append(b)

# Create a DataFrame to store the filter coefficients
filter_df = pd.DataFrame(columns=[f"Filter {i + 1}" for i in range(len(filters))])

# Add filter coefficients to the DataFrame
for i, b in enumerate(filters):
    filter_df[f"Filter {i + 1}"] = b

# Save the DataFrame to a CSV file
filter_df.to_csv("filter_coefficients.csv", index=False)

print("Filter coefficients saved to 'filter_coefficients.csv'")
