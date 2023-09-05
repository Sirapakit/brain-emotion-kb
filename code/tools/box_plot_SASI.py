import matplotlib.pyplot as plt
import numpy as np

# Replace this with your actual sasi_results_for_subject data
sasi_results_for_subject = np.load('./SASI_S03G4.npy')
sasi_results_for_subject_2 = np.load('./SASI_S03G1.npy')

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 6))

# Plot for the first data
axes[0].boxplot(sasi_results_for_subject, labels=[f"Channel {i+1}" for i in range(sasi_results_for_subject.shape[1])])
axes[0].set_ylabel("SASI Value")
axes[0].set_title("SASI Values Distribution for Subject 1")

# Plot for the second data
axes[1].boxplot(sasi_results_for_subject_2, labels=[f"Channel {i+1}" for i in range(sasi_results_for_subject_2.shape[1])])
axes[1].set_ylabel("SASI Value")
axes[1].set_title("SASI Values Distribution for Subject 2")

plt.tight_layout()
plt.show()
