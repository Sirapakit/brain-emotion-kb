import numpy as np


file_path = './SASI_S01G2.npy'
sasi_results_for_subject = np.load(file_path)
print(f'Shape Original data is {sasi_results_for_subject.shape}')

constant_label = 1

labels = np.full((sasi_results_for_subject.shape[0], 1), constant_label)
print(f'Shape Labels is {labels.shape}')

data_with_labels = np.hstack((sasi_results_for_subject, labels))
np.save(f'./{file_path.split(".")[-2]}-labels.npy', data_with_labels)
print("Shape of data_with_labels:", data_with_labels)
