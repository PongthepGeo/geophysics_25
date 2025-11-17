from readgssi import readgssi
import matplotlib.pyplot as plt
import numpy as np
import os
# pip install readgssi

GPR_file = 'dataset/line_001.DZT'

# Read metadata using readgssi
metadata = readgssi.readgssi(infile=GPR_file, plotting=False)
# print(metadata)
# Select the data arrays from the metadata. The metadata is a list with two elements.
extracted_values = metadata[1][0]

print(extracted_values.shape)

# top = extracted_values[0:100, :]
# print(top.shape)
# plt.imshow(top, cmap='Greys')
# plt.show()
print(f'number of columns (traces): {extracted_values.shape[1]} and number of rows (time): {extracted_values.shape[0]}')
print(f'data type: {extracted_values.dtype}')

signal = np.zeros((extracted_values.shape[0], extracted_values.shape[1]))
signal[100:, :] = extracted_values[100:, :]

plt.imshow(signal, cmap='Greys')
plt.show()

trace = signal[:, 400]
axis_x = np.arange(0, trace.shape[0])
plt.plot(trace, axis_x)
plt.gca().invert_yaxis()
plt.show()
# # Plot the GPR data
# plt.figure(figsize=(15, 10))
# plt.imshow(extracted_values, cmap='Greys')
# plt.title('GPR at Accounting Department')
# plt.xlabel('Trace number')
# plt.ylabel('Two-way travel time (ms)')
# os.makedirs('figure_plot', exist_ok=True)
# plt.savefig('figure_plot/' + 'GPR' + '.svg', format='svg', bbox_inches='tight', transparent=True, pad_inches=0)
# plt.show()