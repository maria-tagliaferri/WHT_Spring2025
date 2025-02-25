import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks
import os

# Read CSV file and convert to numpy array
grf = np.genfromtxt(r"/Users/maria/Documents/WHT/Walk/4Run_rep0.csv", delimiter=',', skip_header=1)
headers = np.genfromtxt(r"/Users/maria/Documents/WHT/Walk/4Run_rep0.csv", delimiter=',', max_rows=1, dtype=str)

# Extract time data
time = grf[3:, 1]
acx = -1*grf[3:, 20] #change to pos 1 for subjects 1 and 3

# 5th order Zero Lag Butterworth LPF for Joint moment data
fs = 100       # sampling freq. (100Hz)
fc = 6         # cutoff freq. (6Hz)
Wn = fc / (fs / 2)
b, a = butter(5, Wn, 'low')  # 5th order

acx_f = filtfilt(b, a, acx)

# Find peaks in the filtered data
peaks_acx, _ = find_peaks(acx_f)

# Filter peaks based on a threshold
ind = np.where(acx_f[peaks_acx] > 12)
peaks_acx = peaks_acx[ind]

# Output the indices of the peaks
print("Indices of peaks in acx_f:", peaks_acx)

# Plot the filtered data and mark the peaks
plt.plot(time, acx_f, label='Filtered')
plt.plot(time[peaks_acx], acx_f[peaks_acx], "x", label='Peaks')
plt.legend()
plt.show()

def upsampling(data, target_length):
    """Upsample the data to the target length."""
    return np.interp(np.linspace(0, len(data) - 1, target_length), np.arange(len(data)), data)

def segmenting(data, gait_cycles):
    segmented_data = []
    for i in range(len(gait_cycles) - 1):
        current_gc = data[gait_cycles[i]:gait_cycles[i + 1]]
        segmented_data.append(upsampling(current_gc, 1000))
    
    segmented_data = np.array(segmented_data).T
    avg_data = np.mean(segmented_data, axis=1)
    return segmented_data, avg_data

def segment_and_export_all_columns(grf, peaks_acx, headers, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for i in range(len(peaks_acx) - 1):
        start_idx = peaks_acx[i]
        end_idx = peaks_acx[i + 1]
        current_gc = grf[start_idx:end_idx, :]
        
        # Upsample each column of the current gait cycle
        upsampled_gc = np.array([upsampling(current_gc[:, col], 1000) for col in range(current_gc.shape[1])]).T
        
        output_file = os.path.join(output_dir, f'gait_cycle_{i+1}.csv')
        np.savetxt(output_file, upsampled_gc, delimiter=',', header=','.join(headers), comments='')

# Segment and export the data from every column
output_dir = "/Users/maria/Documents/WHT/Walk/segmented_gait_cycles__sub1"
segment_and_export_all_columns(grf, peaks_acx, headers, output_dir)