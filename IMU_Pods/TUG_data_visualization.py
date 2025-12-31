import csv
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.signal import butter, filtfilt

'''
with open('HC100_TUG.csv') as file:
    reader_object = csv.DictReader(file)
    idx = 0
    # arr = np.array(list(map(float, [row['acc_tot'] for row in reader_object])))
    # Calc magnitude: 'ACC_X(m/(s^2))', 'ACC_Y(m/(s^2))', 'ACC_Z(m/(s^2))'
    for reader in reader_object:
        print(reader.keys())
        break
    arr = np.array([float(row['LowerBack_Gyr_X']) for row in reader_object])
    for row in reader_object:
        a_x = float(row['ACC_X(m/(s^2))'])
        a_y = float(row['ACC_Y(m/(s^2))'])
        a_z = float(row['ACC_Z(m/(s^2))'])
        magnitude = math.sqrt(a_x**2 + a_y**2 + a_z**2)
        arr.append(magnitude)
    arr = np.array(arr)
    arr = arr-arr.mean()
    arr /= arr.std()
    plt.plot(arr)
    plt.show() '''

# 6 hz accordign to this paper: https://www.mdpi.com/2076-3417/15/4/2177

# TODO: Tune with fourier analysis, maybe determine if necessary?
def butter_bandpass(x, fs, highcut=20, order=4):
    nyq = 0.5 * fs
    high = highcut / nyq
    b, a = butter(order,high, btype="lowpass")
    return filtfilt(b, a, x)

with open('HS_2_2_processed_data.txt') as file:
    lines = file.readlines()
    dataset = None
    keys = None
    first_line = False
    idx = 0
    end_of_set = False
    for l in lines:
        if not first_line:
            dataset = {i: [] for i in l.rstrip('\n').split("\t")}
            keys = list(dataset.keys())
            first_line = True
        else:
            i_temp = 0
            parts = l.rstrip('\n').split("\t")
            if len(parts) < len(keys):
                parts += [''] * (len(keys) - len(parts))
            for num in [float(p) if p != '' else math.nan for p in parts]:
                dataset[keys[i_temp]].append(num)
                i_temp += 1
        idx += 1
    array = butter_bandpass(dataset["LB_Gyr_X"], 100)
    plt.plot(array)
    plt.show()
    # print(dataset.keys())
    # print(dataset)