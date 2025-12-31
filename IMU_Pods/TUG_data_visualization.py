import csv
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.signal import butter, filtfilt

# TODO: Tune with fourier analysis, maybe determine if necessary?
def butter_bandpass(x, fs, upper=20, order=4):
    nyq = 0.5 * fs
    upper_bound = upper / nyq
    b, a = butter(order,upper_bound, btype="lowpass")
    return filtfilt(b, a, x)

def state_detector(gyr_x_data):
    # Initially tried 1 Hz, too much fluctuation in walking + turning signals made it somewhat unreliable, 0.5 Hz works better
    gyr_x_data = butter_bandpass(gyr_x_data,100,upper=0.5) # Aggressive filter to cleanly isolate turn
    max_idx = gyr_x_data.argmax()
    max_val = gyr_x_data[max_idx]
    THRESHOLDING_CONSTANT = 0.15 # Tune by averaging Gyr_X's
    threshold = THRESHOLDING_CONSTANT*max_val
    possible_bounds = np.where(gyr_x_data >= threshold)[0]
    left_threshold = possible_bounds[0]
    right_threshold = possible_bounds[-1]
    walking_until_turn = np.arange(0, left_threshold)
    turn = np.arange(left_threshold,right_threshold+1)
    walking_after_turn = np.arange(right_threshold+1,gyr_x_data.size)
    return gyr_x_data[walking_until_turn],gyr_x_data[turn],gyr_x_data[walking_after_turn]


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
    array = butter_bandpass(dataset["LB_Gyr_X"], 100) # Reduce noise
    pre_turn, turn, post_turn = state_detector(array)
    # plt.plot(array)
    # plt.show()