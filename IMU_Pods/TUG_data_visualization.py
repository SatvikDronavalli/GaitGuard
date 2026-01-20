import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.signal import butter, filtfilt
from pathlib import Path

def butter_lowpass(x, fs, upper=20, order=4):
    nyq = 0.5 * fs
    upper_bound = upper / nyq
    b, a = butter(order,upper_bound, btype="lowpass")
    return filtfilt(b, a, x)

def state_detector(gyr_x_data,debug=False):
    # Initially tried 1 Hz, too much fluctuation in walking + turning signals made it somewhat unreliable, 0.5 Hz works better
    gyr_x_data = abs(butter_lowpass(gyr_x_data,100,upper=0.5)) # Aggressive filter to cleanly isolate turn
    max_idx = gyr_x_data.argmax()
    max_val = gyr_x_data[max_idx]
    THRESHOLDING_CONSTANT = 0.15 # Tune by averaging Gyr_X's
    threshold = THRESHOLDING_CONSTANT*max_val
    possible_bounds = np.where(gyr_x_data >= threshold)[0]
    T = len(gyr_x_data)
    mask = (possible_bounds >= int(0.1 * T)) & (possible_bounds <= int(0.9 * T))
    possible_bounds = possible_bounds[mask]
    print(possible_bounds)
    print(max_idx)
    if not np.where(possible_bounds == max_idx)[0]:
        return -1,-1,-1
    second_half = possible_bounds[np.where(possible_bounds == max_idx)[0][0]+1:]
    first_half = possible_bounds[:np.where(possible_bounds == max_idx)[0][0]]
    prev = None
    right_bound = None
    for val in second_half:
        if prev is None:
            prev = val
            continue
        if val != prev+1:
            right_bound = prev
            break
        prev = val
    if right_bound is None:
        right_bound = second_half[-1]
    left_bound = None
    prev = None
    for val in first_half[::-1]:
        if prev is None:
            prev = val
            continue
        if val != prev-1:
            left_bound = prev
            break
        prev = val
    if left_bound is None:
        left_bound = first_half[0]
    left_threshold = left_bound
    right_threshold = right_bound
    # left_threshold = possible_bounds[0]
    # right_threshold = possible_bounds[-1]
    walking_until_turn = np.arange(0, left_threshold)
    turn = np.arange(left_threshold,right_threshold+1)
    walking_after_turn = np.arange(right_threshold+1,gyr_x_data.size)
    if debug:
        print(f"Maximum value: {max_val}")
        print(f"Left bound: {left_threshold.size}")
        print(f"Right bound: {right_threshold.size}")
        plt.plot(gyr_x_data)
        thresh_line = [threshold for _ in range(len(gyr_x_data))]
        plt.plot(thresh_line)
        plt.show()
    return walking_until_turn, turn, walking_after_turn


def dataset_preprocessing(file):
    # PathLib object!!!
    p = Path(file)
    lines = p.read_text(encoding="utf-8").splitlines()
    dataset = None
    keys = None
    first_line = False
    idx = 0
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
    return dataset

smth = 'HS_2_2_processed_data.txt'
with open('HS_2_2_processed_data.txt') as file:
    dataset = dataset_preprocessing(Path(smth))
    array = butter_lowpass(dataset["LB_Gyr_X"], 100) # Reduce noise
    pre_turn, turn, post_turn = state_detector(array)
    pre_turn = array[pre_turn]
    turn = array[turn]
    post_turn = array[post_turn]
    # plt.plot(array)
    # plt.show()
