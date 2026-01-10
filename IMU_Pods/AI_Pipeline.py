import numpy as np
from pathlib import Path
from TUG_data_visualization import state_detector, butter_lowpass
import matplotlib.pyplot as plt
import pandas as pd

curr_path = Path.cwd() / "Processed_Data"

total = len(list(curr_path.iterdir()))

bad = 0

def normalize(val):
    new_val = val.copy()
    channels = new_val.shape[1]
    if channels:
        for i in range(channels):
            mean = np.mean(new_val[:, i])
            std = np.std(new_val[:, i])
            new_val[:, i] = (new_val[:, i] - mean) / (std + 1e-8)
    else:
        mean = np.mean(new_val[:])
        std = np.std(new_val[:])
        new_val[:, 0] = (new_val[:] - mean) / (std + 1e-8)

    return new_val

def resample(norm_val, desired_len):
    n, c = norm_val.shape
    t_old = np.linspace(0.0, 1.0, n, dtype=np.float64)
    t_new = np.linspace(0.0, 1.0, desired_len, dtype=np.float64)
    x = norm_val.astype(np.float64, copy=False)
    y = np.empty((desired_len, c), dtype=np.float64)
    for ch in range(c):
        y[:, ch] = np.interp(t_new, t_old, x[:, ch])
    return y


file_df = pd.read_csv("dataset.csv")

walk1 = np.array([])


walk1_vals = []
turn_vals = []
walk2_vals = []

rows = []
full_walks = []

for c in file_df.itertuples(index=False):
    name = c.Patient
    arr = np.load(c.Data_Path,allow_pickle=True)
    if np.isnan(arr).any():
        arr = np.nan_to_num(arr, nan=0.0)
    processed_arr = normalize(arr)
    walk1 = resample(processed_arr[c.Gait_Start:c.UTurn_Start],2000)
    turn = resample(processed_arr[c.UTurn_Start:c.UTurn_End+1],2000)
    walk2 = resample(processed_arr[c.UTurn_End+1:c.Gait_End],2000)
    six_k_resample = np.concatenate((walk1[:, 0],turn[:, 0],walk2[:, 0]))
    full_walks.append(six_k_resample)
    rows.append({"walk1": walk1[:, 0], "turn": turn[:, 0], "walk2": walk2[:, 0]})
    # lb_gyr_x = processed_arr[:, 0]

df = pd.DataFrame(rows)
N = len(full_walks)
sum_walks = np.zeros(len(full_walks[0]))
min_val = 10e99
for f in full_walks:
    min_val = min(min_val,min(sum_walks))
    sum_walks = sum_walks + f
avg_walks = sum_walks / N

print(min_val)
plt.plot(avg_walks)
plt.show()

plt.plot(full_walks[0])
plt.show()

pd.set_option("display.max_rows", 10000000000000)
# print(df.iloc[0])
# print(f"Bad split percent {round((bad/total) * 100,2)}%")


'''

    state detection code, not necessary right now b/c of existing metadata info

    a,b,c = state_detector(lb_gyr_x)
    if type(a) != type(np.array([])):
        print("bruh")
        plt.plot(lb_gyr_x)
        plt.show()
        continue
    T = len(arr)

    MIN_WALK = int(0.05 * T)
    MIN_TURN = int(0.02 * T)
    MAX_TURN = int(0.25 * T)

    not_good = (a.size < MIN_WALK or c.size < MIN_WALK or b.size < MIN_TURN or b.size > MAX_TURN)
    if not_good:
        bad += 1
        # print("uh oh, smth went wrong")
        # print(a.size, b.size, c.size)
        d_a,d_b,d_c = state_detector(lb_gyr_x,debug=True)
        debug_gyr_x = abs(butter_lowpass(lb_gyr_x,100,upper=0.5))
        maximum = debug_gyr_x[debug_gyr_x.argmax()]
        # print(debug_gyr_x.argmax())
        cutoff = maximum*0.15
        line = [cutoff for _ in range(len(debug_gyr_x))]
    '''

'''
Segmentation tests (not necessary right away lol): 

Initial Error rate: 54.35%
Error rate after removing endpoints from bound consideration: 23.29%
'''