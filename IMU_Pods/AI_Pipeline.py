import numpy as np
from pathlib import Path
# from TUG_data_visualization import state_detector, butter_lowpass
import matplotlib.pyplot as plt
import pandas as pd

curr_path = Path.cwd() / "Processed_Data"
print(curr_path)

total = len(list(curr_path.iterdir()))

bad = 0

def norm_and_resample(x,desired_len):
    n, c = x.shape
    t_old = np.linspace(0.0, 1.0, n, dtype=np.float64)
    t_new = np.linspace(0.0,1.0,desired_len, dtype=np.float64)
    x = x.astype(np.float64, copy=False)
    y = np.empty((desired_len, c), dtype=np.float64)
    for ch in range(c):
        y[:, ch] = np.interp(t_new, t_old, x[:, ch])
        mu = np.mean(y[:, ch])
        std = np.std(y[:, ch])
        y[:, ch] = (y[:, ch] - mu) / (std + 1e-8) # Maybe calculate absolute features too?
    return y.astype(np.float32)

file_df = pd.read_csv("dataset.csv")

walk1 = np.array([])


for c in file_df.itertuples(index=False):
    name = c.Patient
    arr = np.load(c.Data_Path)
    if np.isnan(arr).any():
        arr = np.nan_to_num(arr, nan=0.0)
    walk1 = norm_and_resample(arr[c.Gait_Start:c.UTurn_Start],2000)
    turn = norm_and_resample(arr[c.UTurn_Start:c.UTurn_End+1],2000)
    walk2 = norm_and_resample(arr[c.UTurn_End+1:c.Gait_End],2000)
    np.save(c.Data_Path, {
        "pre_walk": walk1,
        "turn": turn,
        "post_walk": walk2,
     }, allow_pickle=True)
    '''
    
    lb_gyr_x = arr[:, 0]
    
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
        print(debug_gyr_x.argmax())
        cutoff = maximum*0.15
        line = [cutoff for _ in range(len(debug_gyr_x))] '''

'''
Segmentation tests (not necessary right away lol): 

Initial Error rate: 54.35%
Error rate after removing endpoints from bound consideration: 23.29%
'''
print("success")
# print(f"Bad split percent {round((bad/total) * 100,2)}%")