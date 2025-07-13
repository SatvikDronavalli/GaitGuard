import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.signal import resample
import os

for file in os.listdir("./New_Data"):
    # Load the dataset
    patient = file.split("_")[0]
    new_file = os.path.join("./New_Data", file)
    print(new_file)
    df = pd.read_csv(new_file,low_memory=False)
    if "RTotalForce" not in df.columns:
        continue
    metadata = pd.read_csv("CONTROLS - Demographic+Clinical - datasetV1.csv")
    # print(metadata[metadata.columns[8]])
    weight = float(metadata.loc[metadata[metadata.columns[0]] == patient][metadata.columns[8]].tolist()[0])*9.81
    # Remove 'Standing' periods
    df = df[df["GeneralEvent"] != "Standing"].reset_index(drop=True)
    new_df = df.iloc[:df[df["GeneralEvent"] == "Turn"].index[0]]
    plt.plot(new_df["RTotalForce"])
    plt.show()
    # Identify all sections where the subject is walking straight (i.e., 'Walk' events only)
    walk_df = df[df["GeneralEvent"] == "Walk"].reset_index(drop=True)
    '''
    # Parameters for step detection
    threshold = 50
    min_length = 30
    max_length = 120

    r_force = []
    r_acc_x = []
    r_acc_y = []
    r_acc_z = []
    r_gyr_x = []
    r_gyr_y = []
    r_gyr_z = []
    r_cop_x = []
    r_cop_y = []
    def normalize_to_101_points(curve):
        """
        Interpolates a 1D array (e.g., a GRF curve) to have exactly 101 points.
        """
        original_length = len(curve)
        original_x = np.linspace(0, 1, original_length)
        target_x = np.linspace(0, 1, 101)

        f = interp1d(original_x, curve, kind='linear')  # or 'cubic' for smoother interpolation
        normalized_curve = f(target_x)

        return normalized_curve

    while True:
        stepStarted = False
        started_idx = 0
        ended_idx = 0
        found_step = False

        for i in range(0, len(walk_df["LTotalForce"]) - 1):
            # Checks if a step started
            if not stepStarted and pd.notna(walk_df.iloc[i]["LTotalForce"]) and pd.notna(walk_df.iloc[i+1]["LTotalForce"]) and \
               walk_df.iloc[i]["LTotalForce"] < threshold and walk_df.iloc[i+1]["LTotalForce"] >= threshold:
                stepStarted = True
                started_idx = i
                ended_idx = i
                found_step = True
            # Checks if a step ended
            elif stepStarted and pd.notna(walk_df.iloc[i]["LTotalForce"]) and pd.notna(walk_df.iloc[i+1]["LTotalForce"]) and \
                 walk_df.iloc[i]["LTotalForce"] < threshold and walk_df.iloc[i+1]["LTotalForce"] < threshold:
                break
            elif stepStarted:
                ended_idx += 1

        if not found_step or ended_idx <= started_idx or ended_idx + 2 > len(walk_df):
            break

        new_df = walk_df.iloc[started_idx:ended_idx+2].reset_index(drop=True)

        if max_length >= len(new_df) >= min_length:
            if not new_df["LTotalForce"].isna().any():
                r_force.append(new_df["LTotalForce"].values / weight)
            if not new_df["Linsole:Acc_X"].isna().any():
                r_acc_x.append(new_df["Linsole:Acc_X"].values)
            if not new_df["Linsole:Acc_Y"].isna().any():
                r_acc_y.append(new_df["Linsole:Acc_Y"].values)
            if not new_df["Linsole:Acc_Z"].isna().any():
                r_acc_z.append(new_df["Linsole:Acc_Z"].values)
            if not new_df["Linsole:Gyr_X"].isna().any():
                r_gyr_x.append(new_df["Linsole:Gyr_X"].values)
            if not new_df["Linsole:Gyr_Y"].isna().any():
                r_gyr_y.append(new_df["Linsole:Gyr_Y"].values)
            if not new_df["Linsole:Gyr_Z"].isna().any():
                r_gyr_z.append(new_df["Linsole:Gyr_Z"].values)
            if not new_df["LCoP_X"].isna().any():
                r_cop_x.append(new_df["LCoP_X"].values)
            if not new_df["LCoP_Y"].isna().any():
                r_cop_y.append(new_df["LCoP_Y"].values)

        walk_df = walk_df.iloc[ended_idx+2:].reset_index(drop=True)
        if len(walk_df) <= 100:
            break
    if r_force[-1][-1] > (threshold / weight):
        r_force.pop()
    columns = [f"T{i}" for i in range(0, 101)]
    r_force = [pd.DataFrame(normalize_to_101_points(c).reshape(1,-1),columns=columns) for c in r_force]
    r_acc_x = [pd.DataFrame(normalize_to_101_points(c).reshape(1,-1),columns=columns) for c in r_acc_x]
    r_acc_y = [pd.DataFrame(normalize_to_101_points(c).reshape(1,-1),columns=columns) for c in r_acc_y]
    r_acc_z = [pd.DataFrame(normalize_to_101_points(c).reshape(1,-1),columns=columns) for c in r_acc_z]
    r_gyr_x = [pd.DataFrame(normalize_to_101_points(c).reshape(1,-1),columns=columns) for c in r_gyr_x]
    r_gyr_y = [pd.DataFrame(normalize_to_101_points(c).reshape(1,-1),columns=columns) for c in r_gyr_y]
    r_gyr_z = [pd.DataFrame(normalize_to_101_points(c).reshape(1,-1),columns=columns) for c in r_gyr_z]
    r_cop_x = [pd.DataFrame(normalize_to_101_points(c).reshape(1,-1),columns=columns) for c in r_cop_x]
    r_cop_y = [pd.DataFrame(normalize_to_101_points(c).reshape(1,-1),columns=columns) for c in r_cop_y]
    for r in r_force:
        r['Fall Risk'] = 0
        write_header =  os.path.getsize("right_grf_dataset.csv") == 0
        r.to_csv("right_grf_dataset.csv",mode='a',header=write_header,index=False)
    for r in r_acc_x:
        r['Fall Risk'] = 0
        write_header = os.path.getsize("right_acc_x_dataset.csv") == 0
        r.to_csv("right_acc_x_dataset.csv", mode='a', header=write_header, index=False)
    for r in r_acc_y:
        r['Fall Risk'] = 0
        write_header = os.path.getsize("right_acc_y_dataset.csv") == 0
        r.to_csv("right_acc_y_dataset.csv", mode='a', header=write_header, index=False)
    for r in r_acc_z:
        r['Fall Risk'] = 0
        write_header = os.path.getsize("right_acc_z_dataset.csv") == 0
        r.to_csv("right_acc_z_dataset.csv", mode='a', header=write_header, index=False)
    for r in r_gyr_x:
        r['Fall Risk'] = 0
        write_header = os.path.getsize("right_gyr_x_dataset.csv") == 0
        r.to_csv("right_gyr_x_dataset.csv", mode='a', header=write_header, index=False)
    for r in r_gyr_y:
        r['Fall Risk'] = 0
        write_header = os.path.getsize("right_gyr_y_dataset.csv") == 0
        r.to_csv("right_gyr_y_dataset.csv", mode='a', header=write_header, index=False)
    for r in r_gyr_z:
        r['Fall Risk'] = 0
        write_header = os.path.getsize("right_gyr_z_dataset.csv") == 0
        r.to_csv("right_gyr_z_dataset.csv", mode='a', header=write_header, index=False)
    for r in r_cop_x:
        r['Fall Risk'] = 0
        write_header = os.path.getsize("right_cop_x_dataset.csv") == 0
        r.to_csv("right_cop_x_dataset.csv", mode='a', header=write_header, index=False)
    for r in r_cop_y:
        r['Fall Risk'] = 0
        write_header = os.path.getsize("right_cop_y_dataset.csv") == 0
        r.to_csv("right_cop_y_dataset.csv", mode='a', header=write_header, index=False) '''
