import pandas as pd
import json
import numpy as np
from pathlib import Path
from TUG_data_visualization import dataset_preprocessing, butter_lowpass
import matplotlib.pyplot as plt

DATASET_LOCATION = "C:/Users/Satvik/Downloads" # Change this as necessary
healthy_dir = Path(f"{DATASET_LOCATION}/dataset/dataset/data/healthy")
neuro_dir = Path(f"{DATASET_LOCATION}/dataset/dataset/data/neuro")
ortho_dir = Path(f"{DATASET_LOCATION}/dataset/dataset/data/ortho")
dataset_dir = Path(f"{DATASET_LOCATION}/dataset/dataset/data")
dataset = pd.DataFrame()

# Clears out dataset folder for easier testing

folder = Path("Processed_Data")
for item in folder.iterdir():
    if item.is_file():
        item.unlink()

# Extracting Visual Gait Assessment (VGA) results

rows = []

for t in dataset_dir.iterdir():
    for condition in t.iterdir():
        for item in condition.iterdir():
            patient = item.name
            trial_amt = 0
            total_vga = 0
            no_vga = False
            for trial in item.iterdir():
                metadata = list(trial.iterdir())[0]
                with open(metadata,'r') as file:
                    data = json.load(file)
                    if data['visualGaitAssessment'] == 'Not evaluated':
                        no_vga = True
                        break
                total_vga += float(data['visualGaitAssessment'])
                trial_amt += 1

            trial_idx = 1
            for trial in item.iterdir():
                imu_data = dataset_preprocessing(list(trial.iterdir())[2])
                metadata = list(trial.iterdir())[0]
                with open(metadata, 'r') as file:
                    data = json.load(file)
                    curr_vga = data['visualGaitAssessment']
                    start = min(data["leftGaitEvents"][0][1],data["rightGaitEvents"][0][1])
                    end = max(data["leftGaitEvents"][-1][1],data["rightGaitEvents"][-1][1])
                    turn_bounds = data["uturnBoundaries"]
                    if curr_vga == 'Not evaluated':
                        # print(f"{patient} didn't have a VGA evaluation")
                        break
                    elif data['protocol'] != '10.0m - uturn - 10.0m':
                        # print(f"{patient} has protocol {data['protocol']} instead of the 10m walk test")
                        break
                    curr_vga = float(curr_vga)
                lb_imu_data_x = butter_lowpass(imu_data["LB_Gyr_X"],100)
                copy_x = lb_imu_data_x.copy()
                mid_bound = int((turn_bounds[1] + 1 - turn_bounds[0])/2)
                flip = -1 if lb_imu_data_x[mid_bound] < 0 else 1
                # TODO: Check if flipping makes it hard to generalize on collected data
                lb_imu_data_x = np.concatenate([lb_imu_data_x[:turn_bounds[0]],
                                               lb_imu_data_x[turn_bounds[0]:turn_bounds[1] + 1] * flip,
                                               lb_imu_data_x[turn_bounds[1] + 1:]])
                lb_imu_data_y = butter_lowpass(imu_data["LB_Gyr_Y"], 100)
                lb_imu_data_y = np.concatenate([lb_imu_data_y[:turn_bounds[0]],
                                               lb_imu_data_y[turn_bounds[0]:turn_bounds[1] + 1] * flip,
                                               lb_imu_data_y[turn_bounds[1] + 1:]])
                lb_imu_data_z = butter_lowpass(imu_data["LB_Gyr_Z"], 100)

                lb_imu_data_z = np.concatenate([lb_imu_data_z[:turn_bounds[0]],
                                               lb_imu_data_z[turn_bounds[0]:turn_bounds[1] + 1] * flip,
                                               lb_imu_data_z[turn_bounds[1] + 1:]])

                stacked_data = np.stack((lb_imu_data_x,lb_imu_data_y,lb_imu_data_z), axis=1)
                output_path = Path("Processed_Data") / f"{trial.name}.npy"
                output_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(output_path, stacked_data)
                rows.append({"Patient": patient,
                             "Trial_Name": trial.name,
                             "Current_VGA": curr_vga,
                             "Unstable_Gait": 1 if curr_vga >= 2 else 0, # Trial level
                             "Trial_Number": trial_idx,
                             "Condition": condition.name,
                             "Gait_Start": start,
                             "Gait_End": end,
                             "UTurn_Start": turn_bounds[0],
                             "UTurn_End": turn_bounds[1],
                             "Data_Path": output_path
                             })
                trial_idx += 1

            # add stuff here
df = pd.DataFrame(rows)
df.to_csv("dataset.csv")

print("success")
