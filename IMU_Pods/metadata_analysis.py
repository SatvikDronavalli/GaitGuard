from pathlib import Path
from TUG_data_visualization import dataset_preprocessing, butter_lowpass
import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt

DATASET_LOCATION = "C:/Users/Satvik/Downloads" # Change this as necessary
healthy_dir = Path(f"{DATASET_LOCATION}/dataset/dataset/data/healthy")
neuro_dir = Path(f"{DATASET_LOCATION}/dataset/dataset/data/neuro")
ortho_dir = Path(f"{DATASET_LOCATION}/dataset/dataset/data/ortho")
dataset_dir = Path(f"{DATASET_LOCATION}/dataset/dataset/data")
dataset = pd.DataFrame()

# Extracting Visual Gait Assessment (VGA) results

rows = []
laterality = {}
protocol = {}

for t in dataset_dir.iterdir():
    for condition in t.iterdir():
        for item in condition.iterdir():
            patient = item.name
            metadata = list(list(item.iterdir())[0].iterdir())[0]
            imu_data = dataset_preprocessing(list(list(item.iterdir())[0].iterdir())[2])
            lb_imu_data_x = butter_lowpass(imu_data["LB_Gyr_X"], 100)
            turn = None
            with open(metadata, 'r') as file:
                data = json.load(file)
                if data["visualGaitAssessment"] == "Not evaluated":
                    break
                turn = data["laterality"]
                gait_instability = 1 if float(data["visualGaitAssessment"]) >= 2 else 0
                if data["laterality"] not in laterality.keys():
                    laterality[data["laterality"]] = [1, gait_instability]
                else:
                    laterality[data["laterality"]][0] += 1
                    laterality[data["laterality"]][1] += gait_instability
                if data["protocol"] not in protocol.keys():
                    protocol[data["protocol"]] = [1, gait_instability]
                else:
                    protocol[data["protocol"]][0] += 1
                    protocol[data["protocol"]][1] += gait_instability
            if turn == 'left': #TODO: Measure and count directions based on lower back yaw (gyr_x) peak direction
                plt.plot(lb_imu_data_x)
                plt.show()
            # add stuff here

print(laterality)
print(protocol)
