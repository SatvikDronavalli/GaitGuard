import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.signal import resample

# Load the dataset
file_name = "../New_Data/HC101_SelfPace.csv"
patient = file_name.split("_")[0]
df = pd.read_csv(file_name)
plt.plot(df["R Foot Pressure"])
plt.show()
metadata = pd.read_csv("CONTROLS - Demographic+Clinical - datasetV1.csv")
# print(metadata[metadata.columns[8]])
weight = int(metadata.loc[metadata[metadata.columns[0]] == patient][metadata.columns[8]].tolist()[0])*9.81
# Remove 'Standing' periods
df = df[df["GeneralEvent"] != "Standing"].reset_index(drop=True)