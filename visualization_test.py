import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd

'''
with open('right_cop_y_dataset.csv') as file:
    lines = file.readlines()
    curves = [list(map(float, line.strip().split(",")[:-1])) for line in lines[1:] if int(float(line.strip().split(",")[-1])) == 0]
    curves2 = [list(map(float, line.strip().split(",")[:-1])) for line in lines[1:] if int(float(line.strip().split(",")[-1])) == 1]

    mean_curve = np.mean(curves, axis=0)
    mean_curve2 = np.mean(curves2, axis=0)

    plt.plot(mean_curve, color='black', linewidth=2, label='Label 0 Mean')
    plt.plot(mean_curve2, color='blue', linewidth=2, label='Label 1 Mean')
    plt.legend()
    plt.show() '''

grf_data = pd.read_csv("right_grf_dataset.csv").to_numpy()
y_vals = grf_data[:,-1]
grf_data = grf_data[:,:-1]
acc_x_data = pd.read_csv("right_acc_x_dataset.csv").to_numpy()
print(grf_data.shape,acc_x_data.shape)
acc_x_data = acc_x_data[:2381,:-1]
acc_y_data = pd.read_csv("right_acc_y_dataset.csv").to_numpy()
acc_y_data = acc_y_data[:2381,:-1]
acc_z_data = pd.read_csv("right_acc_z_dataset.csv").to_numpy()
acc_z_data = acc_z_data[:2381,:-1]
cop_x_data = pd.read_csv("right_cop_x_dataset.csv").to_numpy()
cop_x_data = cop_x_data[:2381,:-1]
cop_y_data = pd.read_csv("right_cop_y_dataset.csv").to_numpy()
cop_y_data = cop_y_data[:2381,:-1]
gyr_x_data = pd.read_csv("right_gyr_x_dataset.csv").to_numpy()
gyr_x_data = gyr_x_data[:2381,:-1]
gyr_y_data = pd.read_csv("right_gyr_y_dataset.csv").to_numpy()
gyr_y_data = gyr_y_data[:2381,:-1]
gyr_z_data = pd.read_csv("right_gyr_z_dataset.csv").to_numpy()
gyr_z_data = gyr_z_data[:2381,:-1]
new_x = np.stack([acc_x_data,acc_y_data,acc_z_data,cop_x_data,cop_y_data,gyr_x_data,gyr_y_data,gyr_z_data],-1)

X_train, X_test, y_train, y_test = train_test_split(
    new_x, y_vals, test_size=0.2, random_state=42, stratify=y_vals
)
print(len(X_train),len(X_test))
print(new_x.shape)