import matplotlib.pyplot as plt
import numpy as np

with open('right_cop_y_dataset.csv') as file:
    lines = file.readlines()
    curves = [list(map(float, lines[i].split(","))) for i in range(1, len(lines))]
    mean_curve = np.mean(curves, axis=0)
    plt.plot(mean_curve, color='black', linewidth=2, label='Mean')
    plt.legend()
    plt.show()