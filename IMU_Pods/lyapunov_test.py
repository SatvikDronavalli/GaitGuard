import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.signal import periodogram
from scipy.spatial.distance import cdist
from sklearn.linear_model import LinearRegression
from TUG_data_visualization import array

# Rosenstein lyapunov approximation implementation, adapted from the Logical Errors blog (https://logicalerrors.wordpress.com/2020/02/22/estimating-the-first-lyapunov-exponent-in-r-using-rosensteins-method/)
def logistic(x, mu):
    return mu * x * (1 - x)

#TODO: Later experiment with multiple channels for higher accuracy (try calculating angular velocity magnitudes for now?)
#Dataset: https://springernature.figshare.com/articles/dataset/A_Dataset_of_Clinical_Gait_Signals_with_Wearable_Sensors_from_Healthy_Neurological_and_Orthopedic_Cohorts/28806086?file=53704514

#TODO: Add phase separation

x_obs = array # generate_trajectory(mu=4.0, x0=0.1, length=500)
J = 1 # TODO: read Rosenstein paper to determine empirical derivation methods for J (autocorrection function, FFT)
m = 2 # TODO: Experimentally test for m
M = len(x_obs) - (m - 1) * J

X = np.zeros((M, m))

#Delay Embedding

for i in range(M):
    idx = np.arange(i, i + m*J, J)
    X[i, :] = x_obs[idx]
#TODO: Determine a better way to find theiler

'''
Initially look at heuristic threshold from https://www.sciencedirect.com/science/article/pii/S2352340923007382?ref=pdf_download&fr=RR-2&rr=9b65ee25da5c2720

Future extension could be Short Time Fourier Transform method from: https://www.scitepress.org/PublishedPapers/2022/107753/107753.pdf
'''



def mean_period(ts):
    freqs, psd = periodogram(ts)
    w = psd / psd.sum()
    mean_freq = np.sum(freqs * w)
    return 1 / mean_freq if mean_freq > 0 else 1


def get_nearest_neighbors(X, theiler):
    D = cdist(X, X)
    np.fill_diagonal(D, np.inf)

    for i in range(len(X)):
        D[i, max(0, i-theiler):i+theiler+1] = np.inf

    return np.argmin(D, axis=1)

theiler = int(mean_period(x_obs))
neighbors = get_nearest_neighbors(X, theiler)
def mean_log_divergence(X, neighbors, t_end=25):
    M = len(X)
    mean_log = []

    for k in range(t_end):
        d_k = []
        for i in range(M - k):
            j = neighbors[i]
            if j + k < M:
                d = np.linalg.norm(X[i+k] - X[j+k])
                if d > 0:
                    d_k.append(np.log(d))
        mean_log.append(np.mean(d_k))
    return np.array(mean_log)

t_end = 25
mean_log_dist = mean_log_divergence(X, neighbors, t_end)
time = np.arange(t_end)

plt.plot(time, mean_log_dist)
plt.xlabel("Time")
plt.ylabel("Mean log distance")
plt.show()

# Determines "knee" of time vs mean log distance graph to maximize the length of line of best fit while keeping it linear

def kneedle_alg(x, y, direction="increasing", curve="concave", smooth_window=0):
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    n = x.size
    x_min, x_max = x[0], x[-1]
    x_n = (x - x_min) / (x_max - x_min)
    y_min, y_max = np.min(y), np.max(y)
    y_n = (y - y_min) / (y_max - y_min)
    # Compute deviation from chord (line from endpoints).
    # In normalized space, chord is y = x (because endpoints map to (0,0) and (1,1))
    # Since all points on the line satisfy y-x=0, y-x > 0 are points above the line (deviation)

    scores = y_n - x_n
    # Avoid picking endpoints
    inner = np.arange(1, n - 1)

    knee_sorted_idx = inner[np.argmax(scores[inner])]
    return int(knee_sorted_idx)

optimal_time = kneedle_alg(time,mean_log_dist)
reg = LinearRegression().fit(time[:optimal_time].reshape(-1,1), mean_log_dist[:optimal_time])
lyapunov = reg.coef_[0]
# "Lyapunov" during turning: 0.246

print("Estimated Lyapunov exponent:", lyapunov)
