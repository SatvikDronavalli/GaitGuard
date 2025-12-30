import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.signal import periodogram
from scipy.spatial.distance import cdist
from sklearn.linear_model import LinearRegression

# Rosenstein lyapunov approximation implementation, adapted from the Logical Errors blog (https://logicalerrors.wordpress.com/2020/02/22/estimating-the-first-lyapunov-exponent-in-r-using-rosensteins-method/)
def logistic(x, mu):
    return mu * x * (1 - x)

#TODO: Use real gait data as the trajectory, later experiment with multiple channels for higher accuracy
def generate_trajectory(mu, x0=0.1, length=500):
    x = np.zeros(length)
    x[0] = x0
    for i in range(length - 1):
        x[i+1] = logistic(x[i], mu)
    return x

x_obs = generate_trajectory(mu=4.0, x0=0.1, length=500)
J = 1 # TODO: read Rosenstein paper to determine empirical derivation methods for J
m = 2 # TODO: Experimentally test for m
M = len(x_obs) - (m - 1) * J

X = np.zeros((M, m))
for i in range(M):
    idx = np.arange(i, i + m*J, J)
    X[i, :] = x_obs[idx]
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
reg = LinearRegression().fit(time[:10].reshape(-1,1), mean_log_dist[:10])
lyapunov = reg.coef_[0]

print("Estimated Lyapunov exponent:", lyapunov)
