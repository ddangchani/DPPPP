# Neyman-Scott synthesizer

import numpy as np
import scipy.stats as stats
from scipy.stats import norm, poisson
from scipy.optimize import bisect
import matplotlib.pyplot as plt
from src.eval import pMSE_int

# 2D Gaussian

def Scott(data):
    n, d = data.shape
    return n**(-1/(d+4)) * np.max(np.std(data, axis=0))

def edge_correction_2D(points, h, window):
    xmin, xmax, ymin, ymax = window
    n = len(points)
    # rv1 = norm(loc=points[0], scale=h)
    # rv2 = norm(loc=points[1], scale=h)
    # return (rv1.cdf(xmax) - rv1.cdf(xmin)) * (rv2.cdf(ymax) - rv2.cdf(ymin))
    xmins = np.full_like(n, xmin)
    xmaxs = np.full_like(n, xmax)
    ymins = np.full_like(n, ymin)
    ymaxs = np.full_like(n, ymax)
    xmins = (xmins - points[:, 0]) / h
    xmaxs = (xmaxs - points[:, 0]) / h
    ymins = (ymins - points[:, 1]) / h
    ymaxs = (ymaxs - points[:, 1]) / h
    return (norm.cdf(xmaxs) - norm.cdf(xmins)) * (norm.cdf(ymaxs) - norm.cdf(ymins))

def max_ratio_edge_correction_2D(h, alpha, window):
    xmin, xmax, ymin, ymax = window
    p1 = np.array([xmin, ymin]).reshape(1, -1)
    p2 = np.array([xmin + alpha / np.sqrt(2), ymin + alpha / np.sqrt(2)]).reshape(1, -1)
    return np.log(edge_correction_2D(p2, h, window)) - np.log(edge_correction_2D(p1, h, window))

def Gaussiankernel_2D(t, x, h):
    t = t[:, np.newaxis, :]  # Shape (N^2, 1, 2)
    x = x[np.newaxis, :, :]  # Shape (1, n, 2)
    return np.exp(-0.5 * np.sum((x - t)**2, axis=2) / h**2) / (2 * np.pi * h**2)

def intensity_estimate_point_2D(point, data, h, window, corrections=True, kernel="Gaussian"):
    if kernel == "Gaussian":
        kernel = Gaussiankernel_2D
    else:
        raise ValueError("Kernel not implemented")
    if corrections:
        edge_correction_values = edge_correction_2D(data, h, window)
    else:
        edge_correction_values = np.ones_like(data)

    kernel_values = kernel(point, data, h) / edge_correction_values
    return kernel_values.sum(axis=1)

def estimate_intensity_2D(data, h, window, N, kernel="Gaussian"):
    xmin, xmax, ymin, ymax = window
    xx = np.linspace(xmin, xmax, N)
    yy = np.linspace(ymin, ymax, N)
    X, Y = np.meshgrid(xx, yy)
    grids = np.array(list(zip(X.ravel(), Y.ravel())))

    intensity_values = intensity_estimate_point_2D(grids, data, h, window, corrections=True, kernel=kernel)
    return intensity_values.reshape(N, N)

# 1D Gaussian
def edge_correction_1D(points, h, window):
    xmin, xmax = window
    mins = np.full_like(points, xmin)
    maxs = np.full_like(points, xmax)
    mins = (mins - points) / h
    maxs = (maxs - points) / h
    return norm.cdf(maxs) - norm.cdf(mins)

def max_ratio_edge_correction_1D(h, alpha, window):
    xmin, xmax = window
    p1 = xmin
    p2 = xmin + alpha / np.sqrt(2)
    return np.log(edge_correction_1D(p2, h, window)) - np.log(edge_correction_1D(p1, h, window))

def Gaussiankernel_1D(t, x, h):
    return np.exp(-0.5 * ((x - t) / h)**2) / (np.sqrt(2 * np.pi) * h)

def intensity_estimate_point_1D(point, data, h, window, corrections=True, kernel="Gaussian"):
    if kernel == "Gaussian":
        kernel = Gaussiankernel_1D
    else:
        raise ValueError("Kernel not implemented")
    if corrections:
        edge_correction_values = edge_correction_1D(data, h, window)
    else:
        edge_correction_values = np.ones_like(data)

    kernel_values = kernel(point.reshape(-1, 1), data, h) / edge_correction_values
    return kernel_values.sum(axis=1) 

def estimate_intensity_1D(data, h, window, N, kernel="Gaussian"):
    xmin, xmax = window
    xx = np.linspace(xmin, xmax, N)
    intensity_values = intensity_estimate_point_1D(xx, data, h, window, kernel=kernel)
    return intensity_values.reshape(-1, 1)

# DP condition
def find_bw(epsilon, n, window, alpha=None, delta=None, dim=2):
    if alpha is None:
        alpha = 1 / n
    if delta is None:
        delta = 1 / n

    if dim == 2:
        xmin, xmax, ymin, ymax = window
        B = np.sqrt((xmax-xmin)**2 + (ymax-ymin)**2)
    elif dim == 1:
        B = window[1] - window[0]

    k = poisson.ppf(mu=n, q=1-delta)
    
    # Define the objective function to find the root
    def objective_1d(h):
        cond = (2 * alpha * B + alpha**2) / (2 * h**2) + max_ratio_edge_correction_1D(h, alpha, window)
        return cond - epsilon / k

    def objective_2d(h):
        cond = (2 * alpha * B + alpha**2) / (2 * h**2) + max_ratio_edge_correction_2D(h, alpha, window)
        return cond - epsilon / k

    # Define the bounds for h
    hmin = 1e-6
    hmax = 10 * B
    
    # Use scipy.optimize.bisect to find the root of the objective function
    try:
        if dim == 1:
            h_root = bisect(objective_1d, hmin, hmax)
        elif dim == 2:
            h_root = bisect(objective_2d, hmin, hmax)
        return h_root
    except ValueError as e:
        return None

def generate_from_estimate_1d_ns(estimate, window, N, seed=None):
    # N : resolution
    xmin, xmax = window
    if seed is not None:
        np.random.seed(seed)
    max_intensity = np.max(estimate)
    npoints = np.random.poisson(max_intensity * (xmax - xmin))

    # generate points
    xx = np.random.uniform(xmin, xmax, npoints).reshape(-1, 1)

    # thinning
    x_idx = ((xx - xmin) / (xmax - xmin) * N).astype(int)
    intensities = estimate[x_idx]
    probs = intensities / max_intensity

    keep = np.random.uniform(size=npoints) < probs.flatten()

    return xx[keep], intensities[keep]
    

def generate_from_estimate_2d_ns(estimate, window, N, seed=None):
    xmin, xmax, ymin, ymax = window
    # N : resolution
    if seed is None:
        seed = 0
    np.random.seed(seed)
    max_intensity = np.max(estimate)
    npoints = np.random.poisson(max_intensity * (xmax - xmin) * (ymax - ymin))

    # generate points
    xx = np.random.uniform(xmin, xmax, npoints)
    yy = np.random.uniform(ymin, ymax, npoints)
    new_points = np.array(list(zip(xx, yy)))
    
    # thinning
    x_idx = ((xx - xmin) / (xmax - xmin) * N).astype(int)
    y_idx = ((yy - ymin) / (ymax - ymin) * N).astype(int)
    intensities = estimate[x_idx, y_idx]
    probs = intensities / max_intensity

    keep = np.random.uniform(size=npoints) < probs.flatten()

    return new_points[keep], intensities[keep]


# RUN NS
def run_ns(my_fn, data, window, seed, nx, ny, eps, N=100):
    np.random.seed(seed)
    n = data.shape[0]

    # NS Synthesis
    h = find_bw(epsilon=eps, n=n, dim=2, window=window, alpha=1/nx, delta=1/n)
    rot_h = Scott(data)
    h = max(h, rot_h)
    estimates_NS = estimate_intensity_2D(data, h, window, N)

    # Synthesize from the estimate
    newpts_list = []
    pMSE_list = []
    intensity_list = []
    for i in range(10):
        newpts, intensities = generate_from_estimate_2d_ns(estimates_NS, window, N, seed=i)
        syn_fn = lambda x: intensity_estimate_point_2D(x, data, h, window)

        # pMSE
        pMSE = pMSE_int(syn=newpts, ori=data, syn_fn=syn_fn, ori_fn=my_fn, window=window, mu_syn=n)

        newpts_list.append(newpts)
        pMSE_list.append(pMSE)
        intensity_list.append(intensities)

    return pMSE_list, data, newpts_list, intensity_list
