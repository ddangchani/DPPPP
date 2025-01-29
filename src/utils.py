# Parametric classes of completely monotone and nonconstant function
import numpy as np
from shapely.geometry import Point, Polygon
import geopandas as gpd
from scipy.special import kv, gamma, loggamma
from scipy.optimize import fmin, minimize
from scipy.integrate import quad
import pytensor.tensor as pt
import pandas as pd
import matplotlib.pyplot as plt

def expi_approx(x, eps = 1e-7):
    x = pt.switch(pt.eq(x, 0), eps, x)  # Replace 0 with epsilon
    
    return pt.switch(
        pt.lt(x, 0),
        -pt.exp(x) * pt.gammaincc(0, -x),
        pt.exp(x) * pt.gammaincc(0, x)
    )

def __expi_approx(x, eps = 1e-7):
    x = np.where(x == 0, eps, x)  # Replace 0 with epsilon
    
    return np.where(
        x < 0,
        -np.exp(x) * gamma(0) - np.exp(x) * loggamma(0) + np.exp(x) * np.exp(x) * np.exp(-x) * (1 - np.exp(-x)),
        np.exp(x) * gamma(0) + np.exp(x) * loggamma(0) - np.exp(x) * np.exp(-x) * (1 - np.exp(-x))
    )

def powerexponential(t, bw, alpha=1.):
    """
    Power exponential function
    Args:
        t (np.array) : Input array
        bw (float) : Lengthscale parameter (lengthscale > 0)
        alpha (float) : Shape parameter (0 < alpha <= 1), default 1
    Returns:
        np.array : Output array
    """
    # parameter range
    # assert (0 < alpha) and (alpha <= 1), "alpha must be in (0, 1]"
    # assert bw > 0, "bw must be positive"

    return np.exp(-t**alpha / bw)

def matern(t, bw, alpha=0.5):
    """
    Matern function
    Args:
        t (np.array) : Input array
        bw (float) : Lengthscale parameter (lengthscale > 0)
        alpha (float) : Shape parameter (0 < alpha <= 1), default 0.5
    Returns:
        np.array : Output array
    """
    # parameter range
    # assert (alpha > 0) and (alpha <= 1/2), "alpha must be in (0, 1/2]"
    # assert bw > 0, "bw must be positive"

    return np.where(t != 0, (2**(1-alpha) / np.math.gamma(alpha)) * (t / bw)**alpha * kv(alpha, t / bw), 1)

def generalized_cauchy(t, alpha, beta, xi):
    """
    Generalized Cauchy function
    Args:
        t (np.array) : Input array
        alpha (float) : parameter
        beta (float) : parameter
        xi (float) : parameter
    Returns:
        np.array : Output array
    """
    # parameter range
    assert (alpha > 0) and (alpha <= 1), "alpha must be in (0, 1]"
    assert beta > 0, "beta must be positive"
    assert xi > 0, "xi must be positive"

    return np.pow((beta * np.pow(t, alpha) + 1), -xi/alpha) if t != 0 else 1

def dagum(t, alpha, beta, xi):
    """
    Dagum function
    Args:
        t (np.array) : Input array
        alpha (float) : parameter
        beta (float) : parameter
        xi (float) : parameter
    Returns:
        np.array : Output array
    """
    # parameter range
    assert (alpha > 0) and (alpha <= 1), "alpha must be in (0, 1]"
    assert beta > 0, "beta must be positive"
    assert (xi > 0) and (xi <= 1), "xi must be in (0, 1]"

    return 1 - np.pow((np.div(beta * np.pow(t, alpha), 1 + beta*np.pow(t, alpha))), xi/alpha) if t != 0 else 1


def generate_poisson_process_1d(intensity_fn, window=None, seed=0):
    """
    Simulate a Poisson process on the real line
    Args:
        intensity_fn (function) : intensity function
        seed (int) : random seed
        window (np.array) : study area (default [0, 1])
    Returns:
        np.array : points
    """
    np.random.seed(seed)
    if window is None:
        window = np.array([0, 1])

    area_size = window[1] - window[0]
    # find the maximum intensity in the study area
    max_intensity = minimize(lambda x: -intensity_fn(x), x0=np.array([0.5]), bounds=[(0, 1)]).fun
    max_intensity = -max_intensity
    n = np.random.poisson(max_intensity * area_size)

    # generate points (homogeneous Poisson process)
    xs = np.random.uniform(window[0], window[1], n)
    points = xs

    # thinning
    probs = intensity_fn(points) / max_intensity
    keep = np.random.uniform(size=n) < probs
    points = points[keep]

    return points

def generate_poisson_process_2d(intensity_fn, window=None, seed=0):
    """
    Simulate a Poisson process on the R^2 plane
    Args:
        intensity_fn (function) : intensity function
        seed (int) : random seed
        window (np.array) : study area (default [0, 1] x [0, 1])
    Returns:
        np.array : points
    """
    np.random.seed(seed)
    if window is None:
        xmin, xmax, ymin, ymax = 0, 1, 0, 1
    else:
        xmin, xmax, ymin, ymax = window

    area_size = (xmax - xmin) * (ymax - ymin)
    # find the maximum intensity in the study area
    max_intensity = minimize(lambda x: -intensity_fn(x[0], x[1]), x0=np.array([0.5, 0.5]), bounds=[(xmin, xmax), (ymin, ymax)]).fun
    max_intensity = -max_intensity
    n = np.random.poisson(max_intensity * area_size)

    # generate points (homogeneous Poisson process)
    xs = np.random.uniform(xmin, xmax, n)
    ys = np.random.uniform(ymin, ymax, n)
    points = np.array(list(zip(xs, ys)))

    # thinning
    probs = intensity_fn(points[:, 0], points[:, 1]) / max_intensity
    keep = np.random.uniform(size=n) < probs
    points = points[keep]

    return points

def generate_matern_hardcore(r, mean, window, type=1, seed=None):
    """
    Matérn hard-core point process (Type 1 and 2)
    r: hardcore distance
    mean: intensity of the underlying homogeneous Poisson process
    window: tuple with the limits of the window
    type: 1 or 2
    
    Source:
    https://github.com/hpaulkeeler/posts/blob/master/MaternHardcoreRectangle/MaternHardcoreRectangle.py
    """
    if seed is not None:
        np.random.seed(seed)  # Random seed

    xmin, xmax, ymin, ymax = window
    # Catch edge effects
    xminExt = xmin - r
    xmaxExt = xmax + r
    yminExt = ymin - r
    ymaxExt = ymax + r

    # 1. Simulate a homogeneous Poisson process
    n = np.random.poisson((xmaxExt - xminExt) * (ymaxExt - yminExt) * mean)
    xs = np.random.uniform(xminExt, xmaxExt, n)
    ys = np.random.uniform(yminExt, ymaxExt, n)

    # 2. Thinning points outside the window
    inside = (xs >= xmin) & (xs <= xmax) & (ys >= ymin) & (ys <= ymax)
    xs = xs[inside]
    ys = ys[inside]
    pts = np.column_stack((xs, ys))

    # 3. Thinning using vectorized operations
    dists = np.linalg.norm(pts[:, None] - pts, axis=2)
    np.fill_diagonal(dists, np.inf)  # Ignore self-distances

    if type == 1:
        keep = np.all(dists >= r, axis=1)
    elif type == 2:
        marks = np.random.rand(len(xs))
        keep = np.ones(len(xs), dtype=bool)
        for i in range(len(xs)):
            if keep[i]:
                neighbors = dists[i, :] < r
                if np.any(neighbors):
                    keep[neighbors] = marks[i] < marks[neighbors]

    pts = pts[keep]
    return pts

def mincontrast(points, window, rmin, rmax, bounds, ratio=None, dim=2, q=1/4, p=2):
    """
    Minimum contrast estimator for the covariance function
    Args:
        points (np.array) : point data
        window (tuple) : window boundaries
        rmin (float) : minimum support value of radius r
        rmax (float) : maximum support value of radius r
        bounds (list) : bounds for the optimization (pass to scipy.optimize.minimize)
        dim (int) : dimension (1 or 2)
        q (float) : q parameter (default 1/4)
        p (int) : p parameter (default 2)
    Returns:
        float : sigma
        float : ratio
    """
    n = points.shape[0]
    if dim == 1:
        B = window[1] - window[0]
        dist = np.abs(points[:, None] - points)
        corrections = np.ones((n, n))
    elif dim == 2:
        B = (window[1] - window[0]) * (window[3] - window[2])
        dist = np.linalg.norm(points[:, None] - points, axis=2)
        corrections = correction_matrix_(points, dist, window)
    else:
        raise ValueError("dim must be 1 or 2")
    
    dist_sorted, corrections_sorted = K_est_vals(points, dist, corrections, dim)

    # Optimize the objective function
    if not ratio:
        obj = lambda x: mce_obj(x[0], x[1], rmin, rmax, points, B, dist_sorted, corrections_sorted, q, p)
        res = minimize(obj, x0=np.array([1, 1]), bounds=bounds, method='L-BFGS-B')
        sigma, ratio = res.x
        return sigma, ratio
    else:
        obj = lambda x: mce_obj(x, ratio, rmin, rmax, points, B, dist_sorted, corrections_sorted, q, p)
        res = minimize(obj, x0=np.array([1]), bounds=bounds, method='L-BFGS-B')
        sigma = res.x
        return sigma, ratio
    

def mce_obj(sigma, ratio, rmin, rmax, data, B, dist_sorted, corrections_sorted, q=1/4, p=2):
    ls = sigma / ratio
    obj = quad(lambda r: np.abs(K_gauss(r, sigma, ls) ** q - K_est_r(r, data, B, dist_sorted, corrections_sorted) ** q) ** p, rmin, rmax)
    return obj[0]

def K_gauss(r, sigma=1, ls=1):
    return quad(lambda s: 2*np.pi*s*np.exp(sigma**2 * np.exp(- s**2 / (2*ls**2))), 0, r)[0]

def K_exp(r, sigma=1, ls=1):
    return quad(lambda s: 2*np.pi*s*np.exp(sigma**2 * np.exp(-s/ls)), 0, r)[0]

def Ripley_correction(point, d, window):
    # fraction of length of the circle lying within the window
    xl, xu, yl, yu = window
    length = 2*np.pi*d
    circle = Point(point).buffer(d)
    window_ = Polygon([(xl, yl), (xl, yu), (xu, yu), (xu, yl)])
    intersection = circle.intersection(window_)

    return intersection.boundary.length/length

def correction_matrix_(points, dist, window):
    n = points.shape[0]
    corrections = np.ones((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                corrections[i, j] = Ripley_correction(points[i], dist[i, j], window)
    return corrections

def K_est_vals(points, dist, corrections, dim):
    if dim == 1:
        dist = np.abs(points[:, None] - points)
    elif dim == 2:
        dist = np.linalg.norm(points[:, None] - points, axis=2)
    
    np.fill_diagonal(dist, np.inf)

    dist = dist.flatten()
    corrections = corrections.flatten()

    idx = np.argsort(dist)
    dist = dist[idx]
    corrections = corrections[idx]

    corrections = corrections[dist != np.inf]
    dist = dist[dist != np.inf]
    
    return dist, corrections

def K_est_r(r, points, B, dist_sorted, corrections_sorted):
    n = points.shape[0]
    lambdahat = n / B
    K_r = np.sum((dist_sorted <= r) / corrections_sorted)
    K_r = K_r / lambdahat / n
    return K_r

def K_est(points, window, r_values, dim=2, correction=True):
    """
    K function estimation for given support points
    
    Args:
        points (np.array): point data
        window (tuple): window boundaries
        r_values (np.array): support values
        dim (int): dimension (1 or 2)
        correction (bool): Ripley's edge correction (default True)
    Returns:
        np.array: Ripley's K function values
    """
    n = points.shape[0]
    
    if dim == 1:
        B = window[1] - window[0]
        dist = np.abs(points[:, None] - points)
    elif dim == 2:
        B = (window[1] - window[0]) * (window[3] - window[2])
        dist = np.linalg.norm(points[:, None] - points, axis=2)
    else:
        raise ValueError("dim must be 1 or 2")
    
    np.fill_diagonal(dist, np.inf)

    lambdahat = n / B
    if correction and dim == 2:
        correction_factors = correction_matrix_(points, dist, window)
    else:
        correction_factors = np.ones((n, n))

   #  Vectorized K calculation
    K_r = np.sum((dist[:, :, None] <= r_values) / correction_factors[:, :, None], axis=(0, 1))
    K_r = K_r / lambdahat / n

    return K_r

def K_est_inhom(points, intensities, window, r_values, correction=True):
    """
    K function estimation for given support points
    
    Args:
        points (np.array): point data
        intensities (np.array): intensities at points
        r_values (np.array): support values
        correction (bool): Ripley's edge correction (default True)
    Returns:
        np.array: Ripley's K function values
    """
    n = points.shape[0]
    dist = np.linalg.norm(points[:, None] - points, axis=2)
    int_outer = np.outer(intensities, intensities)
    B = (window[1] - window[0]) * (window[3] - window[2])
    
    np.fill_diagonal(dist, np.inf)

    if correction:
        correction_factors = correction_matrix_(points, dist, window) * int_outer
    else:
        correction_factors = np.ones((n, n)) * int_outer

   #  Vectorized K calculation
    K_r = np.sum((dist[:, :, None] <= r_values) / correction_factors[:, :, None], axis=(0, 1))
    K_r = K_r / B

    return K_r

def generate_from_estimate_1D(estimate, window, N, seed=None):
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
    

def generate_from_estimate_2D(estimate, window, N, seed=None):
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

def plot_K_band(intensity_fn, window, r_values, B=30, correction=True, ax=None, band=True, return_mean=True, **kwargs):
    """
    Plot the K function and the confidence band for known intensity function
    Args:
        intensity_fn (function): intensity function
        r_values (np.array): support values
        B (int): number of simulations
    """
    ser = pd.DataFrame({'pts': [generate_poisson_process_2d(intensity_fn=intensity_fn, window=window, seed=i) for i in range(B)]})
    ser['intensity'] = ser.apply(lambda x: [intensity_fn(x1, x2) for x1, x2 in x.pts], axis=1)
    ser['K'] = ser.apply(lambda x: K_est_inhom(points=np.array(x.pts), intensities=np.array(x.intensity), window=window, r_values=r_values, correction=correction), axis=1)

    mean_K_true = np.stack(ser.K.values).mean(axis=0)
    upper_K_true = np.percentile(np.stack(ser.K.values), 97.5, axis=0)
    lower_K_true = np.percentile(np.stack(ser.K.values), 2.5, axis=0)

    if ax is None:
        fig, ax = plt.subplots()
    ax.plot(r_values, mean_K_true, **kwargs)
    if band:
        ax.fill_between(r_values, lower_K_true, upper_K_true, alpha=0.5, **kwargs)
    
    if return_mean:
        return mean_K_true

def scale_Gaussian_Noise(eps, delta, sens=1):
    return np.sqrt(2 * np.log(1.25 / delta)) / eps * sens


def add_Gaussian_Noise(data, window, scale_DP, n=11, seed=None):
    """
    Gaussian Noise Histogram synthesis
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Discretize the space
    x = np.linspace(window[0], window[1], n)
    y = np.linspace(window[2], window[3], n)

    # Count the number of points in each grid
    hist_ = np.histogram2d(data[:, 0], data[:, 1], bins=[x, y])[0]

    # Add Gaussian noise
    hist_ += np.random.normal(0, scale_DP, hist_.shape)

    # Generate new points > For each count in grid, generate uniformly distributed points
    newpts_DP = []
    for i in range(n-1):
        for j in range(n-1):
            _x = np.random.uniform(x[i], x[i+1], max(0, int(hist_[i, j])))
            _y = np.random.uniform(y[j], y[j+1], max(0, int(hist_[i, j])))
            newpts_DP.append(np.vstack([_x, _y]).T)

    newpts_DP = np.vstack(newpts_DP)
    return newpts_DP

def add_Laplace_noise(data, window, eps, n=11, seed=None, return_intensity=False):
    """
    Laplace Mechanism Histogram synthesis for the square window
    """
    if seed is not None:
        np.random.seed(seed)
    x = np.linspace(window[0], window[1], n)
    y = np.linspace(window[2], window[3], n)

    cell_size = (x[1] - x[0]) * (y[1] - y[0])
    sens = 2 / cell_size

    # Count
    hist_ = np.histogram2d(data[:, 0], data[:, 1], bins=[x, y])[0]
    int_ = hist_ / cell_size

    # Add noise
    # hist_ += np.random.laplace(0, 2/eps, hist_.shape)
    # hist_ = np.clip(hist_, 0, None)
    int_ += np.random.laplace(0, sens/eps, int_.shape)
    int_ = np.clip(int_, 0, None)
    mean_ = int_ * cell_size
    
    # Generate new points
    newpts_DP = []
    newpts_int = []
    # npoints_DP = np.random.poisson(lam=hist_)
    npoints_DP = np.random.poisson(lam=mean_)

    for i in range(n-1):
        for j in range(n-1):
            npt = npoints_DP[i, j]
            _x = np.random.uniform(x[i], x[i+1], npt)
            _y = np.random.uniform(y[j], y[j+1], npt)
            newpts_DP.append(np.vstack([_x, _y]).T)
            newpts_int.append(np.ones(npt) * int_[i, j])

    newpts_DP = np.vstack(newpts_DP)
    newpts_int = np.hstack(newpts_int)

    if return_intensity:
        return newpts_DP, newpts_int, int_

    else:
        return newpts_DP