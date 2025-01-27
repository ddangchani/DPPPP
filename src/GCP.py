import numpy as np
import matplotlib.pyplot as plt
import pymc as pm
import geopandas as gpd
from src.linnet import pairdistR
from src.utils import matern, expi_approx, powerexponential
from src.eval import pMSE_int
from src.triangulation import create_triangulation, proj, get_areas, get_dual, tessellation, get_dual_rec
import pytensor.tensor as pt
from pytensor.compile.ops import as_op
from scipy.special import gamma, expi
from shapely.geometry import Polygon
import datetime
import arviz as az
import pandas as pd

class Resistancecov_Matern(pm.gp.cov.Covariance):
    def __init__(self, sigma, linnet, bandwidth, alpha):
        super(Resistancecov_Matern, self).__init__(2, None)
        self.alpha = alpha
        self.sigma = sigma
        self.linnet = linnet
        self.bandwidth = bandwidth

    def diag(self, X):
        return np.ones(X.shape[0]) * self.sigma**2  
    
    def full(self, X, Xs=None):
        gdf = gpd.GeoSeries.from_xy(X[:, 0], X[:, 1])
        R = pairdistR(gdf, self.linnet)
        R = matern(t=R, bw=self.bandwidth, alpha=self.alpha)
        return R * self.sigma**2
    
class Resistancecov_exp(pm.gp.cov.Covariance):
    def __init__(self, sigma, linnet, ls, alpha=1.):
        super(Resistancecov_exp, self).__init__(2, None)
        self.alpha = alpha # shape parameter > (0, 1] for power exponential
        self.sigma = sigma # variance
        self.linnet = linnet
        self.ls = ls # lengthscale

    def diag(self, X):
        return np.ones(X.shape[0]) * self.sigma**2  
    
    def full(self, X, Xs=None):
        gdf = gpd.GeoSeries.from_xy(X[:, 0], X[:, 1])
        R = pairdistR(gdf, self.linnet)
        R = np.exp(-R ** self.alpha / self.ls)
        return R * self.sigma**2

# Likelihood function (1D) > Used Pytensor
def hat_basis_matrix_1d(observed, knots): # 1D hat basis Matrix (n x m)
    h = pt.diff(knots)[0]
    diffs = pt.abs(np.subtract.outer(observed, knots))
    basis = pt.maximum(0, 1 - diffs / h)
    return basis

def hat_basis_matrix_1d_(observed, knots): # numpy version
    h = np.diff(knots)[0]
    diffs = np.abs(np.subtract.outer(observed, knots))
    basis = np.maximum(0, 1 - diffs / h)
    return basis

def intensity_measure_hat(beta, knots, epsilon=1e-7):
    delta = pt.diff(knots)[0]
    beta_diff = pt.diff(beta)
    exp_beta_diff = pt.diff(np.exp(beta))
    return pt.sum(delta * exp_beta_diff / (beta_diff + epsilon))

def likelihood_hat_1d(beta, knots, observed):
    mat = hat_basis_matrix_1d(observed, knots)
    intensity = intensity_measure_hat(beta, knots)
    n = np.shape(observed)[0]
    # Poisson likelihood
    return pt.exp(-intensity) * pt.exp(pt.sum(mat @ beta))

def intensity_fn_1d_hat(x, betas, knots):
    basis = hat_basis_matrix_1d_(x, knots)
    return np.exp(basis @ betas)

def intensity_fn_1d_const(x, betas, knots):
    arr = np.digitize(x, knots, right=True) - 1
    return np.exp(betas[np.where(arr == -1, 0, arr)])

# Likelihood function (2D)

def hat_basis_matrix_2d_(observed, knots_x, knots_y): # numpy version
    n = observed.shape[0]
    mx, my = len(knots_x), len(knots_y)
    # Produce n x (mx * my) matrix
    basis_x = hat_basis_matrix_1d_(observed[:, 0], knots_x) # n x mx
    basis_y = hat_basis_matrix_1d_(observed[:, 1], knots_y) # n x my
    basis = np.einsum('ij,ik->ijk', basis_x, basis_y).reshape(n, mx * my)
    
    return basis

def hat_basis_matrix_2d(observed, knots_x, knots_y):
    n = observed.shape[0]
    mx, my = len(knots_x), len(knots_y)

    basis_x = hat_basis_matrix_1d(observed[:, 0], knots_x)  # n x mx
    basis_y = hat_basis_matrix_1d(observed[:, 1], knots_y)  # n x my

    basis_x = basis_x.dimshuffle(0, 1, 'x')  # n x mx x 1
    basis_y = basis_y.dimshuffle(0, 'x', 1)  # n x 1 x my
    
    basis = basis_x * basis_y  # n x mx x my
    basis = basis.reshape((n, mx * my))  # n x (mx * my)
    
    return basis

@as_op(itypes=[pt.dmatrix, pt.dmatrix, pt.dmatrix, pt.dmatrix], otypes=[pt.dmatrix])
def compute_cell_integral(beta_ij, beta_ip1_j, beta_i_jp1, beta_ip1_jp1):
    eps = 1e-7
    a = beta_ij - beta_ip1_j - beta_i_jp1 + beta_ip1_jp1 + eps
    b = -beta_ij + beta_ip1_j
    c = -beta_ij + beta_i_jp1
    d = beta_ij                         

    result = np.exp(d) * (1/a) * np.exp(-b*c/a) * (
            expi((a+b)*(1 + c/a)) - expi(b*(1 + c/a)) -
            expi(c*(a + b)/a) + expi(b*c/a)
        )
    
    return result

def intensity_measure_hat_2d(beta, mx, hx, my=None, hy=None, epsilon=1e-7):
    if my is None:
        my = mx
    if hy is None:
        hy = hx
    
    beta = beta.reshape((mx, my))

    beta_ij = beta[:-1, :-1]
    beta_ip1j = beta[1:, :-1]
    beta_ijp1 = beta[:-1, 1:]
    beta_ip1jp1 = beta[1:, 1:]

    cell_integrals = compute_cell_integral(beta_ij, beta_ip1j, beta_ijp1, beta_ip1jp1)
    mu_S = pt.sum(cell_integrals) * hx * hy

    return mu_S

def intensity_measure_hat_2d_mc(beta, knots_x, knots_y, n_samples=1000):
    # Monte Carlo approximation
    xmin, xmax = knots_x[0], knots_x[-1]
    ymin, ymax = knots_y[0], knots_y[-1]
    x = np.random.uniform(xmin, xmax, n_samples)
    y = np.random.uniform(ymin, ymax, n_samples)
    pts = np.column_stack([x, y])

    basis = hat_basis_matrix_2d(pts, knots_x, knots_y)
    msr_hat = pt.exp(basis.dot(beta))
    return msr_hat.mean()


def likelihood_hat_2d(beta, knots_x, knots_y, observed, approx=False):
    mat = hat_basis_matrix_2d(observed, knots_x, knots_y)
    if approx:
        intensity = intensity_measure_hat_2d_mc(beta, knots_x, knots_y)
    else:
        intensity = intensity_measure_hat_2d(beta, len(knots_x), np.diff(knots_x)[0], len(knots_y), np.diff(knots_y)[0])
    return pt.exp(-intensity) * pt.exp(pt.sum(mat @ beta))

def intensity_fn_2d_hat(x, betas, knots_x, knots_y):
    basis = hat_basis_matrix_2d_(x, knots_x, knots_y)
    return np.exp(basis @ betas)

def intensity_fn_2d_const(x, betas, knots_x, knots_y):
    betas = betas.reshape((len(knots_x)-1, len(knots_y)-1))
    arr_x = np.digitize(x[:,0], knots_x, right=True) - 1
    arr_y = np.digitize(x[:,1], knots_y, right=True) - 1
    return np.exp(betas[arr_x, arr_y])

# DP condition (p=2)
def rmax(epsilon, delta, N, B=1, dim=1, basis='const'):
    if dim == 1:
        if basis == 'const':
            return np.sqrt(epsilon**2 * delta * N / 4) / B
        elif basis == 'hat':
            return np.sqrt(epsilon**2 * delta * N / 16) / B
    elif dim == 2:
        if basis == 'const':
            return np.sqrt(epsilon**2 * delta / 24) / B
        elif basis == 'hat':
            return np.sqrt(epsilon**2 * delta / 104) / B


# Synthesis
def generate_from_estimate_1d_const(intensity_sample, knots):
    h = np.diff(knots)[0]
    n_points = [np.random.poisson(intensity_sample[i] * h) for i in range(len(knots)-1)]
    points = np.array([])
    for i, n in enumerate(n_points):
        points = np.append(points, np.random.uniform(knots[i], knots[i+1], n))
    return points

def generate_from_estimate_1d_hat(beta_sample, knots): # Thinning
    h = np.diff(knots)[0]
    B = knots[-1] - knots[0]
    max_intensity = np.max(intensity_fn_1d_hat(knots, beta_sample, knots))
    n_points = np.random.poisson(max_intensity * B)
    xx = np.random.uniform(knots[0], knots[-1], n_points)
    probs = intensity_fn_1d_hat(xx, beta_sample, knots) / max_intensity
    return xx[np.random.uniform(size=n_points) < probs]

def generate_from_estimate_2d_const(intensity_sample, knots_x, knots_y):
    area = np.diff(knots_x)[0] * np.diff(knots_y)[0]
    grids = gpd.GeoDataFrame(geometry=[Polygon([(x, y), (x, y+1), (x+1, y+1), (x+1, y)]) for x in knots_x[:-1] for y in knots_y[:-1]])
    grids['intensity'] = intensity_sample
    npoints = grids['intensity'].apply(lambda x: np.random.poisson(x * area))
    samples = grids.sample_points(npoints).explode().reset_index(drop=True).geometry.get_coordinates().to_numpy()
    return samples

def generate_from_estimate_2d_hat(beta_sample, knots_x, knots_y): # Thinning
    B = (knots_x[-1] - knots_x[0]) * (knots_y[-1] - knots_y[0])
    max_intensity = np.max(np.exp(beta_sample))
    n_points = np.random.poisson(max_intensity * B)
    xx = np.random.uniform(knots_x[0], knots_x[-1], n_points)
    yy = np.random.uniform(knots_y[0], knots_y[-1], n_points)
    pts = np.column_stack([xx, yy])
    probs = intensity_fn_2d_hat(pts, beta_sample, knots_x, knots_y) / max_intensity
    keep = np.random.uniform(size=n_points) < probs
    return np.column_stack([xx[keep], yy[keep]])

# RUN GCP

def run_tri(my_fn, data, window, seed, nx, ny, eps, save=False, prior="invgamma", burnin=2000):
    """
    Run the simulation for the 2D spatial point process using the triangulation method.
    
    Parameters
    ----------
    my_fn : function
        The intensity function to be used. (True intensity function)
    data : numpy.ndarray
        The observed data points.
    window : list
        The window of the observed data points.
    seed : int
        The seed for the random number generator.
    nx : int
        The number of knots in the x-direction.
    ny : int
        The number of knots in the y-direction.
    eps : float
        The privacy parameter.
    save : bool
        Save the traces.
    prior : str
        The prior distribution for the lengthscale. Choose from ['invgamma', 'lognormal', 'halfcauchy'].
    burnin : int
        The number of burnin steps. (Default: 2000)
    """
    xmin, xmax, ymin, ymax = window

    # Triangulation
    # create a triangulation
    n = data.shape[0]
    triang = create_triangulation(nx, ny, window)
    N = triang.x.shape[0]
    P = proj(data, triang)
    alpha_0 = get_areas(get_dual(triang, window))
    y = np.hstack([np.zeros(N), np.ones(n)]).reshape(1, -1)
    alpha = np.hstack([alpha_0, np.zeros(n)]).reshape(1, -1)
    area = (xmax - xmin) * (ymax - ymin)
    lambda0 = np.log(n / area)

    # Delta and ratio
    delta = 1/n
    ratio = eps * np.sqrt(delta / 544) / (xmax - xmin)

    # pymc model
    with pm.Model() as model:
        # weights
        mean_func = pm.gp.mean.Constant(lambda0)
        if prior == "invgamma":
            ls = pm.InverseGamma('lengthscale', alpha=1, beta=1) # lengthscale
        elif prior == "lognormal":
            ls = pm.Lognormal('lengthscale', mu=0, sigma=1.)
        elif prior == "halfcauchy":
            ls = pm.HalfCauchy('lengthscale', beta=1.)
        else:
            raise ValueError("Invalid prior distribution for the lengthscale.")
        sigma = ratio * ls * np.sqrt(2) # noise
        cov_func = pm.gp.cov.ExpQuad(2, ls=ls) * sigma**2
        gp = pm.gp.Latent(mean_func=mean_func, cov_func=cov_func)
        weights = gp.prior('weights', X=np.vstack([triang.x, triang.y]).T)

        # likelihood
        logeta = pt.concatenate([weights, pt.dot(weights, P.T)]).reshape((1, -1))
        loglik = pm.Potential('loglik', y * logeta - alpha * pm.math.exp(logeta))
        
        # sample
        trace = pm.sample(1000, tune=burnin, random_seed=seed, cores=1, chains=1)

    if save:
        az.to_netcdf(trace, f'data/trace_tri_{my_fn.__name__}_{datetime.datetime.now().strftime("%Y%m%d%H%M%S")}.nc')

    grid_for_plot = np.meshgrid(np.linspace(xmin, xmax, 100), np.linspace(ymin, ymax, 100))
    grid_points_for_plot = np.vstack([grid_for_plot[0].ravel(), grid_for_plot[1].ravel()]).T
    P_plot = proj(grid_points_for_plot, triang)

   # Sample last 10 samples
    newpts_list = []
    pMSE_list = []
    intensity_list = []

    for i in range(10):
        sample = trace.posterior.weights.values[0][-i].reshape(-1, 1)
        result = np.exp(P_plot @ sample)
        maxintensity = np.max(result)

        # Thinning
        new_npts = np.random.poisson(maxintensity * area)
        xs = np.random.uniform(xmin, xmax, new_npts)
        ys = np.random.uniform(ymin, ymax, new_npts)
        newpts = np.vstack([xs, ys]).T
        us = np.random.uniform(0, maxintensity, new_npts)

        _P_new = proj(newpts, triang)
        _intensities = np.exp(_P_new @ sample).flatten()
        keep = us < _intensities
        newpts = newpts[keep]
        intensity_list.append(_intensities[keep])

        # pMSE
        syn_fn = lambda x: np.exp(proj(x, triang) @ sample)
        mu_syn = np.sum(alpha_0 * np.exp(sample.flatten()))
        pMSE = pMSE_int(syn=newpts, ori=data, syn_fn=syn_fn, ori_fn=my_fn, window=window, mu_syn=mu_syn)
        newpts_list.append(newpts)
        pMSE_list.append(pMSE)

    return pMSE_list, data, newpts_list, intensity_list



def run_sqr(my_fn, data, window, seed, nx, ny, eps, save=False, prior="invgamma", burnin=2000):
    """
    Run the simulation for the 2D spatial point process using the square grid method.

    Parameters
    ----------
    my_fn : function
        The intensity function to be used. (True intensity function)
    data : numpy.ndarray
        The observed data points.
    window : list
        The window of the observed data points.
    seed : int
        The seed for the random number generator.
    nx : int
        The number of knots in the x-direction.
    ny : int
        The number of knots in the y-direction.
    eps : float
        The privacy parameter.
    save : bool
        Save the traces.
    prior : str
        The prior distribution for the lengthscale. Choose from ['invgamma', 'lognormal', 'halfcauchy'].
    burnin : int
        The number of burnin steps. (Default: 2000)
    """
    xmin, xmax, ymin, ymax = window

    # Tessellation
    knot_x = np.linspace(xmin, xmax, nx)
    knot_y = np.linspace(ymin, ymax, ny)
    X, Y = np.meshgrid(knot_x, knot_y)
    grid_points = np.vstack([X.ravel(), Y.ravel()]).T

    grid_poly = tessellation(nx, ny, window)
    dual = get_dual_rec(window, nx, ny)

    n = data.shape[0]
    N = nx * ny
    P = hat_basis_matrix_2d_(data, knot_x, knot_y)
    alpha_0 = get_areas(dual)
    y = np.hstack([np.zeros(N), np.ones(n)]).reshape(1, -1)
    alpha = np.hstack([alpha_0, np.zeros(n)]).reshape(1, -1)
    area = (xmax - xmin) * (ymax - ymin)
    lambda0 = n / area

    # Delta and ratio
    delta = 1/n
    ratio = eps * np.sqrt(delta / 104) / (xmax - xmin)

    # pymc model
    with pm.Model() as model:
        # weights
        mean_func = pm.gp.mean.Constant(lambda0)
        if prior == "invgamma":
            ls = pm.InverseGamma('lengthscale', alpha=1, beta=1) # lengthscale
        elif prior == "lognormal":
            ls = pm.Lognormal('lengthscale', mu=0, sigma=1.)
        elif prior == "halfcauchy":
            ls = pm.HalfCauchy('lengthscale', beta=1.)
        else:
            raise ValueError("Invalid prior distribution for the lengthscale.")
        sigma = ratio * ls * np.sqrt(2) # noise
        cov_func = pm.gp.cov.ExpQuad(2, ls=ls) * sigma**2
        gp = pm.gp.Latent(mean_func=mean_func, cov_func=cov_func)
        weights = gp.prior('weights', X=grid_points)

        # likelihood
        logeta = pt.concatenate([weights, pt.dot(weights, P.T)]).reshape((1, -1))
        loglik = pm.Potential('loglik', y * logeta - alpha * pm.math.exp(logeta))
        
        # sample
        trace = pm.sample(1000, tune=burnin, random_seed=seed, cores=1, chains=1)

    if save:
        az.to_netcdf(trace, f'data/trace_tri_{my_fn.__name__}_{datetime.datetime.now().strftime("%Y%m%d%H%M%S")}.nc')

    grid_for_plot = np.meshgrid(np.linspace(xmin, xmax, 100), np.linspace(ymin, ymax, 100))
    grid_points_for_plot = np.vstack([grid_for_plot[0].ravel(), grid_for_plot[1].ravel()]).T
    P_plot = hat_basis_matrix_2d_(grid_points_for_plot, knot_x, knot_y)

   # Sample last 10 samples
    newpts_list = []
    pMSE_list = []
    intensity_list = []

    for i in range(10):
        sample = trace.posterior.weights.values[0][-i].reshape(-1, 1)
        result = np.exp(P_plot @ sample)
        maxintensity = np.max(result)

        # Thinning
        new_npts = np.random.poisson(maxintensity * area)
        xs = np.random.uniform(xmin, xmax, new_npts)
        ys = np.random.uniform(ymin, ymax, new_npts)
        newpts = np.vstack([xs, ys]).T
        us = np.random.uniform(0, maxintensity, new_npts)

        _P_new = hat_basis_matrix_2d_(newpts, knot_x, knot_y)
        _intensities = np.exp(_P_new @ sample).flatten()
        keep = us < _intensities
        newpts = newpts[keep]
        intensity_list.append(_intensities[keep])
      
        # pMSE
        syn_fn = lambda x: np.exp(hat_basis_matrix_2d_(x, knot_x, knot_y) @ sample)
        mu_syn = np.sum(alpha_0 * np.exp(sample.flatten()))
        pMSE = pMSE_int(syn=newpts, ori=data, syn_fn=syn_fn, ori_fn=my_fn, window=window, mu_syn=mu_syn)
        newpts_list.append(newpts)
        pMSE_list.append(pMSE)
        # print(pMSE)

    return pMSE_list, data, newpts_list, intensity_list
