import numpy as np
import geopandas as gpd
import matplotlib.tri as mtri
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, MultiPolygon, Point
from shapely.plotting import plot_polygon, plot_points
import pymc as pm
import pytensor.tensor as pt
from src.triangulation import *
from src.GCP import *
from src.utils import generate_poisson_process_2d
from src.linnet import *
import arviz as az
from tqdm import tqdm
import os
import argparse

import scienceplots

plt.style.use('science')

# Parse arguments
parser = argparse.ArgumentParser(description='Run GCP Synthesis for Chicago Crime Data')
parser.add_argument('--eps', type=float, default=1.0, help='Epsilon value for GCP')
parser.add_argument('--resolution', type=float, default=50.0, help='Resolution for discretization')
parser.add_argument('--seed', type=int, default=0, help='Random seed')
parser.add_argument('--rep', type=int, default=30, help='Number of synthetic datasets')
parser.add_argument('--burnin', type=int, default=10000, help='Number of burn-in samples')
args = parser.parse_args()
# Load Data
crime = gpd.read_file('data/collapsed/crimes.shp')
street = gpd.read_file('data/collapsed/streets.shp')

# Drop duplicated points
crime = crime.drop_duplicates(subset=['geometry'])
crime = crime.reset_index(drop=True)

# Create Linnet object
linnet = Linnet(edges = street)
n = crime.shape[0]

# Discretize
discretized = linnet.discretize(args.resolution)
discretized.count_points(crime)

N = discretized.nodes.shape[0]
P = discretized.projection(crime) # Projection matrix
alpha_0 = discretized.get_dual_lengths()
y = np.hstack([np.zeros(N), np.ones(n)]).reshape(1, -1)
alpha = np.hstack([alpha_0, np.zeros(n)]).reshape(1, -1)
total_len = np.sum(alpha_0)
lambda0 = n / total_len 


eps = args.eps
delta = 1/n
dGmatrix = discretized.shortest_path
dGmatrix[np.diag_indices(N)] = 0
perturb = np.min(dGmatrix[dGmatrix > 0])

# Find the upper bound R of sigma/lengthscale
dRmatrix = pairdistR(discretized.nodes, discretized)
node_start = discretized.edges['node_start'].values
node_end = discretized.edges['node_end'].values
num_edges = len(node_start)
shortest_dG = np.full((num_edges, num_edges), np.inf)
for i in range(num_edges):
    for j in range(num_edges):
        shortest_dG[i, j] = min(dGmatrix[node_start[i], node_start[j]], dGmatrix[node_start[i], node_end[j]], dGmatrix[node_end[i], node_start[j]], dGmatrix[node_end[i], node_end[j]])

furthest_dR = np.full((num_edges, num_edges), np.inf)
for i in range(num_edges):
    for j in range(num_edges):
        furthest_dR[i, j] = max(dRmatrix[node_start[i], node_start[j]], dRmatrix[node_start[i], node_end[j]], dRmatrix[node_end[i], node_start[j]], dRmatrix[node_end[i], node_end[j]])

shortest_dG[np.triu_indices(num_edges)] = np.inf
furthest_dR[np.diag_indices(num_edges)] = 0

is_, js_ = np.where(shortest_dG < perturb)
dRs = furthest_dR[is_, js_]
S = np.sum(dRs)
ratio = eps * np.sqrt(delta / S / 8)

# RUN GCP
X = discretized.nodes.get_coordinates().to_numpy()

# pymc model
## Check if the trace already exists
if os.path.exists(f'data/trace/linnet_eps_{eps}_seed_{args.seed}.nc'):
    trace = az.from_netcdf(f'data/trace/linnet_eps_{eps}_seed_{args.seed}.nc')
    print('Trace already exists')
else:
    with pm.Model() as model:
        # weights
        mean_func = pm.gp.mean.Constant(lambda0)
        ls = pm.InverseGamma('ls', alpha=1, beta=1)
        sigma = np.sqrt(ratio * ls)
        cov_func = Resistancecov_exp(sigma=sigma, linnet=discretized, ls=ls, alpha=1.0)
        gp = pm.gp.Latent(mean_func=mean_func, cov_func=cov_func)
        weights = gp.prior('weights', X=X)

        # likelihood
        logeta = pt.concatenate([weights, pt.dot(weights, P.T)]).reshape((1, -1))
        loglik = pm.Potential('loglik', y * logeta - alpha * pm.math.exp(logeta))
        
        # sample
        trace = pm.sample(1000, tune=args.burnin, random_seed=args.seed, cores=1, chains=1)

    trace.to_netcdf(f'data/trace/linnet_eps_{eps}_seed_{args.seed}.nc')

# Synthesis
np.random.seed(args.seed)

newpts_list = []
intensities_list = []

n_sample = args.rep

for i in tqdm(range(n_sample)):
    sample = trace.posterior.weights.values[0][-i].reshape(-1, 1)

    # Thinning
    maxintensity = np.max(np.exp(sample))
    new_npts = np.random.poisson(maxintensity * total_len)

    # Generate new points
    rs = np.random.uniform(0, 1, new_npts)
    newpts = gpd.GeoSeries(discretized.edges.unary_union.interpolate(rs, normalized=True))
    P_new = discretized.projection(newpts)
    _intensities = np.exp(P_new @ sample)
    keep = np.random.uniform(0, maxintensity, new_npts) < _intensities.flatten()
    newpts = newpts[keep]
    intensities_newpts = _intensities[keep]
    newpts.reset_index(drop=True, inplace=True)
    newpts_list.append(newpts)
    intensities_list.append(intensities_newpts)

# Plot K function comparison

rs = np.linspace(0, 800, 100)

fig1, ax1 = plt.subplots(1, 1, figsize=(5, 5))

# Original data
K_original = K_est_linnet_hom(points=crime.geometry, linnet=linnet, r_values=rs, correction=True)
ax1.plot(rs, K_original, color='tab:red', label='Original Data')

# Synthetic data
K_synth = np.zeros((n_sample, len(rs)))
for i in tqdm(range(n_sample)):
    K_synth[i] = K_est_linnet_inhom(points=newpts_list[i], intensities=intensities_list[i], linnet=linnet, r_values=rs, correction=True)
K_synth_mean = np.mean(K_synth, axis=0)
K_synth_upper = np.percentile(K_synth, 97.5, axis=0)
K_synth_lower = np.percentile(K_synth, 2.5, axis=0)
ax1.plot(rs, K_synth_mean, color='tab:blue', label='Synthetic Data')
ax1.fill_between(rs, K_synth_upper, K_synth_lower, color='tab:blue', alpha=0.3)
ax1.set_xlabel('r (m)')
ax1.set_ylabel('K(r)')
ax1.legend(loc='upper left', fontsize=12)
fig1.tight_layout()
fig1.savefig(f'plots/linnet_K_comparison_eps_{eps}.pdf', transparent=True)

# Laplace Mechanism
hist = discretized.edges['count']
newpts_Laplace = []
for i in tqdm(range(n_sample)):
    hist_perturbed = hist + np.random.laplace(0, 1/eps, hist.shape)
    hist_perturbed = np.maximum(hist_perturbed.astype(int), 0)
    count_Lap = np.random.poisson(hist_perturbed)
    newpts_perturbed = discretized.edges.sample_points(count_Lap).dropna().explode().reset_index(drop=True)
    newpts_Laplace.append(newpts_perturbed)

# Plot K function comparison
fig2, ax2 = plt.subplots(1, 1, figsize=(5, 5))

# Original data
K_original = K_est_linnet_hom(points=crime.geometry, linnet=linnet, r_values=rs, correction=True)
ax2.plot(rs, K_original, color='tab:red', label='Original Data')

# Synthetic data (LM)
K_Laplace = np.zeros((n_sample, len(rs)))
for i in tqdm(range(n_sample)):
    K_Laplace[i] = K_est_linnet_hom(points=newpts_Laplace[i], linnet=linnet, r_values=rs, correction=True)

# Drop NA
K_Laplace = K_Laplace[~np.any(np.isnan(K_Laplace), axis=1)]
K_Laplace_mean = np.mean(K_Laplace, axis=0)
K_Laplace_upper = np.percentile(K_Laplace, 97.5, axis=0)
K_Laplace_lower = np.percentile(K_Laplace, 2.5, axis=0)

ax2.plot(rs, K_Laplace_mean, color='tab:green', label='Laplace Mechanism')
ax2.fill_between(rs, K_Laplace_upper, K_Laplace_lower, color='tab:green', alpha=0.3)

ax2.set_xlabel('r (m)')
ax2.set_ylabel('K(r)')
ax2.legend(loc='upper left', fontsize=12)
fig2.tight_layout()
fig2.savefig(f'plots/linnet_K_comparison_Laplace_eps_{eps}_seed_{args.seed}.pdf', transparent=True)

# K function MISE
dr = rs[1] - rs[0]
mise_gcp_vals = np.power(K_synth - K_original, 2).sum(axis=1) * dr
mise_Laplace_vals = np.power(K_Laplace - K_original, 2).sum(axis=1) * dr

mise_gcp = np.mean(mise_gcp_vals)
mise_Laplace = np.mean(mise_Laplace_vals)
std_mise_gcp = np.std(mise_gcp_vals)
std_mise_Laplace = np.std(mise_Laplace_vals)

# MISE2

ratio_gcp = K_synth / K_original
ratio_Laplace = K_Laplace / K_original

mise2_gcp_vals = np.array([np.sum((r[np.isfinite(r)] - 1)**2) * dr for r in ratio_gcp])
mise2_Laplace_vals = np.array([np.sum((r[np.isfinite(r)] - 1)**2) * dr for r in ratio_Laplace])

mise2_gcp = np.mean(mise2_gcp_vals)
mise2_Laplace = np.mean(mise2_Laplace_vals)
std_mise2_gcp = np.std(mise2_gcp_vals)
std_mise2_Laplace = np.std(mise2_Laplace_vals)

pd.DataFrame({
    'eps': [eps],
    'seed': [args.seed],
    'MISE_GCP': [mise_gcp],
    'MISE_Laplace': [mise_Laplace],
    'std_MISE_GCP': [std_mise_gcp],
    'std_MISE_Laplace': [std_mise_Laplace],
    'MISE2_GCP': [mise2_gcp],
    'MISE2_Laplace': [mise2_Laplace],
    'std_MISE2_GCP': [std_mise2_gcp],
    'std_MISE2_Laplace': [std_mise2_Laplace],
    'npts_GCP': [np.mean([len(newpts) for newpts in newpts_list])],
    'npts_std_GCP': [np.std([len(newpts) for newpts in newpts_list])],
    'npts_Laplace': [np.mean([len(newpts) for newpts in newpts_Laplace])],
    'npts_std_Laplace': [np.std([len(newpts) for newpts in newpts_Laplace])]
}).to_csv(f'summary_linnet/MISE_eps_{eps}_seed_{args.seed}.csv', index=False)