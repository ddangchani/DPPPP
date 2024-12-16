# Synthesizer
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import scienceplots
import argparse
import sys

# Import runners
from src.GCP import run_tri, run_sqr
from src.NS import run_ns
from src.utils import generate_poisson_process_2d, generate_matern_hardcore

# Argument parser
parser = argparse.ArgumentParser(description='Run the simulation for the 2D spatial point process.')
parser.add_argument('--type', type=str, default='tri', help='The type of the spatial point process. Choose from [tri, sqr, ns].')
parser.add_argument('--fn', type=str, default='lambda1', help='The intensity function to use: from [lambda1, lambda2, lambda3, lambda4, lambda5].')
parser.add_argument('--nx', type=int, default=11, help='The number of knots in the x-direction.')
parser.add_argument('--ny', type=int, default=11, help='The number of knots in the y-direction.')
parser.add_argument('--eps', type=float, default=1.0, help='The privacy parameter.')
parser.add_argument('--reps', type=int, default=10, help='The number of repetitions.')
parser.add_argument('--seed', type=int, default=0, help='The seed for the random number generator.')
parser.add_argument('--radius', type=float, default=1.0, help='The radius for the hardcore process.')
parser.add_argument('--mean', type=int, default=1.0, help='The underlying mean intensity for the hardcore process.')
parser.add_argument('--save', type=bool, default=False, help='Save the traces.')
parser.add_argument('--prior', type=str, default='invgamma', help='The prior distribution for the lengthscale parameter. Choose from [invgamma, lognoraml, halfcauchy].')
parser.add_argument('--burnin', type=int, default=2000, help='The number of burn-in iterations.')
args = parser.parse_args()

# Print the arguments
print("Running the simulation for the 2D spatial point process.")
print("Synthesizer type: ", args.type)
print("Intensity function: ", args.fn)
print("Privacy parameter: ", args.eps)

def lambda1(x, y):
    return np.full_like(x, 20)

def lambda2(x, y):
    return np.exp(-(x**2 + y**2) / 25)

def lambda3(x, y, lambda0=0.5, lambda1=5):
    return lambda0 + lambda1 * np.exp(-((x - y)**2))

def lambda4(x, y, centers=[(3, 3), (-3, -3)], sigma=1.0):
    return sum(5 * np.exp(-((x - cx)**2 + (y - cy)**2) / (2 * sigma**2)) for cx, cy in centers)

window_dict = {
    lambda1: [0, 1, 0, 1],
    lambda2: [-10, 10, -10, 10],
    lambda3: [0, 10, 0, 10],
    lambda4: [-5, 5, -5, 5],
    'lambda5': [-10, 10, -10, 10]
}

# Settings
my_fn = eval(args.fn) if args.fn != 'lambda5' else 'lambda5'
window = window_dict[my_fn]
nx = args.nx
ny = args.ny

# Privacy parameters
eps = args.eps

# Run the experiments
result_df = pd.DataFrame(columns=['seed', 'pMSE', 'data', 'newpts', 'intensity'])

np.random.seed(seed=args.seed)
seeds = np.random.randint(0, 1000, args.reps)

for i, seed in enumerate(seeds):
    # Generate the data
    if my_fn == 'lambda5':
        data = generate_matern_hardcore(r=args.radius, mean=args.mean, window=window, seed=seed, type=args.materntype)
    else:
        data = generate_poisson_process_2d(my_fn, window, seed=seed)

    if args.type == 'tri':
        pMSE_list, data, newpts_list, intensity_list = run_tri(my_fn, data, window, int(seed), nx, ny, eps, save=args.save, prior=args.prior, burnin=args.burnin)
    elif args.type == 'sqr':
        pMSE_list, data, newpts_list, intensity_list = run_sqr(my_fn, data, window, int(seed), nx, ny, eps, save=args.save, prior=args.prior, burnin=args.burnin)
    elif args.type == 'ns':
        pMSE_list, data, newpts_list, intensity_list = run_ns(my_fn, data, window, seed, nx, ny, eps)
    else:
        raise ValueError("Invalid type. Choose from [tri, sqr, ns].")

    for pMSE, newpts, intensities in zip(pMSE_list, newpts_list, intensity_list):
        result_df = pd.concat([result_df, pd.DataFrame({'seed': seed, 'pMSE': pMSE, 'data': [data.tolist()], 'newpts': [newpts.tolist()], 'intensity': [intensities.tolist()]})])

    # Print the progress
    sys.stdout.write(f"\rProgress: {i+1}/{args.reps} \n")
    sys.stdout.flush()

# Save the results
result_df.to_pickle(f'data/results_{args.type}_{my_fn.__name__}_eps{eps}.pkl')