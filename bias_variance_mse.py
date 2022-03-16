import torch
import numpy as np
import pandas as pd
import stan
import json
from stan_helpers import unconstrained_dim, unconstrain_sample
from pathlib import Path
from tqdm.auto import trange
from random_f import LazyMixtureOfSinusoids, power_law
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--problem', required=True)
parser.add_argument('--num-runs', default=4, type=int)
parser.add_argument('--root-dir', default=Path('.'), type=Path)
parser.add_argument('--lambdas', default='1.000,1.122,1.259,1.413,1.585,1.778,1.995,2.239,2.512,2.818,3.162,3.548,3.981,4.467,5.012,5.623,6.310,7.079,7.943,8.913,10.000', type=str)
parser.add_argument('--freq-min', default=1, type=int)
parser.add_argument('--freq-max', default=300, type=int)
parser.add_argument('--alpha-min', default=-3, type=int)
parser.add_argument('--alpha-max', default=0, type=int)
parser.add_argument('--num-alpha', default=50, type=int)
parser.add_argument('--power-law-norm', default='max', type=str, choices=['max', 'mean', 'none'])
parser.add_argument('--num-fs', default=200, type=int)
parser.add_argument('--num-true', default=10000, type=int)
parser.add_argument('--num-subs', default=1000, type=int)
parser.add_argument('--T', default=10, type=int)
args = parser.parse_args()


def load_nuts_results(mdl: stan.model.Model, problem: str, run:int = 1) -> np.ndarray:
    df = pd.read_csv(args.root_dir / problem / f'nuts_{run}.csv', comment='#')
    return unconstrain_sample(mdl, df[list(mdl.constrained_param_names)].to_numpy())


def load_isvi_results(problem, lam, run=1) -> pd.DataFrame:
    return pd.read_csv(args.root_dir / problem / f'isvi_{lam}_{run}.csv', comment='#')


def load_advi_results(mdl: stan.model.Model, problem: str, run: int = 1) -> dict:
    df = pd.read_csv(args.root_dir / problem / f'advi_{run}.csv', comment='#')
    unconstrained_vals = unconstrain_sample(mdl, df[list(mdl.constrained_param_names)].to_numpy())
    m = unconstrained_vals[0]
    c = np.einsum('ia,ib->ab', unconstrained_vals[1:]-m, unconstrained_vals[1:]-m)/(len(df)-1)
    return {'mean': m, 'cov': c}


def load_stan_model(problem):
    stan_file = args.root_dir / problem / f'{problem}.stan'
    data_file = args.root_dir / problem / 'data.json'
    if data_file.exists():
        with open(data_file, 'r') as f:
            data_arg = json.load(f)
    else:
        data_arg = {}
    with open(stan_file, 'r') as f:
        return stan.build(f.read(), data=data_arg)


def isvi_mu_cov(isvi_df, num):
    mean_cols = [c for c in isvi_df.columns if c.startswith('mu_')]
    omega_cols = [c for c in isvi_df.columns if c.startswith('omega_')]
    assert len(mean_cols) == len(omega_cols), "Expected #mean = #omega (diagonal cov)!"
    mus = isvi_df[mean_cols].iloc[:num].to_numpy().T
    omegas = isvi_df[omega_cols].iloc[:num].to_numpy()
    covariances = np.stack([np.diag(np.exp(2*om)) for om in omegas], axis=-1)
    return mus, covariances


def advi_mu_cov(run):
    return advi_fits[run]['mean'][:, None], advi_fits[run]['cov'][:, :, None]


runs = tuple(range(1, 1+args.num_runs))
lambdas = [l.strip() for l in args.lambdas.split(',')]

stan_model = load_stan_model(args.problem)
print("Loaded", args.problem, "with (constrained) parameters:", *stan_model.constrained_param_names)

print("Loading and unconstraining ADVI results...", end=" ")
advi_fits = [load_advi_results(stan_model, args.problem, r) for r in runs]
print("done.")

print("Loading and unconstraining NUTS results...", end=" ")
nuts_samples = np.concatenate([load_nuts_results(stan_model, args.problem, r) for r in runs], axis=0)
print("done.")

print("Loading ISVI results...", end=" ")
isvi_dict = {
    l: pd.concat([load_isvi_results(args.problem, l, r) for r in runs])
    for l in lambdas
}
print("done.")

# Shuffle samples to destroy autocorrelations
nuts_samples = nuts_samples[np.random.permutation(len(nuts_samples))]
isvi_dict = {l: df.sample(frac=1.) for l, df in isvi_dict.items()}

# dimensionality of the unconstrained space
dim = unconstrained_dim(stan_model)
# frequencies of sinusoids, in units of cycles-per-x where x is the units of the unconstrained parameter space
freqs = np.arange(args.freq_min, args.freq_max+1).astype('float32')
# Which alphas to experiment with
alphas = np.linspace(args.alpha_min, args.alpha_max, args.num_alpha)

# COMPUTE BIAS AND VARIANCE ON GRID

true_ev = np.zeros((args.num_fs, len(alphas)))
nuts_ev = np.zeros((args.num_fs, len(alphas), args.num_subs))
advi_ev = np.zeros((args.num_fs, len(alphas), len(runs)))
isvi_ev = np.zeros((args.num_fs, len(alphas), args.num_subs, len(lambdas)))
t_history = np.zeros((args.num_fs, len(freqs), dim))
phase_history = np.zeros((args.num_fs, len(freqs)))

x_true = nuts_samples[:args.num_true, :].T
x_nuts = nuts_samples[args.num_true:args.num_true+args.num_subs*args.T, :].T

mu_isvi = [isvi_mu_cov(df, args.num_subs*args.T)[0] for l, df in isvi_dict.items()]
cov_isvi = [isvi_mu_cov(df, args.num_subs*args.T)[1] for l, df in isvi_dict.items()]

mu_advi = [advi_mu_cov(r)[0] for r in range(len(runs))]
cov_advi = [advi_mu_cov(r)[1] for r in range(len(runs))]


f_true = LazyMixtureOfSinusoids(dim, freqs)
f_nuts = LazyMixtureOfSinusoids(dim, freqs)
f_advi = [LazyMixtureOfSinusoids(dim, freqs) for _ in mu_advi]
f_isvi = [LazyMixtureOfSinusoids(dim, freqs) for _ in mu_isvi]

progbar = trange(args.num_fs)
for i in progbar:
    # New random f by drawing new directions 't' and new random phases
    phases = np.random.rand(freqs.size)*2*np.pi
    f_true.randomize_t().update_table_x(x_true)
    
    t_history[i, :, :] = f_true._t
    phase_history[i, :] = phases

    # Copy the same random 't' to all LazyMixture objects and update their precomputed tables
    f_nuts.set_t(f_true._t).update_table_x(x_nuts)
    for f, m, c in zip(f_advi, mu_advi, cov_advi):
        f.set_t(f_true._t).update_table_gauss(m, c)
    for f, m, c in zip(f_isvi, mu_isvi, cov_isvi):
        f.set_t(f_true._t).update_table_gauss(m, c)
    
    # Given the precomputed tables above, quickly collate E[f] values for a variety of power-law-decay values
    for j, a in enumerate(alphas):
        true_ev[i, j] = np.mean(f_true.apply(power_law(freqs, a, args.power_law_norm), phases))
        nuts_ev[i, j, :] = np.mean(np.reshape(f_nuts.apply(power_law(freqs, a, args.power_law_norm), phases), (args.num_subs, args.T)), axis=1)
        for k, f in enumerate(f_advi):
            advi_ev[i, j, k] = f.apply(power_law(freqs, a, args.power_law_norm), phases)
        for k, f in enumerate(f_isvi):
            isvi_ev[i, j, :, k] = np.mean(np.reshape(f.apply(power_law(freqs, a, args.power_law_norm), phases), (args.num_subs, args.T)), axis=1)


output_file = args.root_dir / args.problem / 'bias_variance_mse.dat'
torch.save({
    'args': args,
    'true_ev': true_ev,
    'nuts_ev': nuts_ev,
    'advi_ev': advi_ev,
    'isvi_ev': isvi_ev,
    't_history': t_history,
    'phase_history': phase_history,
    }, output_file)
