''' This code is based on SMILE estimator implementation from ermongroup:
Repo of SMILE implementation: https://github.com/ermongroup/smile-mi-estimator
We have replicated smile estimators and run our own method in the same setup for the purpose of comparison
'''

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pickle
import time

import matplotlib.pyplot as plt
# import seaborn as sns
from utils import *
from estimators import estimate_mutual_information

dim = 20

# define the training procedure

CRITICS = {
    'separable': SeparableCritic,
    'concat': ConcatCritic,
}

BASELINES = {
    'constant': lambda: None,
    'unnormalized': lambda: mlp(dim=dim, hidden_dim=512, output_dim=1, layers=2, activation='relu').cuda(),
    'gaussian': lambda: log_prob_gaussian,
}

def save_dict(filename, dict):
    file=filename+ '.pkl'
    f = open(file, "wb")
    pickle.dump(dict, f)
    f.close()


def train_estimator(critic_params, data_params, mi_params, opt_params, **kwargs):
    """Main training loop that estimates time-varying MI."""
    # Ground truth rho is only used by conditional critic
    critic = CRITICS[mi_params.get('critic', 'separable')](
        rho=None, **critic_params).cuda()
    baseline = BASELINES[mi_params.get('baseline', 'constant')]()

    opt_crit = optim.Adam(critic.parameters(), lr=opt_params['learning_rate'])
    if isinstance(baseline, nn.Module):
        opt_base = optim.Adam(baseline.parameters(),
                              lr=opt_params['learning_rate'])
    else:
        opt_base = None

    def train_step(rho, data_params, mi_params):
        # Annoying special case:
        # For the true conditional, the critic depends on the true correlation rho,
        # so we rebuild the critic at each iteration.
        opt_crit.zero_grad()
        if isinstance(baseline, nn.Module):
            opt_base.zero_grad()

        if mi_params['critic'] == 'conditional':
            critic_ = CRITICS['conditional'](rho=rho).cuda()
        else:
            critic_ = critic

        x, y = sample_correlated_gaussian(
            dim=data_params['dim'], rho=rho, batch_size=data_params['batch_size'], cubic=data_params['cubic'])
        mi = estimate_mutual_information(
            mi_params['estimator'], x, y, critic_, baseline, mi_params.get('alpha_logit', None), **kwargs)
        loss = -mi

        loss.backward()
        opt_crit.step()
        if isinstance(baseline, nn.Module):
            opt_base.step()

        return mi

    # Schedule of correlation over iterations
    mis = mi_schedule(opt_params['iterations'])
    rhos = mi_to_rho(data_params['dim'], mis)

    estimates = []
    for i in range(opt_params['iterations']):
        mi = train_step(rhos[i], data_params, mi_params)
        mi = mi.detach().cpu().numpy()
        estimates.append(mi)

    return np.array(estimates)
## parameters for data, critic and optimization

data_params = {
    'dim': dim,
    'batch_size': 64,
    'cubic': None
}

critic_params = {
    'dim': dim,
    'layers': 2,
    'embed_dim': 32,
    'hidden_dim': 256,
    'activation': 'relu',
}

opt_params = {
    'iterations': 10000,
    'learning_rate': 5e-4,
}

if data_params['cubic']:
    out_file = 'Smile_results_cubic_10k'
else:
    out_file = 'Smile_results_gauss_10k'
print(out_file)

mi_numpys = dict()

for critic_type in ['concat']:
    mi_numpys[critic_type] = dict()

    for estimator in ['infonce', 'nwj', 'js', 'smile']:
        t_start = time.time()
        mi_params = dict(estimator=estimator, critic=critic_type, baseline='unnormalized')
        mis = train_estimator(critic_params, data_params, mi_params, opt_params)
        mi_numpys[critic_type][f'{estimator}'] = mis
        t_end = time.time()
        print('Estimator:', estimator, '; Time taken:', t_end-t_start)

    estimator = 'smile'
    for i, clip in enumerate([1.0, 5.0]):
        mi_params = dict(estimator=estimator, critic=critic_type, baseline='unnormalized')
        mis = train_estimator(critic_params, data_params, mi_params, opt_params, clip=clip)
        mi_numpys[critic_type][f'{estimator}_{clip}'] = mis


save_dict(out_file, mi_numpys)