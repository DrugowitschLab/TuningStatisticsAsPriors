#!/usr/bin/env python3

import numpy as np
import scipy as sp

import matplotlib.pyplot as plt

from utils import *
from pt_geometry_sln import solve, prior_draws

import copy

def sigmoid_activations(stims, population):
    stim_tuning = (stims[:, None] - population['centers'][None, :]) / \
        (2 * population['bws'][None, :] ** 2)
    return population['gains'][None, :] * (sp.special.erf(stim_tuning) + 1)

def sigmoid_population(rng,
                       n_neurons=1_000,
                       x_min=-3,
                       x_max=3,
                       L=6,
                       res=1_000,
                       ds_size=10,
                       beta=0.5,
                       shape=3,
                       n_repeats=1,
                       sigma_w=200,
                       sigma_o=0.7):
    su = SimUtils(rng, x_min=x_min, x_max=x_max, n_repeats=n_repeats, res=res)
    populations = dict()
    populations['base_params'] = dict()
    populations['base_params']['n_neurons'] = n_neurons
    populations['base_params']['tf'] = sigmoid_activations
    stims = np.linspace(x_min, x_max, num=res)
    populations['base_params']['stims'] = stims

    base_dict = dict()
    base_dict['bws'] = beta * np.ones(n_neurons)
    base_dict['gains'] = 0.5 * np.ones(n_neurons)
    base_dict['centers'] = np.linspace(-L, L, num=n_neurons)
    populations['homogeneous'] = base_dict

    gain_het = copy.deepcopy(base_dict)
    gain_het['gains'] = np.linspace(0.1, 1, num=n_neurons)
    populations['gain'] = gain_het

    bw_het = copy.deepcopy(base_dict)
    bw_het['bws'] = rng.gamma(shape, beta / shape, size=n_neurons)
    populations['bws'] = bw_het

    solutions, sample_pts = solve(
        populations, sigma_w, sigma_o, su, ds_size=ds_size
    )

    ret = dict()
    ret['stims'] = stims
    ret['solutions'] = solutions
    ret['sample_points'] = sample_pts
    ret['prior_draws'] = prior_draws(populations, sigma_w, su)
    ret['tuning_curves'] = dict()
    for k in solutions.keys():
        ret['tuning_curves'][k] = sigmoid_activations(stims, populations[k])

    return ret

if __name__ == '__main__':
    rng = np.random.default_rng(42)
    res = sigmoid_population(rng)
