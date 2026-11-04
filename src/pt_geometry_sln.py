#!/usr/bin/env python3

import numpy as np
import scipy as sp

import matplotlib.pyplot as plt

from utils import *

def gaussian_activations(stims, population):
    stim_tuning = (stims[:, None] - population['centers'][None, :]) / \
        (np.sqrt(2) * population['bws'][None, :])
    return np.exp(-(stim_tuning ** 2))

def cosine_activations(stims, population):
    stim_tuning = (stims[:, None] - population['centers'][None, :]) / \
        (np.sqrt(2) * population['bws'][None, :])
    return np.cos(stim_tuning)

def prior_draws(populations, sigma_w, su):
    n_neurons = populations['base_params']['n_neurons']

    draws = dict()
    stims = populations['base_params']['stims']
    activation_function = populations['base_params']['tf']
    for param, population in populations.items():
        if param == 'base_params':
            continue

        r = activation_function(stims, population)
        weights = su.rng.normal(loc=0, scale=sigma_w, size=n_neurons)
        draws[param] = r @ weights
        draws[param] -= np.mean(draws[param])

    return draws

def solve(populations, sigma_w, sigma_o, su, ds_size=10):
    # y^* = cos(x)
    activation_function = populations['base_params']['tf']
    stims = populations['base_params']['stims']
    xs = su.rng.choice(
        np.arange(stims.shape[0]), size=ds_size, replace=False
    )
    ys = np.cos(stims)[xs] + sigma_o * \
        su.rng.normal(0, 1, size=ds_size)

    n_neurons = populations['base_params']['n_neurons']
    lamb = (sigma_o / sigma_w) ** 2

    solutions = dict()
    for param, population in populations.items():
        if param == 'base_params':
            continue

        r = activation_function(stims, population)
        optim_weights = np.sqrt(n_neurons) * \
            np.linalg.inv(
                r[xs].T @ r[xs] + n_neurons * lamb * np.eye(n_neurons)
            ) @ r[xs].T @ ys

        solutions[param] = 1 / np.sqrt(n_neurons) * (r @ optim_weights)

    return solutions, (stims[xs], ys)

def projection(populations, su, scale=0.1):
    n_neurons = populations['base_params']['n_neurons']
    activation_function = populations['base_params']['tf']
    projection_mat = su.rng.normal(loc=0, scale=scale, size=(n_neurons, 3))
    stims = populations['base_params']['stims']

    projections = dict()
    for param, population in populations.items():
        if param == 'base_params':
            continue

        r = activation_function(stims, population)
        projections[param] = r @ projection_mat

    return projections

def generate_bw_het_example(rng,
                            n_neurons=1_000,
                            x_min=-np.pi,
                            x_max=np.pi,
                            res=1_000,
                            sigma_w=200,
                            sigma_o=0.7,
                            L=6,
                            beta=0.8,
                            gain=0.1,
                            shapes=[0.1, 1, 3, 1000]):
    su = SimUtils(rng, x_min=x_min, x_max=x_max, res=res)
    populations = dict()

    populations['base_params'] = dict()
    populations['base_params']['n_neurons'] = n_neurons
    stims = np.linspace(x_min, x_max, num=res)
    populations['base_params']['stims'] = stims
    populations['base_params']['tf'] = gaussian_activations

    for shape in shapes:
        populations[shape] = dict()
        populations[shape]['centers'] = rng.uniform(
            -L, L, size=n_neurons
        )
        populations[shape]['bws'] = rng.gamma(
            shape, beta / shape, size=n_neurons
        )
        populations[shape]['gain'] = np.ones(n_neurons) * gain

    result = dict()
    result['projections'] = projection(populations, su)
    result['prior_draws'] = prior_draws(populations, sigma_w, su)
    slns, pts = solve(populations, sigma_w, sigma_o, su)
    result['solutions'] = slns
    result['sample_points'] = pts
    result['stims'] = stims

    return result

def generate_center_het_example(rng,
                                n_neurons=1_000,
                                x_min=-np.pi,
                                x_max=np.pi,
                                res=1_000,
                                sigma_w=200,
                                sigma_o=0.7,
                                tuning_var=0.1,
                                bw=0.1,
                                n_center=0,
                                L=6,
                                gain=0.1,
                                lmbds=[0.1, 0.5, 0.9, 0.999]):
    su = SimUtils(rng, x_min=x_min, x_max=x_max, res=res)
    populations = dict()

    populations['base_params'] = dict()
    populations['base_params']['n_neurons'] = n_neurons
    stims = np.linspace(x_min, x_max, num=res)
    populations['base_params']['stims'] = stims
    populations['base_params']['tf'] = gaussian_activations

    highlight_inds = dict()
    fixed_shape = 1.35  # aids in illustration

    for lm in lmbds:
        populations[lm] = dict()
        populations[lm]['bws'] = np.sqrt(rng.gamma(
            fixed_shape, bw / fixed_shape, size=n_neurons
        ))

        n_tuning_centers = rng.normal(
            loc=n_center, scale=tuning_var, size=n_neurons)
        u_tuning_centers = rng.uniform(low=-L, high=L, size=n_neurons)

        populations[lm]['centers'] = np.where(
            rng.uniform(0, 1, size=n_neurons) < lm,
            n_tuning_centers,
            u_tuning_centers
        )

        populations[lm]['gains'] = np.ones(n_neurons) * gain
        std2 = 2 * np.sqrt(tuning_var)
        highlight_inds[lm] = np.argwhere(
            np.logical_and(
                populations['base_params']['stims'] > (n_center - std2),
                populations['base_params']['stims'] < (n_center + std2)
            )
        )

    result = dict()
    result['projections'] = projection(populations, su)
    result['prior_draws'] = prior_draws(populations, sigma_w, su)
    slns, pts = solve(populations, sigma_w, sigma_o, su)
    result['solutions'] = slns
    result['sample_points'] = pts
    result['highlight_inds'] = highlight_inds
    result['stims'] = stims

    return result

if __name__ == '__main__':
    rng = np.random.default_rng(42)
    bw_res = generate_bw_het_example(rng)
    center_res = generate_center_het_example(rng)
