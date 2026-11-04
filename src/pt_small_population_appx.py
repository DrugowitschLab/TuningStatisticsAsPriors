#!/usr/bin/env python3

import numpy as np
import scipy as sp
from tqdm import tqdm
from tqdm.contrib import tenumerate

from utils import *

def run_rq_exp(pop_sizes,
               su,
               alpha=1,
               ell=1,
               g=1,
               G=0.5,
               L=6):
    ret = dict()
    stims = np.linspace(su.x_min, su.x_max, num=su.res)
    stim_diffs = stims[:, None] - stims[None, :]

    analytic_kernel = g * np.sqrt(np.pi) * \
        sp.special.gamma(alpha - 0.5) / sp.special.gamma(alpha) * \
        np.sqrt(alpha) * ell * \
        np.power(1 + (stim_diffs ** 2) / (4 * alpha * (ell ** 2)), -alpha + 0.5)
    ret['analytic_kernel'] = analytic_kernel
    ret['pop_sizes'] = pop_sizes
    ret['empirical_kernels'] = np.zeros(
        (len(pop_sizes), su.res, su.res, su.n_repeats)
    )
    for i, ps in tenumerate(pop_sizes):
        centers = su.rng.uniform(-L, L, size=(ps, su.n_repeats))
        widths = su.rng.gamma(
            alpha, 1 / (alpha * (ell ** 2)), size=(ps, su.n_repeats)
        )
        gains = np.sqrt(su.rng.uniform(g - G, g + G, size=(ps, su.n_repeats)))
        act = (((stims[None, :, None] - centers[:, None, :]) ** 2) * \
               widths[:, None, :] / 2)
        r = gains[:, None, :] * np.exp(-act)
        ret['empirical_kernels'][i] = np.einsum(
            'rnp,rmp->nmp', r, r
        ) / ps * 2 * L

    return ret

def run_rbf_exp(pop_sizes,
                su,
                bw=0.1,
                g=1,
                G=0.5,
                L=3):
    ret = dict()
    stims = np.linspace(su.x_min, su.x_max, num=su.res)
    stim_diffs = stims[:, None] - stims[None, :]

    analytic_kernel = g * np.sqrt(np.pi) * bw * \
        np.exp(-((stim_diffs ** 2) / (4 * bw ** 2)))
    ret['analytic_kernel'] = analytic_kernel
    ret['pop_sizes'] = pop_sizes
    ret['empirical_kernels'] = np.zeros(
        (len(pop_sizes), su.res, su.res, su.n_repeats)
    )
    for i, ps in tenumerate(pop_sizes):
        centers = su.rng.uniform(-L, L, size=(ps, su.n_repeats))
        widths = bw * np.ones_like(centers)
        gains = np.sqrt(su.rng.uniform(g - G, g + G, size=(ps, su.n_repeats)))
        act = ((stims[None, :, None] - centers[:, None, :]) / \
               (np.sqrt(2) * widths[:, None, :])) ** 2
        r = gains[:, None, :] * np.exp(-act)
        ret['empirical_kernels'][i] = np.einsum(
            'rnp,rmp->nmp', r, r
        ) / ps * 2 * L

    return ret

def run_pop_size_approximation_exp(rng,
                                   analytic_kernel='rbf',
                                   alpha=1,
                                   ell=1,
                                   bw=0.1,
                                   g=1,
                                   G=0.5,
                                   n_repeats=500,
                                   x_min=-1,
                                   x_max=1,
                                   res=100,
                                   pop_sizes=[10, 100, 1000, 10000]):
    su = SimUtils(rng, x_min=x_min, x_max=x_max, res=res, n_repeats=n_repeats)
    assert analytic_kernel in ['rbf', 'rq']
    if analytic_kernel == 'rbf':
        return run_rbf_exp(pop_sizes, su, bw=bw, g=g, G=G)
    else:
        return run_rq_exp(pop_sizes, su, alpha=alpha, ell=ell, g=g, G=G)

if __name__ == '__main__':
    rng = np.random.default_rng(42)
    result = run_pop_size_approximation_exp(rng,
                                            analytic_kernel='rq',
                                            n_repeats=10)
