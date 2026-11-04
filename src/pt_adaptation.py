#!/usr/bin/env python3

import numpy as np
import scipy as sp
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RationalQuadratic, RBF
from sklearn.exceptions import ConvergenceWarning
from tqdm import tqdm
from tqdm.contrib import tenumerate

import matplotlib.pyplot as plt

from utils import *

import warnings

def held_out_test(p,
                  xs,
                  targ_fns,
                  alphas,
                  sigma_t,
                  sigma_s,
                  student_ell,
                  su,
                  ds_size=50,
                  hits=None):
    held_out_err = np.zeros(su.n_repeats)
    fun_idx = su.rng.choice(np.arange(targ_fns.shape[0]), size=su.n_repeats)
    x_inds = get_x_samples(ds_size, su)

    for p in range(su.n_repeats):
        if hits is not None:
            if hits[p]:
                held_out_err[p] = np.nan
                continue

        ys = targ_fns[fun_idx[p], x_inds[:, p]] + \
            np.sqrt(sigma_t) * su.rng.standard_normal(x_inds[:, p].shape)
        student_kernel = RationalQuadratic(
            length_scale=student_ell,
            alpha=alphas[p],
            length_scale_bounds='fixed',
            alpha_bounds='fixed'
        )
        student_gp = GaussianProcessRegressor(kernel=student_kernel,
                                              alpha=sigma_s,
                                              normalize_y=False)
        student_gp.fit(xs[x_inds[:, p]], ys)
        student_pred = student_gp.predict(xs)

        held_out_err[p] = np.mean((student_pred - targ_fns[fun_idx[p]]) ** 2)

    return held_out_err

def fit_and_test(p,
                 targ_fns,
                 xs,
                 student_alpha,
                 student_ell,
                 sigma_t,
                 sigma_s,
                 su,
                 adapt=True,
                 alpha_bounds=(1e-5, 100)):
    err = np.zeros(su.n_repeats)
    adapted_alphas = np.zeros_like(err)
    hits = np.zeros_like(err, dtype=bool)

    x_inds = get_x_samples(p, su)
    fun_idx = su.rng.choice(np.arange(targ_fns.shape[0]), size=su.n_repeats)
    ys = targ_fns[fun_idx[None, :], x_inds] + \
        np.sqrt(sigma_t) * su.rng.standard_normal(x_inds.shape)

    for i in range(su.n_repeats):
        student_kernel = RationalQuadratic(
            length_scale=student_ell,
            alpha=student_alpha,
            length_scale_bounds='fixed',
            alpha_bounds=alpha_bounds if adapt else 'fixed'
        )
        student_gp = GaussianProcessRegressor(
            kernel=student_kernel,
            alpha=sigma_s,
            n_restarts_optimizer=5,
            normalize_y=False,
            random_state=np.random.RandomState(su.rng.bit_generator)
        )

        student_gp.fit(xs[x_inds[:, i]], ys[:, i])
        if np.isclose(student_gp.kernel_.alpha, alpha_bounds[0]) or \
            np.isclose(student_gp.kernel_.alpha, alpha_bounds[1]):
            hits[i] = True

        adapted_alphas[i] = student_gp.kernel_.alpha

        student_pred = student_gp.predict(xs)
        err[i] = np.mean((student_pred - targ_fns[fun_idx[i]]) ** 2)

    return err, adapted_alphas, hits

def run_adaptation_exp(rng,
                       teacher_alpha=0.2,
                       student_alpha=100,
                       teacher_ell=0.01,
                       student_ell=0.01,
                       sigma_t=0.02,
                       sigma_s=0.02,
                       n_repeats=100,
                       n_funs=500,
                       ds_sizes=np.logspace(0, 2, num=10).astype(int),
                       suppress_warnings=True,
                       x_min=0,
                       x_max=10,
                       res=200):
    su = SimUtils(rng, x_min=x_min, x_max=x_max, res=res, n_repeats=n_repeats)
    if suppress_warnings:
        warnings.simplefilter("ignore", category=ConvergenceWarning)
    x_range = np.linspace(x_min, x_max, num=res).reshape(-1, 1)

    teacher_kernel = RationalQuadratic(length_scale=teacher_ell,
                                       alpha=teacher_alpha,
                                       length_scale_bounds='fixed',
                                       alpha_bounds='fixed')
    teacher_gp = GaussianProcessRegressor(teacher_kernel, alpha=sigma_t)
    targ_fns = teacher_gp.sample_y(
        x_range,
        n_samples=n_funs,
        random_state=np.random.RandomState(rng.bit_generator)
    ).T

    student_adapt = np.zeros((len(ds_sizes), n_repeats))
    student_fixed = np.zeros_like(student_adapt)
    student_adapt_held = np.zeros_like(student_adapt)
    student_fixed_held = np.zeros_like(student_adapt)
    adapted_alphas = np.zeros_like(student_adapt)
    hits = np.zeros_like(student_adapt, dtype=bool)

    for i, p in tenumerate(ds_sizes):
        adapt_errs = fit_and_test(p,
                                  targ_fns,
                                  x_range,
                                  student_alpha,
                                  student_ell,
                                  sigma_t,
                                  sigma_s,
                                  su)
        student_adapt[i] = adapt_errs[0]
        adapted_alphas[i] = adapt_errs[1]
        hits[i] = adapt_errs[2]
        student_adapt_held[i] = held_out_test(p,
                                              x_range,
                                              targ_fns,
                                              adapted_alphas[i],
                                              sigma_t,
                                              sigma_s,
                                              student_ell,
                                              su,
                                              hits=hits[i])

        fixed_errs = fit_and_test(p,
                                  targ_fns,
                                  x_range,
                                  student_alpha,
                                  student_ell,
                                  sigma_t,
                                  sigma_s,
                                  su,
                                  adapt=False)
        student_fixed[i] = fixed_errs[0]
        student_fixed_held[i] = held_out_test(p,
                                              x_range,
                                              targ_fns,
                                              fixed_errs[1],
                                              sigma_t,
                                              sigma_s,
                                              student_ell,
                                              su)

    return {
        'adapt_eg' : student_adapt,
        'fixed_eg' : student_fixed,
        'adapt_held_eg' : student_adapt_held,
        'fixed_held_eg' : student_fixed_held,
        'adapted_alphas' : adapted_alphas,
        'ds' : ds_sizes,
        'hits' : hits,
        'student_ell' : student_ell,
        'teacher_ell' : teacher_ell,
        'teacher_alpha' : teacher_alpha,
        'sigma_s' : sigma_s,
        'sigma_t' : sigma_t,
        'student_alpha_f' : student_alpha
    }

if __name__ == '__main__':
    rng = np.random.default_rng(42)
    res = run_adaptation_exp(rng)
