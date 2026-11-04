#!/usr/bin/env python3

import numpy as np
import scipy as sp
from tqdm import tqdm
from tqdm.contrib import tenumerate

import matplotlib.pyplot as plt

from utils import *

import warnings

def empirical_error(p, student, teacher, s_lamb, targ_fns, su):
    x_test = np.linspace(su.x_min, su.x_max, su.res).reshape(-1, 1)

    x_inds = get_x_samples(p, su).T
    all_ys = np.take_along_axis(targ_fns.T, x_inds.T, axis=0).T
    all_ys += np.sqrt(teacher['obs_noise']) * \
        su.rng.standard_normal(all_ys.shape)

    s_full_kern = (1 - s_lamb) * student['rbf'] + s_lamb * student['env']

    s_gram = s_full_kern[x_inds[:, :, None], x_inds[:, None, :]]
    s_lik = s_gram + student['obs_noise'] * np.eye(p)

    alpha = np.linalg.solve(s_lik,
                            all_ys[..., None])[..., 0]

    s_kern_test_points = s_full_kern[:, x_inds]
    predictions = np.einsum(
        'rnp,np->rn', s_kern_test_points, alpha, optimize=True)

    return np.mean((targ_fns.T - predictions) ** 2, axis=0)

def marginal_likelihood(p,
                        student,
                        teacher,
                        t_lamb,
                        s_lamb,
                        su):
    x_inds = get_x_samples(p, su).T

    s_gram_rbf = student['rbf'][x_inds[:, :, None], x_inds[:, None, :]]
    s_gram_rbf_bc = s_gram_rbf[None, :, :, :]
    t_gram_rbf = teacher['rbf'][x_inds[:, :, None], x_inds[:, None, :]]
    t_gram_rbf_bc = t_gram_rbf[None, :, :, :]

    s_gram_env = student['env'][x_inds[:, :, None], x_inds[:, None, :]]
    s_gram_env_bc = s_gram_env[None, :, :, :]
    t_gram_env = teacher['env'][x_inds[:, :, None], x_inds[:, None, :]]
    t_gram_env_bc = t_gram_env[None, :, :, :]

    t_lamb_broad = t_lamb[:, None, None, None]
    s_lamb_broad = s_lamb[:, None, None, None]

    t_kern = (1 - t_lamb_broad) * t_gram_rbf_bc + t_lamb_broad * t_gram_env_bc \
        + teacher['obs_noise'] * np.eye(p)
    s_kern = (1 - s_lamb_broad) * s_gram_rbf_bc + s_lamb_broad * s_gram_env_bc \
        + student['obs_noise'] * np.eye(p)

    tr_term = -0.5 * np.trace(
        np.transpose(t_kern @ np.linalg.inv(s_kern), (2, 3, 0, 1))
    )
    log_det_term = -0.5 * np.linalg.slogdet(s_kern)[1]
    reg_term = -p / 2 * np.log(2 * np.pi)

    return tr_term + log_det_term + reg_term

def build_kernel(mu_r, sigma_r, bw_rbf, bw_env, obs_noise, su):
    # The population kernel is a weighted mixture of a standard RBF kernel and
    # a Gaussian envelope kernel. Because we are testing the student kernel
    # over different mixture weights, we just build and provide the respective
    # kernels here
    x_range = np.linspace(su.x_min, su.x_max, num=su.res)
    stim_diffs = x_range[:, None] - x_range[None, :]
    r_centered_stim = x_range[:, None] - mu_r

    sig_e = np.power(2 / (bw_env ** 2) + 1 / (sigma_r ** 2), -0.5)
    sig_s = 2 * (bw_env ** 2) + (bw_env ** 4) / (sigma_r ** 2)
    sig_m = 2 * (sigma_r ** 2) + (bw_env ** 2)

    rbf_k = np.sqrt(np.pi) * bw_rbf * np.exp(
        -(stim_diffs ** 2) / (4 * bw_rbf ** 2)
    )
    env_k = sig_e / sigma_r * np.exp(
       -(r_centered_stim ** 2) / (2 * sig_m) - \
        (stim_diffs ** 2) / (2 * sig_s) - \
        (r_centered_stim.T ** 2) / (2 * sig_m)
    )

    # regularize to share scale
    rbf_k /= np.mean(np.diag(rbf_k))
    env_k /= np.mean(np.diag(env_k))

    return {
        'rbf' : rbf_k,
        'env' : env_k,
        'obs_noise' : obs_noise
    }

def run_reward_exp(rng,
                   sigma_s=0.8,
                   sigma_t=0.8,
                   bw_rbf=0.1,
                   bw_env=0.1,
                   sigma_r=0.01,
                   mu_r=0.5,
                   t_lamb=0.7,
                   ds_size=[10, 20, 50, 100],
                   nvs=[0.2, 0.3, 0.55, 0.8],
                   n_repeats=100,
                   x_min=-1,
                   x_max=1,
                   res=100,
                   adaptive_noise=False):
    su = SimUtils(rng, x_min=x_min, x_max=x_max, res=res, n_repeats=n_repeats)
    lmbds = np.linspace(0, 1, num=100)
    eg_adapt = np.zeros((len(ds_size), n_repeats))
    eg_fixed = np.zeros_like(eg_adapt)
    mls = np.zeros((len(ds_size), len(lmbds), n_repeats))

    teacher = build_kernel(mu_r, sigma_r, bw_rbf, bw_env, sigma_t, su)
    teacher_kernel = (1 - t_lamb) * teacher['rbf'] + t_lamb * teacher['env']
    targ = sample_teacher(teacher_kernel, su)
    student = build_kernel(mu_r, sigma_r, bw_rbf, bw_env, sigma_s, su)

    for i, p in tenumerate(ds_size):
        if adaptive_noise:
            student['obs_noise'] = nvs[i]
        mls[i] = marginal_likelihood(p,
                                     student,
                                     teacher,
                                     t_lamb * np.ones_like(lmbds),
                                     lmbds,
                                     su)
        eg_adapt[i] = empirical_error(p,
                                      student,
                                      teacher,
                                      lmbds[np.argmax(mls[i].mean(axis=-1))],
                                      targ,
                                      su)
        if adaptive_noise:
            student['obs_noise'] = nvs[0]
        eg_fixed[i] = empirical_error(p,
                                      student,
                                      teacher,
                                      0,
                                      targ,
                                      su)

    return {
        'eg_adapt' : eg_adapt,
        'eg_fixed' : eg_fixed,
        'mls' : mls,
        'ds' : ds_size,
        'targs' : targ,
        'nvs' : nvs,
        't_lamb' : t_lamb,
        'sigma_r' : sigma_r,
        'bw_rbf' : bw_rbf,
        'bw_env' : bw_env,
        'mu_r' : mu_r
    }, teacher

if __name__ == '__main__':
    rng = np.random.default_rng(42)
    result = run_reward_exp(rng)
