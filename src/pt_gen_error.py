#!/usr/bin/env python3

import numpy as np
import scipy as sp
from tqdm import tqdm
from tqdm.contrib import tenumerate

import matplotlib.pyplot as plt

from utils import *

def bayes_error(teacher, p, su):
    x_inds = get_x_samples(p, su)
    phis = teacher['evecs'][x_inds]
    phi_outers = np.einsum('pni,pnj->ijn', phis, phis, optimize=True)

    eval_invs = np.diag(np.power(teacher['evals'], -1))[:, :, None]

    mats = eval_invs + np.power(teacher['obs_noise'], -1) * phi_outers
    invs = np.transpose(np.linalg.inv(np.transpose(mats, (2, 0, 1))), (1, 2, 0))

    traces = np.trace(invs)
    return np.mean(traces) / su.res

def mismatch_error(teacher, student, p, su):
    x_inds = get_x_samples(p, su).T

    s_gram_kerns = student['kernel'][x_inds[:, :, None], x_inds[:, None, :]]
    t_gram_kerns = teacher['kernel'][x_inds[:, :, None], x_inds[:, None, :]]

    s_lik = s_gram_kerns + student['obs_noise'] * np.eye(p)
    t_lik = t_gram_kerns + teacher['obs_noise'] * np.eye(p)

    s_kern_test_points = student['kernel'][:, x_inds].transpose(1, 2, 0)
    t_kern_test_points = teacher['kernel'][:, x_inds].transpose(1, 2, 0)

    s_component = np.linalg.solve(s_lik, s_kern_test_points)
    t_component = np.linalg.solve(t_lik, t_kern_test_points)

    t_error = np.einsum('npr,npr->nr', t_kern_test_points, t_component)
    st_cross_error = np.einsum('npr,npr->nr', t_kern_test_points, s_component)
    s_error = np.einsum('npr,npr->nr', s_component, t_lik @ s_component)

    total_err = t_error - 2 * st_cross_error + s_error
    total_err_avg_x = np.mean(total_err, axis=1)
    return np.mean(total_err_avg_x)

def theory_error(teacher_kernel, student_kernel, p, su):
    irreducible = bayes_error(teacher_kernel, p, su)
    mismatch = mismatch_error(teacher_kernel,
                              student_kernel,
                              p,
                              su)

    return irreducible + mismatch

def empirical_error(teacher, student, p, su):
    # Because kernels are fixed in this experiment, we don't
    # need to use an GP libraries that implement kernel optimization.
    # Instead, we can just directly compute the posterior mean at
    # all test points, which can be done in closed form with the
    # respective kernels.
    targ_fns = sample_teacher(teacher['kernel'], su).T

    x_inds = get_x_samples(p, su)
    all_ys = np.take_along_axis(targ_fns, x_inds, axis=0)
    all_ys += np.sqrt(teacher['obs_noise']) * su.rng.standard_normal(all_ys.shape)

    s_gram_kerns = student['kernel'][x_inds[:, None, :], x_inds[None, :, :]]
    s_lik = s_gram_kerns + student['obs_noise'] * np.eye(p)[:, :, None]

    alpha = np.linalg.solve(np.transpose(s_lik, (2, 0, 1)),
                            all_ys.T[..., None])[..., 0]

    s_kern_test_points = student['kernel'][:, x_inds]
    predictions = np.einsum(
        'rpn,np->rn', s_kern_test_points, alpha, optimize=True)

    return np.mean((targ_fns - predictions) ** 2, axis=0)

# For this experiment, kernels are all RQs
def build_model(alpha, ell, obs_noise, su):
    model = dict()
    x_range = np.linspace(su.x_min, su.x_max, su.res)
    diffs = x_range[None, :] - x_range[:, None]
    model['kernel'] = np.power(
        1 + np.power(diffs, 2) / (2 * alpha * np.power(ell, 2)),
    -alpha)

    evals, evecs = np.linalg.eigh(model['kernel'])
    model['evals'] = evals
    model['evecs'] = evecs

    model['obs_noise'] = obs_noise
    model['alpha'] = alpha
    model['ell'] = ell

    return model

def run_alpha_exp(rng,
                  alphas=[0.1, 1, 100],
                  teacher_alpha=1,
                  sigma_t=1.0,
                  sigma_s=1.0,
                  ell=1.0,
                  student_ell=1.0,
                  ds_size=np.logspace(1, 2, num=10).astype(int),
                  x_min=0,
                  x_max=10,
                  res=100,
                  n_repeats=600):

    su = SimUtils(rng, x_min=x_min, x_max=x_max, res=res, n_repeats=n_repeats)
    bayes_error_theory = np.zeros(len(ds_size))
    emp_error = np.zeros((len(alphas), len(ds_size), su.n_repeats))
    th_error = np.zeros((len(alphas), len(ds_size)))

    teacher = build_model(teacher_alpha, ell, sigma_t, su)
    students = \
        [build_model(alpha, student_ell, sigma_s, su) for alpha in alphas]

    for i, p in tenumerate(ds_size):
        bayes_error_theory[i] = theory_error(teacher,
                                             teacher,
                                             p,
                                             su)

        for j, alpha in enumerate(alphas):
            emp_error[j, i] = empirical_error(teacher,
                                              students[j],
                                              p,
                                              su)
            th_error[j, i] = theory_error(teacher,
                                          students[j],
                                          p,
                                          su)

    return {'bayes_error' : bayes_error_theory,
            'empirical_error' : emp_error,
            'theoretical_error' : th_error,
            'ds' : ds_size,
            'alphas' : alphas}

if __name__ == '__main__':
    rng = np.random.default_rng(seed=42)
    result = run_alpha_exp(rng)
    print(np.round(result['bayes_error'], 4))
    print()
    print(np.round(result['theoretical_error'], 4))
