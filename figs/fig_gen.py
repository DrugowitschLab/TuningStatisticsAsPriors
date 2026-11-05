#!/usr/bin/env python3

import numpy as np
import scipy as sp

import matplotlib.pyplot as plt
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Arial']})

import warnings
import argparse
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1] / 'src'))
import functools
import copy

from utils import SimUtils

DEFAULT_SEED = 42

def verbose(msg):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            print(f'Generating {msg}.')
            try:
                res = func(*args, **kwargs)
                print(f"Generated {msg}.")
                return res
            except:
                print(f"Failed to generate {msg}.")
                raise
        return wrapper

    return decorator


@verbose('main figure 1')
def make_fig_1m(save_dir):
    import pt_geometry_sln

    def colors(bw, n_examples):
        if bw:
            return plt.cm.Reds(np.linspace(0.8, 0.2, num=n_examples))
        else:
            return plt.cm.Blues(np.linspace(1, 0.6, num=n_examples))

    def plot_projections(res, bw=True):
        n_examples = len(res['projections'])
        fig, ax = plt.subplots(subplot_kw={'projection':'3d'},
                               figsize=(2 * n_examples, 2),
                               nrows=1,
                               ncols=n_examples)

        cols = colors(bw, n_examples)
        for i, (param, proj_act) in enumerate(res['projections'].items()):
            ax[i].plot(
                proj_act[:, 0], proj_act[:, 1], proj_act[:, 2], color=cols[i]
            )
            if not bw:
                inds = res['highlight_inds'][param]
                ax[i].plot(proj_act[inds, 0],
                           proj_act[inds, 1],
                           proj_act[inds, 2],
                           color='lightsteelblue')
            ax[i].set(xticklabels=[], yticklabels=[], zticklabels=[])
            ax[i].xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            ax[i].yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            ax[i].zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

        fig.savefig(
            str(save_dir) + f"/{'bw' if bw else 'center'}_projections.pdf",
            dpi=300,
            bbox_inches='tight',
            format='pdf',
            transparent=True
        )

    def plot_priors_and_solutions(res, bw=True):
        n_examples = len(res['solutions'])
        cols = colors(bw, n_examples)
        fig, ax = plt.subplots(figsize=(4, 2), nrows=1, ncols=2, layout='tight')

        for i, param in enumerate(res['solutions'].keys()):
            col = cols[i]
            if not bw:
                if i == n_examples - 1:
                    col = 'lightsteelblue'

            ax[0].plot(res['stims'], res['prior_draws'][param], color=col)
            ax[1].plot(res['stims'], res['solutions'][param], color=col)

        ax[1].scatter(*res['sample_points'], s=10, color='black')

        fig.savefig(
            str(save_dir) + f"/{'bw' if bw else 'center'}_prd_slns.pdf",
            dpi=300,
            bbox_inches='tight',
            format='pdf',
            transparent=True
        )

    rng = np.random.default_rng(DEFAULT_SEED)
    bw_res = pt_geometry_sln.generate_bw_het_example(rng)
    center_res = pt_geometry_sln.generate_center_het_example(rng)

    plot_projections(bw_res, bw=True)
    plot_projections(center_res, bw=False)
    plot_priors_and_solutions(bw_res, bw=True)
    plot_priors_and_solutions(center_res, bw=False)

@verbose('main figure 2')
def make_fig_2m(save_dir):
    import pt_gen_error
    import matplotlib.ticker as mticker

    colors = ['darkolivegreen', 'yellowgreen', 'paleturquoise']
    rng = np.random.default_rng(DEFAULT_SEED)
    res = pt_gen_error.run_alpha_exp(rng)

    fig, ax = plt.subplots(figsize=(4, 2), nrows=1, ncols=2, layout='tight')
    n_alphas = len(res['alphas'])
    avg_emp_error = res['empirical_error'].mean(axis=-1)

    for i in range(n_alphas):
        ax[0].errorbar(
            res['ds'],
            avg_emp_error[i],
            yerr=sp.stats.sem(res['empirical_error'][i], axis=-1),
            color=colors[i],
            label=rf'$\alpha = {res["alphas"][i]}$',
            fmt='o',
            markersize=2
        )
        ax[0].plot(res['ds'],
                   res['theoretical_error'][i],
                   linestyle='--',
                   color=colors[i])

        ax[1].errorbar(
            res['ds'],
            avg_emp_error[i] / res['bayes_error'],
            yerr=sp.stats.sem(
                res['empirical_error'][i] / res['bayes_error'][:, None],
                axis=-1
            ),
            label=rf'$\alpha = {res["alphas"][i]}$',
            color=colors[i],
            fmt='o',
            markersize=2
        )
        ax[1].plot(
            res['ds'],
            res['theoretical_error'][i] / res['bayes_error'],
            linestyle='--',
            color=colors[i]
        )

    ax[0].plot(res['ds'],
               res['bayes_error'],
               linestyle='--',
               color='blue',
               label='Bayes opt.')
    ax[0].set(yscale='log', xlabel=r'$P$', ylabel=r'$\varepsilon_g(P)$')
    ax[0].legend(frameon=False, fontsize='xx-small')
    ax[0].yaxis.set_major_formatter(mticker.ScalarFormatter())
    ax[0].yaxis.set_minor_formatter(mticker.ScalarFormatter())

    ax[1].axhline(1, color='blue', linestyle='--')
    ax[1].set(xlabel=r'$P$', ylabel='Error ratio')

    fig.savefig(
        str(save_dir) + "/gen_error_and_theory.pdf",
        dpi=300,
        bbox_inches='tight',
        format='pdf',
        transparent=True
    )

    rng = np.random.default_rng(DEFAULT_SEED)
    colors = ['darkolivegreen', 'yellowgreen', 'cadetblue']
    alphas = [0.1, 1, 5, 1000]
    teacher_ell = 1
    student_ell = 3
    ell_res = pt_gen_error.run_alpha_exp(
        rng,
        alphas=alphas,
        ell=teacher_ell,
        student_ell=student_ell,
        teacher_alpha=1000, # RBF
    )

    n_alphas = len(ell_res['alphas']) - 1
    avg_emp_error = ell_res['empirical_error'].mean(axis=-1)
    fig, ax = plt.subplots(figsize=(4, 2), nrows=1, ncols=2, layout='tight')

    for i in range(n_alphas):
        ax[0].errorbar(
            ell_res['ds'],
            avg_emp_error[i],
            yerr=sp.stats.sem(ell_res['empirical_error'][i], axis=-1),
            color=colors[i],
            label=rf'$\alpha = {ell_res["alphas"][i]}$',
            fmt='o',
            markersize=2
        )
        ax[0].plot(ell_res['ds'],
                   ell_res['theoretical_error'][i],
                   linestyle='--',
                   color=colors[i])

        ax[1].errorbar(
            ell_res['ds'],
            avg_emp_error[-1] / avg_emp_error[i],
            yerr=sp.stats.sem(avg_emp_error[-1][:, None] / \
                              ell_res['empirical_error'][i], axis=-1),
            label=rf'$\alpha = {ell_res["alphas"][i]}$',
            color=colors[i],
            fmt='o',
            markersize=2
        )
        ax[1].plot(
            ell_res['ds'],
            ell_res['theoretical_error'][-1] / ell_res['theoretical_error'][i],
            linestyle='--',
            color=colors[i]
        )

    ax[0].plot(ell_res['ds'],
               ell_res['theoretical_error'][-1],
               linestyle='--',
               color='red',
               label='homog. pop.')
    ax[0].set(yscale='log', xlabel=r'$P$', ylabel=r'$\varepsilon_g(P)$')
    ax[0].legend(frameon=False, fontsize='xx-small')
    ax[0].yaxis.set_major_formatter(mticker.ScalarFormatter())
    ax[0].yaxis.set_minor_formatter(mticker.ScalarFormatter())

    ax[1].axhline(1, color='red', linestyle='--')
    ax[1].set(xlabel=r'$P$', ylabel='Error ratio')

    fig.savefig(
        str(save_dir) + "/mismatch_ell_error.pdf",
        dpi=300,
        bbox_inches='tight',
        format='pdf',
        transparent=True
    )

    fig, ax = plt.subplots(figsize=(2, 2), nrows=1, ncols=1, layout='tight')
    ells = np.linspace(0, 4, num=100)
    for i in range(n_alphas):
        ax.plot(ells,
                sp.stats.gamma.pdf(ells,
                                   ell_res['alphas'][i],
                                   scale=3 / ell_res['alphas'][i]),
                color=colors[i])

    ax.axvline(student_ell, color='firebrick')
    ax.axvline(teacher_ell, color='black', linestyle='--')
    ax.set(xlabel=r'$\ell$', ylabel=r'$p(\ell)$', ylim=(-0.03, 0.4))

    fig.savefig(
        str(save_dir) + "/density_illustration.pdf",
        dpi=300,
        bbox_inches='tight',
        format='pdf',
        transparent=True
    )

@verbose('main figure 3')
def make_fig_3m(save_dir):
    import pt_adaptation
    import pt_gen_error

    adapt_color = 'goldenrod'
    fixed_color = 'brown'
    alpha_colors = ['peachpuff', 'lightsalmon', 'tomato']

    rng = np.random.default_rng(DEFAULT_SEED)
    res = pt_adaptation.run_adaptation_exp(rng)

    adapted_avg_error = np.nanmean(res['adapt_eg'], axis=-1)
    fixed_avg_error = np.nanmean(res['fixed_eg'], axis=-1)

    fig, ax = plt.subplots(figsize=(8, 2), nrows=1, ncols=4, layout='tight')
    # generalization error panel
    ax[0].errorbar(res['ds'],
                   adapted_avg_error,
                   yerr=sp.stats.sem(
                       res['adapt_eg'], axis=-1, nan_policy='omit'
                   ),
                   color=adapt_color,
                   marker='.',
                   label='adapt')
    ax[0].errorbar(res['ds'],
                   fixed_avg_error,
                   yerr=sp.stats.sem(
                       res['fixed_eg'], axis=-1, nan_policy='omit'
                   ),
                   color=fixed_color,
                   marker='.',
                   label='fixed')
    ax[0].set(ylabel=r'$\varepsilon_g(P)$',
              xlabel=r'dataset size, $P$')
    ax[0].legend(frameon=False, fontsize='xx-small')

    held_adapted_avg_error = np.nanmean(res['adapt_held_eg'], axis=-1)
    held_fixed_avg_error = np.nanmean(res['fixed_held_eg'], axis=-1)

    # held-out error panel
    ax[1].errorbar(res['ds'],
                   held_adapted_avg_error,
                   yerr=sp.stats.sem(
                       res['adapt_held_eg'], axis=-1, nan_policy='omit'
                   ),
                   color=adapt_color,
                   marker='.',
                   label='adapt')
    ax[1].errorbar(res['ds'],
                   held_fixed_avg_error,
                   yerr=sp.stats.sem(
                       res['fixed_held_eg'], axis=-1, nan_policy='omit'
                   ),
                   color=fixed_color,
                   marker='.',
                   label='fixed')
    ax[1].set(ylim=(0.5, 1),
              ylabel=r'$\varepsilon_g(P)$',
              xlabel=r'adaptation dataset size, $P$')

    # optimal alpha panel
    def strip_outliers(data):
        q1 = np.nanpercentile(data, 25, axis=-1, keepdims=True)
        q3 = np.nanpercentile(data, 75, axis=-1, keepdims=True)
        iqr = q3 - q1

        lb = q1 - 1.5 * iqr
        ub = q3 + 1.5 * iqr

        mask = (data < lb) | (data > ub)
        return np.where(mask, np.nan, data)

    adapted_alphas_no = strip_outliers(res['adapted_alphas'])
    adapted_alphas_no_mean = np.nanmean(adapted_alphas_no, axis=-1)
    ax[2].errorbar(res['ds'],
                   adapted_alphas_no_mean,
                   yerr=sp.stats.sem(
                       adapted_alphas_no, axis=-1, nan_policy='omit'
                   ),
                   marker='.',
                   color=alpha_colors[-1])
    ax[2].axhline(res['teacher_alpha'], color='magenta', linestyle='--')
    ax[2].set(yscale='log',
              ylim=(1e-1, 110),
              ylabel=r"opt. het. $\alpha^*$",
              xlabel=r'dataset size, $P$')

    # now that alphas are averaged, overlay theory
    adapt_theory_pred = np.zeros(len(res['ds']))
    fixed_theory_pred = np.zeros_like(adapt_theory_pred)
    ho_adapt_theory_pred = np.zeros_like(adapt_theory_pred)
    ho_fixed_theory_pred = np.zeros_like(adapt_theory_pred)
    theory_su = SimUtils(rng, x_min=0, x_max=10, res=200, n_repeats=500)
    teacher = pt_gen_error.build_model(
        res['teacher_alpha'], res['teacher_ell'], res['sigma_t'], theory_su
    )
    fixed_student = pt_gen_error.build_model(
        res['student_alpha_f'], res['student_ell'], res['sigma_s'], theory_su,
    )
    held_out_ds_size = 50
    for i, p in enumerate(res['ds']):
        student = pt_gen_error.build_model(adapted_alphas_no_mean[i],
                                           res['student_ell'],
                                           res['sigma_s'],
                                           theory_su)
        adapt_theory_pred[i] = pt_gen_error.theory_error(
            teacher, student, p, theory_su
        )
        fixed_theory_pred[i] = pt_gen_error.theory_error(
            teacher, fixed_student, p, theory_su
        )
        ho_adapt_theory_pred[i] = pt_gen_error.theory_error(
            teacher, student, held_out_ds_size, theory_su
        )
        ho_fixed_theory_pred[i] = pt_gen_error.theory_error(
            teacher, fixed_student, held_out_ds_size, theory_su
        )

    ax[0].scatter(res['ds'], adapt_theory_pred, marker="x", color=adapt_color)
    ax[0].scatter(res['ds'], fixed_theory_pred, marker="x", color=fixed_color)

    ax[1].scatter(
        res['ds'], ho_adapt_theory_pred, marker="x", color=adapt_color
    )
    ax[1].scatter(
        res['ds'], ho_fixed_theory_pred, marker="x", color=fixed_color
    )

    # kernel panel
    x_min = 0
    x_max = 0.1
    stims = np.linspace(x_min, x_max, 200)
    diffs = stims[None, :] - stims[:, None]
    for i, p in enumerate([1, 6, 8]):
        alp = adapted_alphas_no_mean[p]
        kern = np.power(
            1 + diffs ** 2 / (2 * alp * (res['student_ell'] ** 2)), -alp
        )
        ax[3].plot(kern[0], color=alpha_colors[i])

    t_kern = np.power(
        1 + diffs ** 2 / (2 * res['teacher_alpha'] * (res['teacher_ell'] ** 2)),
        -res['teacher_alpha']
    )
    ax[3].plot(t_kern[0], color='magenta', linestyle='--')
    ax[3].set(
        xlabel=r"distance, $|x - x'|$",
        ylabel=r"$k(|x - x'|)$"
    )

    fig.savefig(
        str(save_dir) + "/adaptation_experiment_with_theory.pdf",
        dpi=300,
        bbox_inches='tight',
        format='pdf',
        transparent=True
    )

def _plot_error_and_lambda(save_dir,
                           adaptive_noise=False,
                           sigma_t=0.8,
                           sigma_s=0.8,
                           be=None):
    import pt_reward
    import pt_gen_error

    rng = np.random.default_rng(DEFAULT_SEED)
    res, teacher = pt_reward.run_reward_exp(rng,
                                            sigma_t=sigma_t,
                                            sigma_s=sigma_s,
                                            adaptive_noise=adaptive_noise)

    log_mean_liks = sp.special.logsumexp(res['mls'], axis=-1) - \
        np.log(res['mls'][0].shape[-1])
    lik = np.exp(
        log_mean_liks.T - sp.special.logsumexp(log_mean_liks, axis=-1)
    )

    fig, ax = plt.subplots(figsize=(4, 2), nrows=1, ncols=2, layout='tight')
    lambs = np.linspace(0, 1, num=100)
    colors = ['lightsteelblue', 'cornflowerblue', 'royalblue', 'blue']
    ax[0].set_prop_cycle(color=colors)
    ax[0].plot(
        lambs[:, None],
        lik
    )
    ax[0].set(xlabel=r'$\lambda$', ylabel=r'$p(y|x,\lambda)$')
    ax[0].axvline(res['t_lamb'], color='olive', linestyle='--')

    ax[1].errorbar(
        res['ds'],
        res['eg_adapt'].mean(axis=-1),
        yerr=sp.stats.sem(res['eg_adapt'], axis=-1),
        label='adapt',
        color='royalblue',
        marker='.',
    )
    ax[1].errorbar(
        res['ds'],
        res['eg_fixed'].mean(axis=-1),
        yerr=sp.stats.sem(res['eg_fixed'], axis=-1),
        label='fixed',
        color='grey',
        marker='.',
    )

    theory_su = SimUtils(rng, x_min=-1, x_max=1, res=100, n_repeats=50)
    base_kern = pt_reward.build_kernel(res['mu_r'],
                                       res['sigma_r'],
                                       res['bw_rbf'],
                                       res['bw_env'],
                                       sigma_t,
                                       theory_su)
    adapt_theory_pred = np.zeros(len(res['ds']))
    fixed_theory_pred = np.zeros_like(adapt_theory_pred)

    # reformat kernel object for compatability with pt_gen_error module
    teacher = copy.deepcopy(base_kern)
    teacher['kernel'] = (1 - res['t_lamb']) * teacher['rbf'] + \
        res['t_lamb'] * teacher['env']
    evals, evecs = np.linalg.eigh(teacher['kernel'])
    teacher['evals'] = evals
    teacher['evecs'] = evecs

    student = copy.deepcopy(base_kern)
    student['obs_noise'] = sigma_s

    fixed_student = copy.deepcopy(student)
    for i, p in enumerate(res['ds']):
        if adaptive_noise:
            student['obs_noise'] = res['nvs'][i]
            fixed_student['obs_noise'] = res['nvs'][0]

        s_lamb = lambs[np.argmax(lik[:, i])]
        student['kernel'] = (1 - s_lamb) * student['rbf'] + \
            s_lamb * student['env']
        # dont need eigendecomposition of student for this
        adapt_theory_pred[i] = pt_gen_error.theory_error(
            teacher, student, p, theory_su
        )

        fixed_student['kernel'] = student['rbf']
        fixed_theory_pred[i] = pt_gen_error.theory_error(
            teacher, fixed_student, p, theory_su
        )

    ax[1].scatter(res['ds'], adapt_theory_pred, marker='x', color='royalblue')
    ax[1].scatter(res['ds'], fixed_theory_pred, marker='x', color='gray')

    if be is not None:
        ax[1].plot(res['ds'], be, color=colors[-2], linestyle='--')

    ax[1].set(xlabel=r"dataset size, $P$", ylabel=r'$\varepsilon_g(P)$')
    rel = 'under' if sigma_s < sigma_t else 'over' if sigma_s > sigma_t else 'matched'
    fname = f"/reward_experiment_with_theory_{'an' if adaptive_noise else 'fn'}_{rel}.pdf"
    fig.savefig(
        str(save_dir) + fname,
        dpi=300,
        bbox_inches='tight',
        format='pdf',
        transparent=True
    )

    return res['eg_adapt'].mean(axis=-1)

@verbose('main figure 4')
def make_fig_4m(save_dir):
    res = _plot_error_and_lambda(save_dir, adaptive_noise=False)
    _plot_error_and_lambda(save_dir, adaptive_noise=True, be=res)

@verbose('supplemental figure 1')
def make_fig_1s(save_dir):
    import pt_sigmoid_tuning

    rng = np.random.default_rng(DEFAULT_SEED)
    ret = pt_sigmoid_tuning.sigmoid_population(rng)

    n_sub_select = 13
    sub_select_neurons = np.linspace(200, 800, num=n_sub_select).astype(int)

    fig, ax = plt.subplots(figsize=(10, 2), nrows=1, ncols=5, layout='tight')
    homogeneous_color = 'purple'
    gain_color = 'orchid'
    bw_color = 'mediumslateblue'

    ax[0].plot(np.outer(ret['stims'], np.ones(n_sub_select)),
               ret['tuning_curves']['homogeneous'][:, sub_select_neurons],
               color=homogeneous_color)

    ax[1].plot(np.outer(ret['stims'], np.ones(n_sub_select)),
               ret['tuning_curves']['gain'][:, sub_select_neurons],
               color=gain_color)

    ax[2].plot(np.outer(ret['stims'], np.ones(n_sub_select)),
               ret['tuning_curves']['bws'][:, sub_select_neurons],
               color=bw_color)

    for k, c in [('homogeneous', homogeneous_color), \
                 ('gain', gain_color), \
                 ('bws', bw_color)]:
        ax[3].plot(ret['stims'], ret['prior_draws'][k], color=c)
        ax[4].plot(ret['stims'], ret['solutions'][k], color=c)
        ax[4].scatter(*ret['sample_points'], color='black', s=10)

    fig.savefig(
        str(save_dir) + "/sigmoid_tuning.pdf",
        dpi=300,
        bbox_inches='tight',
        format='pdf',
        transparent=True
    )

@verbose('supplemental figure 2')
def make_fig_2s(save_dir):
    rng = np.random.default_rng(DEFAULT_SEED)

    n_neurons = 1_000_000
    x_min = -1
    x_max = 1
    L = 4
    res = 100
    n_bws = 5
    bws = np.linspace(0.1, 0.5, num=n_bws)
    weights = np.ones_like(bws) / n_bws

    stims = np.linspace(x_min, x_max, num=res)
    stim_diff = stims[:, None] - stims[None, :]

    # empirical population
    centers = np.random.uniform(low=-L, high=L, size=n_neurons)
    pbws = rng.choice(bws, size=n_neurons, p=weights)
    act = np.exp(
        -((stims[:, None] - centers[None, :]) / \
          (np.sqrt(2) * pbws[None, :])) ** 2
    )
    emp_kern = act @ act.T / n_neurons * 2 * L

    # analytic kernel
    ana_kern = (weights * np.sqrt(np.pi) * bws * \
        np.exp(-(stim_diff[..., None] / (2 * bws)) ** 2)).sum(axis=-1)

    fig, ax = plt.subplots(figsize=(9, 2), nrows=1, ncols=4, layout='tight')

    c = ax[0].imshow(ana_kern, aspect='auto', cmap='twilight', origin='lower')
    ax[0].set(title=r'analytic, $K_A$')
    fig.colorbar(c, ax=ax[0])

    c = ax[1].imshow(emp_kern, aspect='auto', cmap='twilight', origin='lower')
    ax[1].set(title=r'empirical, $K_E$')
    fig.colorbar(c, ax=ax[1])

    c = ax[2].imshow(
        ana_kern - emp_kern, aspect='auto', cmap='cividis', origin='lower'
    )
    ax[2].set(title='$K_A - K_E$')
    fig.colorbar(c, ax=ax[2])

    ax[3].plot(emp_kern[0], label='empirical', color='pink')
    ax[3].plot(ana_kern[0], label='theory', linestyle='--', color='grey')
    ax[3].legend(frameon=False)
    ax[3].set(xlabel=r"distance, $|x - x'|$", ylabel=r"$K(|x-x'|)$")

    fig.savefig(
        str(save_dir) + "/weighted_mixture_kernel.pdf",
        dpi=300,
        bbox_inches='tight',
        format='pdf',
        transparent=True
    )

@verbose('supplemental figure 3')
def make_fig_3s(save_dir):
    res = _plot_error_and_lambda(save_dir, adaptive_noise=False)
    _plot_error_and_lambda(
        save_dir, adaptive_noise=False, sigma_s=0.1, sigma_t=0.8, be=res
    )
    _plot_error_and_lambda(
        save_dir, adaptive_noise=False, sigma_s=2.6, sigma_t=0.8, be=res
    )

@verbose('supplemental figure 4')
def make_fig_4s(save_dir):
    import pt_reward

    rng = np.random.default_rng(DEFAULT_SEED)

    mu_r = 0.5
    sigma_r = 0.1
    bw_rbf = 0.1
    bw_env = 0.1
    obs_noise = 1
    su = SimUtils(rng, x_min=-1, x_max=1, res=100)

    base_kernels = pt_reward.build_kernel(
        mu_r, sigma_r, bw_rbf, bw_env, obs_noise, su
    )
    lambdas = np.linspace(0, 1, num=5)
    colors = plt.cm.Reds(np.linspace(0.8, 0.2, num=len(lambdas)))

    fig, ax = plt.subplots(figsize=(2 * (len(lambdas) + 2), 2),
                           nrows=1,
                           ncols=len(lambdas) + 2,
                           layout='tight')

    for i, lamb in enumerate(lambdas):
        kern = lamb * base_kernels['env'] + (1 - lamb) * base_kernels['rbf']
        c = ax[i].imshow(kern, aspect='auto', cmap='twilight', origin='lower')
        ax[i].set(title=rf'$\lambda = {lamb}$')
        fig.colorbar(c, ax=ax[i])
        evals, _ = np.linalg.eigh(kern + np.eye(kern.shape[0]) * obs_noise)
        evals.sort()
        ax[-2].plot(evals[::-1] / np.max(evals),
                    label=rf'$\lambda = {lamb}$',
                    color=colors[i])

    ax[-2].set(yscale='log', xscale='symlog', xlim=(0, 100), ylim=(10e-3, 1))
    ax[-2].legend(frameon=False, fontsize='xx-small')

    lambds = np.linspace(0, 1, num=100)
    complexities = np.zeros_like(lambds)
    for i, lamb in enumerate(lambds):
        kern = lamb * base_kernels['env'] + (1 - lamb) * base_kernels['rbf']

        complexities[i] = np.log(
            np.linalg.det(kern + np.eye(kern.shape[0]) * obs_noise)
        )

    ax[-2].set(xlabel='k', ylabel=r'$\lambda_k$', title='Spectra')
    ax[-1].plot(lambds, complexities, color='orange')
    ax[-1].set(title='Complexity',
               xlabel=r'$\lambda$',
               ylabel=r'$\log|K_\lambda + \sigma^2 I|$')

    fig.savefig(
        str(save_dir) + "/kernel_complexity.pdf",
        dpi=300,
        bbox_inches='tight',
        format='pdf',
        transparent=True
    )

@verbose('supplemental figure 5')
def make_fig_5s(save_dir):
    import pt_small_population_appx

    rng = np.random.default_rng(DEFAULT_SEED)

    def plot_kernels_and_metrics(res, rbf=True):
        n_pop_sizes = len(res['pop_sizes'])

        # plot analytic and empirical kernels
        fig, ax = plt.subplots(
            figsize=(2 * (n_pop_sizes + 1), 2),
            nrows=1,
            ncols=n_pop_sizes + 1,
            layout='tight'
        )

        avg_emp_kerns = res['empirical_kernels'].mean(axis=-1)
        c = ax[0].imshow(res['analytic_kernel'],
                         aspect='auto',
                         cmap='twilight',
                         origin='lower')
        fig.colorbar(c, ax=ax[0])
        ax[0].set(title=r'$K_A$')
        for i in range(n_pop_sizes):
            c = ax[i + 1].imshow(avg_emp_kerns[i],
                                 aspect='auto',
                                 cmap='twilight',
                                 origin='lower')
            fig.colorbar(c, ax=ax[i + 1])
            ax[i + 1].set(
                title=rf'$K_E, N ~ 10^{int(np.log10(res["pop_sizes"][i]))}$'
            )

        fig.savefig(
            str(save_dir) + \
                f"/kernel_appx_small_n_{'rbf' if rbf else 'rq'}.pdf",
            dpi=300,
            bbox_inches='tight',
            format='pdf',
            transparent=True
        )

        # plot metrics (rel fro error, cka, gen error v. theory)

        # first, ||K_A - K_E||/||K_A|| (normalizing by K_A magnitude)
        fig, ax = plt.subplots(figsize=(6, 2), nrows=1, ncols=3, layout='tight')
        rel_error = np.linalg.norm(
            res['analytic_kernel'][None, :, :, None] - res['empirical_kernels'],
            axis=(1, 2)
        ) / np.linalg.norm(
            res['analytic_kernel'][None, :, :, None], axis=(1, 2)
        )
        avg_rel_error = rel_error.mean(axis=-1)
        ax[0].loglog(res['pop_sizes'],
                     avg_rel_error,
                     marker='o',
                     markerfacecolor='white',
                     color='thistle')
        ax[0].set(xlabel='Population size, $N$', ylabel='Relative error')

        # next, cka (centralized kernel alignment)
        def cka(res):
            KA = res['analytic_kernel']
            KE = res['empirical_kernels']

            n = KA.shape[0]
            H = np.eye(n) - 1 / n

            KA_t = KA[None, None, ...]
            KE_t = KE.transpose(0, 3, 1, 2)

            KAc = H @ KA_t @ H
            KEc = H @ KE_t @ H
            def hsic(k, l):
                return 1 / ((n - 1) ** 2) * np.sum(k * l, axis=(2, 3))

            return hsic(KAc, KEc) / np.sqrt(hsic(KAc, KAc) * hsic(KEc, KEc))

        mean_ckas = cka(res).mean(axis=-1)
        ax[1].plot(res['pop_sizes'],
                   mean_ckas,
                   marker='o',
                   markerfacecolor='white',
                   color='skyblue')
        ax[1].set(xscale='log',
                  xlabel='Population size, $N$',
                  ylabel='CKA($K_E$, $K_A$)',
                  ylim=(0, 1.1))
        ax[1].axhline(1, color='grey', linestyle='--')

        # finally, generalization error
        import pt_gen_error

        teacher = dict()
        teacher['kernel'] = res['analytic_kernel']
        evals, evecs = np.linalg.eigh(teacher['kernel'])
        teacher['evals'] = evals
        teacher['evecs'] = evecs
        teacher['obs_noise'] = 1.0

        student = dict()
        student['obs_noise'] = 1.0

        ds = np.logspace(1, 2, num=10).astype(int)
        rng = np.random.default_rng(DEFAULT_SEED)
        su = SimUtils(rng, x_min=-1, x_max=1, res=100, n_repeats=500)

        emp_error = np.zeros((len(ds), len(res['pop_sizes']), 500))
        t_error = np.zeros(len(ds))

        for i in range(len(res['pop_sizes'])):
            student['kernel'] = res['empirical_kernels'][i][:, :, 0]
            for j, d in enumerate(ds):
                t_error[j] = pt_gen_error.theory_error(teacher, student, d, su)
                emp_error[j, i] = pt_gen_error.empirical_error(
                    teacher, student, d, su
                )

        ax[2].set_prop_cycle(
            color=plt.cm.Greens(np.linspace(0.2, 1, num=n_pop_sizes))
        )
        labels = [f"$N ~ 10^{int(np.log10(res['pop_sizes'][i]))}$" \
                  for i in range(n_pop_sizes)]
        ax[2].plot(ds,
                   emp_error.mean(axis=-1),
                   marker='o',
                   markerfacecolor='white',
                   label=labels)
        ax[2].plot(ds, t_error, linestyle='--', color='black')
        ax[2].set(yscale='log',
                  ylabel=r'$\varepsilon_g(P)$',
                  xlabel='dataset size, $P$')
        ax[2].legend(frameon=False, fontsize='xx-small')

        fig.savefig(
            str(save_dir) + \
                f"/kernel_metrics_appx_small_n_{'rbf' if rbf else 'rq'}.pdf",
            dpi=300,
            bbox_inches='tight',
            format='pdf',
            transparent=True
        )

    result_rbf = pt_small_population_appx.run_pop_size_approximation_exp(
        rng, n_repeats=5
    )
    plot_kernels_and_metrics(result_rbf)

    # large matrices in here collide with computation that follows,
    # so explicitly clean things up to make everything run a bit faster
    result_rbf.clear()
    del result_rbf

    result_rq = pt_small_population_appx.run_pop_size_approximation_exp(
        rng, analytic_kernel='rq', alpha=2, ell=0.1
    )
    plot_kernels_and_metrics(result_rq, rbf=False)

def parse(args):
    if args is None:
        return [make_fig_1m, make_fig_2m, make_fig_3m, make_fig_4m, \
                make_fig_1s, make_fig_2s, make_fig_3s, make_fig_4s, make_fig_5s]

    main_function_map = {
        '1' : make_fig_1m,
        '2' : make_fig_2m,
        '3' : make_fig_3m,
        '4' : make_fig_4m
    }
    supp_function_map = {
        '1' : make_fig_1s,
        '2' : make_fig_2s,
        '3' : make_fig_3s,
        '4' : make_fig_4s,
        '5' : make_fig_5s,
    }

    sm_figs = args.split('+')
    figs_to_make = []
    for opt in sm_figs:
        main = (opt[0] == 'm')
        if not main:
            assert opt[0] == 's', "Fig indicators must be \'m\' or \'s\'"

        if main:
            dict_to_pull = main_function_map
        else:
            dict_to_pull = supp_function_map

        fs = opt[1:].split(',')
        for f in fs:
            if '-' in f:
                rng = np.arange(int(f[0]), int(f[-1]) + 1).astype(str)
            else:
                rng = [f]

            for r in rng:
                assert r in dict_to_pull, f"Invalid figure number: {r}"
                figs_to_make.append(dict_to_pull[r])

    return figs_to_make

def main():
    parser = argparse.ArgumentParser(
        prog='fig_gen',
        description="""
            Generate main and supplemental figure panels for the paper \"Neural
            population tuning statistics as priors for multitask
            generalization\" (Cohen & Drugowitsch, NeurIPS 2026). Note, this
            program only generates all figure panels as PDFs. Additional
            formatting in illustrator is used to make the figures that appear
            in the camera-ready version of this paper.
        """
    )

    parser.add_argument('-f',
                        '--figs',
                        help="""
                        Indicate fig panels you would like to generate. Leaving
                        this option blank generates all paper fig panels by
                        default. Otherwise, use the syntax: -f m1-3+s1,3-5.
                        m indicates main figures, s indicates supplemental
                        figures (separated by the + character). m1-3 generates
                        main figures 1-3. s1,3-5 generates supplemental figures
                        1 and, additionally, supplemental figures 3-5. You
                        can also just run -f m1-4 to generate all main figures.
                        The main and supplemental figures can be in any order
                        (for example, -f s1-5+m1-4 is valid).
                        """,
                        default=None)

    args = parser.parse_args()
    figs_to_make = parse(args.figs)

    save_dir = Path.cwd() / 'files'
    save_dir.mkdir(parents=True, exist_ok=True)

    for f in figs_to_make:
        f(save_dir)

if __name__ == '__main__':
    main()
