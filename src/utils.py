#!/usr/bin/env python3

import numpy as np
import scipy as sp

class SimUtils:
    def __init__(self, rng, x_min=-1, x_max=1, res=100, n_repeats=500):
        self.rng = rng
        self.x_min = x_min
        self.x_max = x_max
        self.res = res
        self.n_repeats = n_repeats

def get_x_samples(p, su):
    return su.rng.choice(np.arange(su.res), size=(p, su.n_repeats))

def sample_teacher(teacher_kernel, su):
    teacher_kernel += 1e-8 * np.eye(teacher_kernel.shape[0])

    return su.rng.multivariate_normal(
        np.zeros(teacher_kernel.shape[0]), teacher_kernel, size=su.n_repeats
    )

