"""Kallinger et al. 2014 granulation scaling relations.

Used by the correlated-inference likelihood (amplitude/timescales computed
from the sampled numax and a scatter factor) and by the radius pipeline.
Plain arithmetic, so they work with numpy arrays and inside jax-jitted code.
"""


def gran_a(numax, ascat):
    return 3382 * numax ** (-0.609) * ascat


def gran_b(numax, bscat):
    return 0.317 * numax ** (0.970) * bscat


def gran_d(numax, dscat):
    return 0.948 * numax ** (0.992) * dscat
