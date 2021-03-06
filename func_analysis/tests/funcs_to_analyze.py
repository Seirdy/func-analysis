# -*- coding: utf-8 -*-

"""The plain Python functions to pass to AnalyzedFunc."""
from numbers import Real

import mpmath as mp

from func_analysis.tests.call_counting import ForbidCalling


@ForbidCalling
def trig_func(x_val: mp.mpf) -> mp.mpf:
    """Define a test function requiring high precision for analysis.

    cos(x^2)-sin(x)+x/68
    """
    return mp.cos(x_val ** 2) - mp.sin(x_val) + (x_val / 68)


@ForbidCalling
def sec_der(x_val: Real) -> mp.mpf:
    """Define the actual second derivative."""
    x_squared = mp.power(x_val, 2)
    return (
        mp.cos(x_val)
        + (-4 * x_squared) * mp.cos(x_squared)
        + -2 * mp.sin(x_squared)
        + mp.sin(x_val)
    )


@ForbidCalling
def parab_func(x_val: Real) -> mp.mpf:
    """Define a simple parabola.

    It is concave and symmetric about the y-axis.
    """
    return mp.power(x_val, 2) - 4


@ForbidCalling
def inc_dec_func(x_val):
    """Define a function to test increasing/decreasing intervals.

    ln(x^2)/x subtly switches from decreasing to increasing at x=-e.
    It is concave across (-inf, 0) and convex across (0, inf).
    """
    return mp.fdiv(mp.log(mp.power(x_val, 2)), x_val)
