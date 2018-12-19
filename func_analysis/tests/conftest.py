"""Fixtures to represent sample AnalyzedFunc instances."""
from numbers import Real

import mpmath as mp
import pytest

from .helpers import AnalyzedFuncSavedInstances, CountCalls


@CountCalls
def trig_func(x_val: mp.mpf) -> mp.mpf:
    """Define a test function requiring high precision for analysis.

    cos(x^2)-sin(x)+x/68
    """
    return mp.cos(x_val ** 2) - mp.sin(x_val) + (x_val / 68)


@pytest.fixture
def analyzed_trig_func():
    """Fixture for an AnalyzedFunc describing trig_func."""
    analyzed_func = AnalyzedFuncSavedInstances(
        func=trig_func,
        x_range=(-47.05, -46.3499),
        zeros_wanted=21,
        crits_wanted=21,
        known_zeros=[-47.038_289_673_236_127, -46.406_755_885_040_056],
    )
    return analyzed_func


@CountCalls
def sec_der(x_val: Real) -> mp.mpf:
    """Define the actual second derivative."""
    return (
        mp.cos(x_val)
        + (-4 * (mp.power(x_val, 2))) * mp.cos(mp.power(x_val, 2))
        + -2 * mp.sin(mp.power(x_val, 2))
        + mp.sin(x_val)
    )


@pytest.fixture()
def fp2_zeros():
    """Fixture for an AnalyzedFuncSavedInstances describing sec_der."""
    return AnalyzedFuncSavedInstances(
        func=sec_der, x_range=(-47.05, -46.35), zeros_wanted=21
    )


@CountCalls
def parab_func(x_val: Real) -> mp.mpf:
    """Define a simple parabola.

    It is concave and symmetric about the y-axis.
    """
    return mp.power(x_val, 2) - 4


@pytest.fixture
def analyzed_parab():
    """Fixture for an AnalyzedFuncSavedInstances describing parab_func."""
    return AnalyzedFuncSavedInstances(
        func=parab_func, x_range=(-8, 8), zeros_wanted=2
    )


@CountCalls
def inc_dec_func(x_val):
    """Define a function to test increasing/decreasing intervals.

    ln(x^2)/x subtly switches from decreasing to increasing at x=-e.
    It is concave across (-inf, 0) and convex across (0, inf).
    """
    return mp.fdiv(mp.log(mp.power(x_val, 2)), x_val)


@pytest.fixture()
def analyzed_incdecfunc():
    """Fixture for an AnalyzedFuncSavedInstances describing inc_dec_func."""
    return AnalyzedFuncSavedInstances(
        func=inc_dec_func, x_range=(-3, -0.001), crits_wanted=1, zeros_wanted=1
    )
