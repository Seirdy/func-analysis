"""Fixtures to represent sample AnalyzedFunc instances."""
from __future__ import annotations

from functools import update_wrapper
from numbers import Real
from typing import List

import mpmath as mp
import pytest

from .._analysis_classes import AnalyzedFunc


class CountCalls:
    """Class decorator for tracking state."""

    # pylint: disable=undefined-variable
    functions: List[CountCalls] = []  # NOQA: F821
    # pylint: enable=undefined-variable

    def __init__(self, func):
        """Initialize the object."""
        update_wrapper(self, func)
        self.func = func
        CountCalls.functions.append(self)
        self.call_count = 0

    def __call__(self, *args):
        """Increment counter each time func is called."""
        self.call_count += 1
        return self.func(*args)


@CountCalls
def trig_func(x_val: mp.mpf) -> mp.mpf:
    """Define a test function requiring high precision for analysis.

    cos(x^2)-sin(x)+x/68
    """
    return mp.cos(x_val ** 2) - mp.sin(x_val) + (x_val / 68)


@pytest.fixture
def analyzed_trig_func():
    """Fixture for an AnalyzedFunc describing trig_func."""
    return AnalyzedFunc(
        func=trig_func,
        x_range=(-47.05, -46.3499),
        zeros_wanted=21,
        crits_wanted=21,
        known_zeros=[-47.038_289_673_236_127, -46.406_755_885_040_056],
    )


@CountCalls
def parab_func(x_val: Real) -> mp.mpf:
    """Define a simple parabola.

    It is concave and symmetric about the y-axis.
    """
    return mp.power(x_val, 2) - 4


@pytest.fixture
def analyzed_parab():
    """Fixture for an AnalyzedFunc describing parab_func."""
    return AnalyzedFunc(func=parab_func, x_range=(-8, 8), zeros_wanted=2)


@CountCalls
def inc_dec_func(x_val):
    """Define a function to test increasing/decreasing intervals.

    ln(x^2)/x subtly switches from decreasing to increasing at x=-e.
    It is concave across (-inf, 0) and convex across (0, inf).
    """
    return mp.fdiv(mp.log(mp.power(x_val, 2)), x_val)


@pytest.fixture()
def analyzed_incdecfunc():
    """Fixture for an AnalyzedFunc describing inc_dec_func."""
    return AnalyzedFunc(
        func=inc_dec_func, x_range=(-3, -0.001), crits_wanted=1, zeros_wanted=1
    )
