"""Utility functions to typecheck function analysis."""

from typing import Iterable, List

import mpmath as mp
import numpy as np


def typecheck_multi(item_to_check, *args) -> bool:
    """Check if item_to_check is instance of anything in *args."""
    return any(
        isinstance(item_to_check, allowed_type) for allowed_type in args
    )


def typecheck_number(number_to_check):
    """Assert that item is a Real."""
    assert typecheck_multi(number_to_check, mp.mpf, float, np.float64, int)


def typecheck_iterable(items_to_check: Iterable, *args):
    """Typecheck items_to_check in an Iterable."""
    assert all(
        typecheck_multi(each_item, args) for each_item in items_to_check
    )


def typecheck_zcp(points):
    """Typecheck functions returning arrays of points.

    Such functions include zeros(), crits, pois(),
    relative_maxima(), relative_minima().
    """
    assert isinstance(points, np.ndarray)
    typecheck_iterable(points, mp.mpf)


def typecheck_intervals(intervals):
    """Typecheck of all functions with return type List[Interval]."""
    assert isinstance(intervals, List)
    for interval in intervals:
        assert isinstance(interval, tuple)
        typecheck_number(interval[0])
        typecheck_number(interval[1])