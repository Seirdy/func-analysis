# -*- coding: utf-8 -*-
"""Utility functions to typecheck function analysis."""
from numbers import Real
from typing import Iterable, Sequence

import numpy as np

from func_analysis.custom_types import Interval


def typecheck_multi(item_to_check, *args) -> bool:
    """Check if item_to_check is instance of anything in *args."""
    return any(
        isinstance(item_to_check, allowed_type) for allowed_type in args
    )


def typecheck_number(number_to_check):
    """Assert that item is a Real."""
    assert typecheck_multi(number_to_check, Real, float, np.float64, int)


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
    assert typecheck_multi(points, np.ndarray, Sequence)
    typecheck_iterable(points, Real)


def typecheck_intervals(intervals: Sequence[Interval]):
    """Typecheck of all functions that return a sequence of intervals."""
    assert isinstance(intervals, Sequence)
    for interval in intervals:
        assert isinstance(interval, Interval)
        typecheck_number(interval.start)
        typecheck_number(interval.stop)
