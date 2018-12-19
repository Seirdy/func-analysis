"""Helper functions for testing AnalyzedFunc."""
from __future__ import annotations

from functools import update_wrapper
from typing import Iterable, List, Tuple

import mpmath as mp
import numpy as np

from .._analysis_classes import AnalyzedFunc


# pylint: disable=undefined-variable
class AnalyzedFuncSavedInstances(AnalyzedFunc):
    """Saves all instances of an object."""

    instances: List[AnalyzedFuncSavedInstances] = []  # noqa: F821

    def __init__(self, *args, **kwargs):
        """Intialize the object."""
        super().__init__(*args, **kwargs)
        self.instances.append(self)
        self.func_name = self._func_plotted.__name__

    @property
    def counts(self):
        """Count unique calls to self.func."""
        return len(self.plotted_points)

    @property
    def deduped_counts(self):
        """Remove duplicates from self.counts."""
        return len(np.unique([coord[0] for coord in self.plotted_points]))


# pylint: enable=undefined-variable


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


def total_counts_pre_analysis() -> int:
    """Total calls for analyzed functions."""
    counts = (counted_func.call_count for counted_func in CountCalls.functions)
    return sum(counts)


def mpf_assert_allclose(actual, desired, atol=1e-3):
    """Assert that the two arrays are close enough.

    Similar to numpy.testing.assert_allclose().
    """
    assert np.amax(np.abs(np.subtract(actual, desired))) < atol


def assert_output_lessthan(func, x_vals, max_y):
    """Assert that func(x) < max_y for all x_vals."""
    y_vals = func(x_vals)
    assert np.amax(np.abs(y_vals)) < max_y


def typecheck_multi(item, *args) -> bool:
    """Check if item is instance of anything in *args."""
    return any(isinstance(item, allowed_type) for allowed_type in args)


def typecheck_number(num):
    """Assert that item is a Real."""
    assert typecheck_multi(num, mp.mpf, float, np.float64, int)


def typecheck_iterable(items: Iterable, *args):
    """Typecheck items in an Iterable.

    Assert each item in items is an instance of something in *args.
    Since all items in numpy arrays share the same type, only the first
    item needs to be checked if items is an array.
    """
    if isinstance(items, np.ndarray):
        assert typecheck_multi(items[0], args)
    assert all(typecheck_multi(item, args) for item in items)


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


def workout_analyzed_func(
    analyzed_func: AnalyzedFuncSavedInstances
) -> Tuple[dict, dict]:
    """Track function calls throughout function analysis."""
    sequential_counts = {}
    sequential_deduped_counts = {}

    def logged_calculation(arg, key):
        assert arg is not None
        sequential_counts[key] = analyzed_func.counts
        sequential_deduped_counts[key] = analyzed_func.deduped_counts

    logged_calculation(analyzed_func.zeros, "zeros")
    logged_calculation(analyzed_func.absolute_minimum(), "absmin")
    logged_calculation(analyzed_func.convex(), "convex")
    saved_coords = tuple(analyzed_func.plotted_points)
    saved_points = [coord[0] for coord in saved_coords]
    logged_calculation(analyzed_func.func(saved_points), "dupe")
    logged_calculation(
        analyzed_func.rooted_second_derivative().func(saved_points), "secder"
    )
    return sequential_counts, sequential_deduped_counts
