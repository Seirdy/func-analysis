"""Helper functions for testing AnalyzedFunc."""
from __future__ import annotations

from functools import update_wrapper
from typing import Dict, List, Tuple

import numpy as np

from func_analysis.analyzed_func import AnalyzedFunc


# pylint: disable=undefined-variable
class AnalyzedFuncCounted(AnalyzedFunc):
    """Saves all instances of an object."""

    instances: List[AnalyzedFuncCounted] = []  # noqa: F821

    def __init__(self, *args, **kwargs):
        """Initialize the object."""
        super().__init__(*args, **kwargs)
        self.instances.append(self)

    @property
    def counts(self) -> int:
        """Count unique calls to self.func."""
        return len(self.plotted_points)

    @property
    def deduped_counts(self) -> int:
        """Remove duplicates from self.counts."""
        return len(np.unique([coord[0] for coord in self.plotted_points]))


# pylint: enable=undefined-variable


class ForbidCalling(object):
    """Class decorator for tracking state."""

    def __init__(self, func):
        """Initialize the object."""
        update_wrapper(self, func)
        self.func = func

    def __call__(self, *args):
        """Increment counter each time func is called."""
        raise RuntimeError(
            "AnalyzedFunc.func is supposed to be an altered copy of the func "
            "supplied to the constructor, but the original func was accessed."
        )


SavedCounts = Dict[str, int]


def workout_analyzed_func(
    analyzed_func: AnalyzedFuncCounted
) -> Tuple[SavedCounts, SavedCounts]:
    """Track function calls throughout function analysis."""
    sequential_counts: SavedCounts = {}
    sequential_deduped_counts: SavedCounts = {}

    def logged_calculation(arg, key):
        """Save call counts and evaluate arg."""
        assert arg is not None
        sequential_counts[key] = analyzed_func.counts
        sequential_deduped_counts[key] = analyzed_func.deduped_counts

    logged_calculation(analyzed_func.zeros, "zeros")
    logged_calculation(analyzed_func.absolute_minimum, "abs_min")
    logged_calculation(analyzed_func.convex, "convex")
    saved_coords = set(analyzed_func.plotted_points)
    saved_points = [coord[0] for coord in saved_coords]
    # noinspection PyTypeChecker
    logged_calculation(analyzed_func.func_iterable(saved_points), "dupe")
    # noinspection PyTypeChecker
    logged_calculation(
        analyzed_func.rooted_second_derivative.func_iterable(saved_points),
        "sec_der",
    )
    return sequential_counts, sequential_deduped_counts
