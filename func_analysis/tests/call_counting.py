# -*- coding: utf-8 -*-

"""Helper functions for testing AnalyzedFunc."""
from __future__ import annotations

from functools import update_wrapper
from typing import Callable, Dict, List, Tuple

import numpy as np

from func_analysis import AnalyzedFunc


# pylint: disable=undefined-variable, too-many-ancestors
class AnalyzedFuncCounted(AnalyzedFunc):
    """Save number of calls/unique calls to function."""

    instances: List[AnalyzedFuncCounted] = []  # noqa: F821

    def __init__(self, **kwargs):
        """Initialize the object."""
        super().__init__(**kwargs)
        self.instances.append(self)

    @property
    def counts(self) -> int:
        """Count unique calls to self.func."""
        return len(self.plotted_points)

    @property
    def deduped_counts(self) -> int:
        """Remove duplicates from self.counts."""
        return len(np.unique([coord[0] for coord in self.plotted_points]))


# pylint: enable=undefined-variable, too-many-ancestors


class ForbidCalling(object):
    """Function decorator to forbid calls."""

    def __init__(self, func: Callable):
        """Initialize the object.

        Parameters
        ----------
        func
            The function to wrap and forbid calling.

        """
        update_wrapper(self, func)
        self.func = func

    def __call__(self, *args, **kwargs):
        """Raise error instead of calling the function.

        Parameters
        ----------
        *args, **kwargs
            The arguments that would normally have been passed to the
            function. These are actually unused.

        Raises
        ------
        RuntimeError
            If called.

        """
        raise RuntimeError(
            f"Tried to call forbidden function {self.__name__}."
        )


# Type annotation for dictionary of saved call-counts
SavedCounts = Dict[str, int]


def workout_analyzed_func(
    analyzed_func: AnalyzedFuncCounted
) -> Tuple[SavedCounts, SavedCounts]:
    """Track function calls throughout function analysis.

    Runs just about all the analysis possible on analyzed_func and
    tracks the call count of its function.

    Parameters
    ----------
    analyzed_func
        The ``AnalyzedFuncCounted`` object to run analysis on while
        counting calls.

    Returns
    -------
    Tuple[Dict[str, int], Dict[str, int]]
        The number of times ``analyzed_func.func`` was called, and the
        number of times it was called with a unique argument.

    """
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
