# -*- coding: utf-8 -*-

"""Extend AnalyzedFuncBase to be able to use zeros."""

from __future__ import annotations

from functools import lru_cache
from numbers import Real
from typing import Collection, Iterator, List, Optional, Set, Tuple

import mpmath as mp
import numpy as np
from scipy.optimize import brentq

from func_analysis.analyzed_func.af_base import AnalyzedFuncBase
from func_analysis.custom_types import Func, Interval
from func_analysis.interval_util import make_intervals, make_pairs


class AnalyzedFuncZeros(AnalyzedFuncBase):
    """Function analysis with root-finding."""

    def __init__(
        self, zeros_wanted: int = 0, zeros: Collection[Real] = None, **kwargs
    ):
        """Initialize the object.

        Parameters
        ----------
        zeros_wanted
            The number of zeros to find.
        zeros
            List of zeros already known. Used as starting points for
            more exact computation.
        **kwargs
            Keyword arguments to pass to super. See doc for
            ``AnalyzedFuncBase.__init__()``.

        """
        super().__init__(**kwargs)
        self.zeros_wanted = zeros_wanted
        if zeros is not None:
            self._zeros: np.ndarray = items_in_range(
                np.array(zeros), self.x_range
            )
        else:
            self._zeros = None

    @lru_cache(maxsize=1)
    def _all_zero_intervals(self) -> List[Interval]:
        """Find ALL zero intervals for this object's function.

        Returns
        -------
        List[Interval]
            All x-intervals across which the function being analyzed
            crosses the x-axis. Minimum number of intervals is
            ``self.zeros_wanted``.

        """
        points_to_plot = self.zeros_wanted + 3

        zero_intervals_found: List[Interval] = zero_intervals(
            self.plot(points_to_plot)
        )
        while len(zero_intervals_found) < self.zeros_wanted:
            points_to_plot += 1
            zero_intervals_found = zero_intervals(self.plot(points_to_plot))
        return zero_intervals_found

    @lru_cache(maxsize=1)
    def _solved_intervals(self) -> Set[Interval]:
        """Filter zero intervals containing a zero already known.

        Returns
        -------
        intervals_found : Set[Interval]
            A subset of ``self._all_zero_intervals()`` containing
            intervals that contain values already present in
            ```self._zeros```

        """
        return {
            possible_zero_interval
            for possible_zero_interval in self._all_zero_intervals()
            # if any zeros are found that fit in an interval,
            # include the interval.
            if np.logical_and(
                self._zeros > possible_zero_interval[0],
                self._zeros < possible_zero_interval[1],
            ).any()
        }

    @lru_cache(maxsize=1)
    def _known_zeros(self) -> Optional[Iterator[Real]]:
        """Make ``self._zeros`` an iterator, if possible.

        Returns
        -------
        zeros : Optional[Iterator[Real]]
            ``None`` if ``self._zeros`` is ``None``. Otherwise, an
            iterator that iterates across ``self._zeros``.

        """
        try:
            return iter(self._zeros)
        except TypeError:
            return None

    def _compute_zeros(self) -> Iterator[Real]:
        """Compute each zero wanted.

        mpmath's root-finders can take an imprecise starting point.
        If an interval has an already-found zero, use that as the
        starting point.

        Yields
        ------
        zero : Real
            The next zero for the function.

        """
        for interval in self._all_zero_intervals():
            if self._known_zeros() and interval in self._solved_intervals():
                yield find_one_zero(
                    self.func,
                    interval,
                    # If we made it this far, self._known_zeros will
                    # not raise a StopIteration exception.
                    # pylint: disable=stop-iteration-return
                    next(self._known_zeros())  # type: ignore
                    # pylint: enable=stop-iteration-return
                )
            else:
                yield find_one_zero(self.func, interval)

    @property
    def zeros(self) -> np.ndarray:
        """Find all zeros wanted.

        Returns
        -------
        zeros : ndarray
            An array of precise zeros for the function.

        """
        if not self.zeros_wanted:
            return np.array([])
        if self._zeros is None or len(self._zeros) < self.zeros_wanted:
            # Collect values from self._compute_zeros() into a numpy array.
            self._zeros = np.array(tuple(self._compute_zeros()))
        return self._zeros


def find_one_zero(
    func: Func, x_range: Tuple[Real, Real], starting_point: Real = None
) -> Real:
    """Find the zero of a function in a given interval.

    mpmath's zero-finding algorithms require a starting "guess" point.
    ``scipy.optimize.brentq`` can find an imprecise zero in a given
    interval. Combining these, this method uses the output of
    ``scipy.optimize.brentq`` as a starting point for mpmath's more
    precise root-finding algo.

    If a starting point is provided, the interval argument
    becomes unnecessary.

    Parameters
    ----------
    func
        The function to find a zero for.
    x_range
        The x-interval in which to find a zero. It must contain at
        least one zero.
    starting_point
        A guess-point. Can be ``None``, in which case
        use ``scipy.optimize.brentq`` to calculate one.

    Returns
    -------
    zero : Real
        A single very precise zero.

    """
    # If a starting point is not provided, find one.
    if starting_point is None:
        starting_point = brentq(
            f=func, a=x_range[0], b=x_range[1], maxiter=50, disp=False
        )
    # Maybe this starting point is good enough.
    if not func(starting_point):
        return starting_point
    return mp.findroot(f=func, x0=starting_point)


def items_in_range(
    unfiltered: np.ndarray, interval: Tuple[Real, Real]
) -> np.ndarray:
    """Filter items to contain just items in closed interval.

    Parameters
    ----------
    unfiltered : ndarray of Reals
        The 1D array to filter.
    interval
        The closed interval of acceptable values. Doesn't necessarily
        have to be an instance of Interval.

    Returns
    -------
    filtered_items : ndarray
        A subset of ``unfiltered`` that includes only values in
        ``interval``.

    """
    mask = np.logical_and(
        min(interval) <= unfiltered, max(interval) >= unfiltered
    )
    return unfiltered[mask]


def zero_intervals(coordinates: np.ndarray) -> List[Interval]:
    """Find open intervals containing zeros.

    Parameters
    ----------
    coordinates
        An x-y table represented by a 2d ndarray.

    Returns
    -------
    List[Interval]
        A list of x-intervals across which the function represented
        by the x-y table crosses the x-axis

    """
    x_intervals = make_intervals(coordinates[:, 0])
    is_positive = make_pairs(np.greater(coordinates[:, 1], 0))
    return [
        interval_map[0]
        for interval_map in zip(x_intervals, is_positive)
        if interval_map[1][0] is not interval_map[1][1]
    ]
