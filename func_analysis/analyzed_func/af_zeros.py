"""Extend AnalyzedFuncBase to be able to use zeros."""

from __future__ import annotations

from functools import lru_cache
from numbers import Real
from typing import Iterable, Iterator, List, Optional, Set, Tuple

import mpmath as mp
import numpy as np
from scipy.optimize import brentq

from func_analysis.af_util import Func, Interval, zero_intervals
from func_analysis.analyzed_func.af_base import AnalyzedFuncBase


class AnalyzedFuncZeros(AnalyzedFuncBase):
    """Function analysis with root-finding."""

    def __init__(
        self, zeros_wanted: int = 0, zeros: Iterable[Real] = None, **kwargs
    ):
        """Initialize the object.

        Parameters
        ----------
        zeros_wanted
            The number of zeros to find
        zeros
            List of zeros already known. Used as starting points for
            more exact computation.
        **kwargs
            Keyword arguments to pass to super. See doc for
            AnalyzedFunc.__init__()

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
            All x-intervals across which self.func crosses the x-axis.
            Minimum number of intervals is self.zeros_wanted.

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
            A subset of self._all_zero_intervals() containing zeros
            in self._zeros

        """
        # There are none if there are no zeros already known.
        intervals_found: List[Interval] = []
        for possible_zero_interval in self._all_zero_intervals():
            # if any zeros are found that fit in this interval,
            # append this interval.
            if np.logical_and(
                self._zeros > possible_zero_interval[0],
                self._zeros < possible_zero_interval[1],
            ).any():
                intervals_found.append(possible_zero_interval)
        return set(intervals_found)

    @lru_cache(maxsize=1)
    def _known_zeros(self) -> Optional[Iterator[Real]]:
        """Try to make self._zeros an iterator for _compute_zeros.

        Returns
        -------
        zeros : Optional[Iterator[Real]]
            None if self._zeros is None. Otherwise, an iterator that
            iterates across self._zeros.

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
                # If we made it this far, self._known_zeros will be an iterator
                # that will not raise a StopIteration exception.
                yield find_one_zero(
                    self.func,
                    interval,
                    # pylint: disable=stop-iteration-return
                    next(self._known_zeros())  # type: ignore
                    # pylint: enable=stop-iteration-return
                )
            else:
                yield find_one_zero(self.func, interval)

    @property
    def zeros(self) -> np.ndarray:
        """List all zeros wanted in x_range.

        Returns
        -------
        np.ndarray
            An array of precise zeros for self.func.

        """
        if not self.zeros_wanted:
            return np.array([])
        if self._zeros is None or len(self._zeros) < self.zeros_wanted:
            self._zeros = np.array(tuple(self._compute_zeros()))
        return self._zeros


def find_one_zero(
    func: Func, x_range: Tuple[Real, Real], starting_point: Real = None
) -> mp.mpf:
    """Find the zero of a function in a given interval.

    mpmath's zero-finding algorithms require a starting "guess" point.
    `scipy.optimize.brentq` can find an imprecise zero in a given
    interval. Combining these, this method uses scipy.optimize's output
    as a starting point for mpmath's more precise root-finding algo.

    If a starting point is provided, the interval argument
    becomes unnecessary.

    Parameters
    ----------
    func
        The function to find a zero for.
    x_range
        The x-interval in which to find a zero.
    starting_point
        A guess-point. Can be `None`, in which case
        use `scipy.optimize.brentq` to calculate one.

    Returns
    -------
    mp.mpf
        A single very precise zero.

    """
    # If a starting point is not provided, find one.
    if starting_point is None:
        # noinspection PyTypeChecker
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
    unfiltered
        The 1D array to filter
    interval
        The closed interval of acceptable values.

    Returns
    -------
    filtered_items : np.ndarray
        A subset of items that includes only values in interval

    """
    mask = np.logical_and(
        min(interval) <= unfiltered, max(interval) >= unfiltered
    )
    return unfiltered[mask]
