"""Extend AnalyzedFuncBase to be able to use zeros."""

from __future__ import annotations

from functools import lru_cache
from numbers import Real
from typing import Iterable, Iterator, List, Optional

import numpy as np

from func_analysis.analyzed_func.af_base import AnalyzedFuncBase
from func_analysis.util import (
    Interval,
    find_one_zero,
    items_in_range,
    zero_intervals,
)


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
    def _solved_intervals(self) -> List[Interval]:
        """Filter zero intervals containing a zero already known.

        Returns
        -------
        intervals_found : List[Interval]
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
        return intervals_found

    def _known_zeros(self) -> Optional[Iterator[Real]]:
        """Try to make self._zeros an iteratorfor _compute_zeros.

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
        starting_pts = self._known_zeros()
        for interval in self._all_zero_intervals():
            if starting_pts and interval in self._solved_intervals():
                yield find_one_zero(self.func, interval, next(starting_pts))
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
