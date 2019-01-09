# -*- coding: utf-8 -*-
"""The classes that do the actual function analysis."""

from __future__ import annotations

from numbers import Real
from typing import Callable, Iterator, List, Sequence

import numpy as np

from func_analysis.analyzed_func.af_crits_pois import AnalyzedFuncSpecialPts
from func_analysis.custom_types import Interval
from func_analysis.interval_util import (
    decreasing_intervals,
    increasing_intervals,
    make_intervals,
)


class AnalyzedFuncIntervals(AnalyzedFuncSpecialPts):
    """Function analysis with special intervals.

    These intervals include:
        - Intervals of increase/decrease.
        - Intervals of concavity/convexity.
    """

    def _construct_intervals(self, points: List[Real]) -> Iterator[Interval]:
        """Construct intervals to filter in interval analysis.

        All interval analysis uses intervals bounded by values taken
        from the x-range and a set of special points.

        Parameters
        ----------
        points
            A set of special points to serve as the bounds of intervals.

        Returns
        -------
        Iterator[Interval]
            A bunch of intervals to filter into
            increasing/decreasing/concave/convex intervals.

        """
        points.insert(0, self.x_range.start)
        points.append(self.x_range.stop)
        return make_intervals(points)

    @property
    def increasing(self) -> List[Interval]:
        """List self.func's intervals of increase.

        Returns
        -------
        intervals_of_increase : List[Interval]
            All intervals of self.x_range across which self.func is
            increasing.

        """
        return increasing_intervals(
            self.func, self._construct_intervals(list(self.crits))
        )

    @property
    def decreasing(self) -> List[Interval]:
        """List self.func's intervals of decrease.

        Returns
        -------
        intervals_of_decrease : List[Interval]
            All intervals of self.x_range across which self.func is
            increasing.

        """
        return decreasing_intervals(
            self.func, self._construct_intervals(list(self.crits))
        )

    @property
    def concave(self) -> List[Interval]:
        """List self.func's intervals of concavity (opening up).

        Returns
        -------
        intervals_of_concavity : List[Interval]
            All intervals of self.x_range across which self.func is
            concave (opening up).

        """
        return increasing_intervals(
            self.rooted_first_derivative.func,
            self._construct_intervals(list(self.pois)),
        )

    @property
    def convex(self) -> List[Interval]:
        """List self.func's intervals of convexity. (opening down).

        Returns
        -------
        intervals_of_convexity : List[Interval]
            All intervals of self.x_range across which self.func is
            convex (opening down).

        """
        return decreasing_intervals(
            self.rooted_first_derivative.func,
            self._construct_intervals(list(self.pois)),
        )


class AnalyzedFuncExtrema(AnalyzedFuncSpecialPts):
    """Complete function analysis, with special points and intervals.

    This class adds relative/absolute extrema to the analysis.
    """

    @property
    def relative_maxima(self) -> np.ndarray:
        """List all relative maxima of self.func.

        Find the subset of self.crits that includes critical numbers
        appearing on intervals in which func is convex.

        Returns
        -------
        np.ndarray
            Array of precise relative maxima of self.func appearing in
            x_range.

        """
        fp2_of_crits = self.rooted_second_derivative.func_iterable(self.crits)
        mask = np.less(fp2_of_crits, 0)
        return self.crits[mask]

    @property
    def relative_minima(self) -> np.ndarray:
        """List all relative maxima of self.func.

        Find the subset of self.crits that includes critical numbers
        appearing on intervals in which func is concave.

        Returns
        -------
        np.ndarray
            Array of precise relative minima of self.func appearing in
            x_range.

        """
        fp2_of_crits = self.rooted_second_derivative.func_iterable(self.crits)
        mask = np.greater(fp2_of_crits, 0)
        return self.crits[mask]

    @property
    def absolute_maximum(self) -> np.ndarray:
        """Find the absolute maximum of self.simple_func.

        Find the maximum of self.relative_maxima and the bounds of
        x_range.

        Returns
        -------
        abs_max : np.ndarray
            The x-y coordinate of the absolute maximum of self.func in
            the form [x, y].

        """
        return self._absolute_extrema(self.relative_maxima, np.argmax)

    @property
    def absolute_minimum(self) -> np.ndarray:
        """Find the absolute minimum of self.simple_func.

        Find the minimum of self.relative_minima and the bounds of
        x_range.

        Returns
        -------
        abs_min : ndarray
            The x-y coordinate of the absolute minimum of self.func in
            the form [x, y].

        """
        return self._absolute_extrema(self.relative_minima, np.argmin)

    def _absolute_extrema(self, points: Sequence, extrema_finder: Callable):
        """Generalized absolute-extrema finder.

        Parameters
        ----------
        points
            The contenders for the absolute extrema excluding the
            bounds of self.x_range.
        extrema_finder
            The function that computes the index of the extrema
            (i.e., np.argmax or np.argmin).

        Returns
        -------
        np.ndarray
            The x-y coordinates of the absolute extrema.

        """
        x_vals: List[Real] = np.concatenate((points, self.x_range))
        pairs: np.ndarray = np.stack((x_vals, self.func(x_vals)), axis=-1)
        return pairs[extrema_finder(pairs[:, 1])]
