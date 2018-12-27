"""The classes that do the actual function analysis."""

from __future__ import annotations

from numbers import Real
from typing import List

import numpy as np

from func_analysis.analyzed_func.af_crits_pois import AnalyzedFuncSpecialPts
from func_analysis.util import (
    Interval,
    assemble_table,
    decreasing_intervals,
    increasing_intervals,
    make_intervals,
)


class _AnalyzedFuncIntervals(AnalyzedFuncSpecialPts):
    """Function analysis with special intervals..

    These intervals include:
        - Intervals of increase/decrease.
        - Intervals of concavity/convexity.
    """

    def _construct_intervals(self, points: List[Real]) -> List[Interval]:
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
        intervals_of_concavity : [Interval]
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


class AnalyzedFunc(_AnalyzedFuncIntervals):
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
        x_vals: List[Real] = np.concatenate(
            (self.relative_maxima, self.x_range)
        )
        pairs: np.ndarray = assemble_table(self.func, x_vals)
        return pairs[np.argmax(pairs[:, 1])]

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
        x_vals: List[Real] = np.concatenate(
            (self.relative_minima, self.x_range)
        )
        pairs: np.ndarray = assemble_table(self.func, x_vals)
        return pairs[np.argmin(pairs[:, 1])]
