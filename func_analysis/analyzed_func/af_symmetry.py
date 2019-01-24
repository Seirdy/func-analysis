# -*- coding: utf-8 -*-

"""Add axes-of-symmetry to function analysis."""
import random
from typing import List, Sequence

import numpy as np
from mpmath import mp

from func_analysis.analyzed_func.af_crits_pois import AnalyzedFuncSpecialPts


class AnalyzedFuncSymmetry(object):
    """Add axes-of-symmetry to function analysis."""

    def __init__(
        self, func, x_range, crits_wanted: int = None, crits: Sequence = None
    ):
        """Initialize using composition with ``AnalyzedFuncSpecialPts``."""
        self.analyzed_func_special_pts = AnalyzedFuncSpecialPts(
            func=func, x_range=x_range, crits_wanted=crits_wanted, crits=crits
        )

    def _plot_enough(self, points_to_plot: int = 50):
        """Make plotted_points meet a minimum length.

        Parameters
        ----------
        points_to_plot
            The minimum number of points that should be plotted.

        Returns
        -------
        plotted_points: List[Coordinate]
            Random sample of ``self.plotted_points`` with size specified by
            ``points_to_plot``.

        """
        # Plot at least points_to_plot points.
        num_coords_found = len(self.analyzed_func_special_pts.plotted_points)
        if num_coords_found < points_to_plot:
            self.analyzed_func_special_pts.plot(
                points_to_plot - num_coords_found
            )
        # Don't return more points than necessary.
        all_points = self.analyzed_func_special_pts.plotted_points
        return random.choices(population=all_points, k=points_to_plot)

    def has_symmetry(self, axis: int) -> bool:
        """Determine function symmetry about a vertical axis.

        Reflects a sample of function coordinates across a vertical
        axis to determine if y-values remain unchanged.

        Parameters
        ----------
        axis
            The number representing the x-value of the vertical line
            about which the function might have symmetry.

        Returns
        -------
        has_symmetry : bool
            True if the function is symmetric about ``axis``,
            False otherwise.

        """
        saved_coordinates = np.array(self._plot_enough())
        x_vals = saved_coordinates[:, 0]
        y_vals = saved_coordinates[:, 1]
        x_mirror = np.subtract(2 * axis, x_vals)
        y_mirror = self.analyzed_func_special_pts.func_iterable(x_mirror)
        return np.array_equal(np.abs(y_vals), np.abs(y_mirror))

    @property
    def vertical_axis_of_symmetry(self) -> List[mp.mpf]:
        """Find all vertical axes of symmetry.

        Returns
        -------
        list_of_axes : List[Real]
            A list of x-values for vertical lines about which the
            function has symmetry.

        """
        return [
            crit
            for crit in self.analyzed_func_special_pts.crits
            if self.has_symmetry(axis=crit)
        ]
