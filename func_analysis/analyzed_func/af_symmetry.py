"""Add axes-of-symmetry to function analysis."""
from numbers import Real
from typing import List, Sequence

import numpy as np
from mpmath import mp

from func_analysis.analyzed_func.af_crits_pois import AnalyzedFuncSpecialPts


class AnalyzedFuncSymmetry(object):
    """Add axes-of-symmetry to function analysis."""

    def __init__(
        self, func, x_range, crits_wanted: int = None, crits: Sequence = None
    ):
        """Initialize with class composition for AnalyzedFuncSpecialPts."""
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
            self.plotted_points after the minimum number of points to
            plot has been plotted.

        """
        num_coords_found = len(self.analyzed_func_special_pts.plotted_points)
        if num_coords_found < points_to_plot:
            self.analyzed_func_special_pts.plot(
                points_to_plot - num_coords_found
            )
        return self.analyzed_func_special_pts.plotted_points

    def has_symmetry(self, axis: Real) -> bool:
        """Determine if self.func is symmetric about given axis.

        Parameters
        ----------
        axis
            The number representing the domain of the vertical
            line about which self.func has symmetry.

        Returns
        -------
        bool
            True if self.func is symmetric about axis, False otherwise.

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
        list_of_axes : List[mpf]
            A list of x-values for vertical lines about which self.func
            has symmetry.

        """
        return [
            crit
            for crit in self.analyzed_func_special_pts.crits
            if self.has_symmetry(axis=crit)
        ]
