"""The classes that do the actual function analysis."""
from numbers import Real
from typing import List

import mpmath as mp

from func_analysis.analyzed_func.af_base import AnalyzedFuncArea
from func_analysis.analyzed_func.af_intervals_extrema import (
    AnalyzedFuncExtrema,
    AnalyzedFuncIntervals,
)
from func_analysis.analyzed_func.af_symmetry import AnalyzedFuncSymmetry


class AnalyzedFunc(AnalyzedFuncIntervals, AnalyzedFuncExtrema):
    """Complete function analysis, with special points and intervals."""

    @property
    def _analyzed_func_symmetry(self) -> AnalyzedFuncSymmetry:
        """Class composition for AnalyzedFuncSymmetry."""
        return AnalyzedFuncSymmetry(
            func=self.func_real,
            x_range=self.x_range,
            crits_wanted=self.crits_wanted,
            crits=self._crits,
        )

    @property
    def _analyzed_func_area(self) -> AnalyzedFuncArea:
        """Class composition for AnalyzedFuncArea."""
        return AnalyzedFuncArea(self.func_real, self.x_range)

    @property
    def signed_area(self) -> mp.mpf:
        """Calculate the definite integral bounded by x_range.

        Returns
        -------
        mp.mpf
            The signed area of the analyzed function relative to the
            x-axis.

        """
        return self._analyzed_func_area.signed_area

    @property
    def unsigned_area(self) -> mp.mpf:
        """Calculate the geometric area bounded by x_range.

        Returns
        -------
        mp.mpf
            The unsigned area of the analyzed function relative to the
            x-axis.

        """
        return self._analyzed_func_area.unsigned_area

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
        return self._analyzed_func_symmetry.has_symmetry(axis=axis)

    @property
    def vertical_axis_of_symmetry(self) -> List[mp.mpf]:
        """Find all vertical axes of symmetry.

        Returns
        -------
        list_of_axes : List[mpf]
            A list of x-values for vertical lines about which self.func
            has symmetry.

        """
        return self._analyzed_func_symmetry.vertical_axis_of_symmetry
