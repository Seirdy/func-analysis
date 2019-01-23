# -*- coding: utf-8 -*-
"""The classes that do the actual function analysis."""
from typing import List

import mpmath as mp
import numpy as np

from func_analysis.analyzed_func.af_intervals_extrema import (
    AnalyzedFuncExtrema,
    AnalyzedFuncIntervals,
)
from func_analysis.analyzed_func.af_symmetry import AnalyzedFuncSymmetry
from func_analysis.custom_types import Coordinate
from func_analysis.decorators import copy_docstring_from


# This class is kinda big by design; it needs a lot of members.
class AnalyzedFunc(AnalyzedFuncIntervals):  # noqa: Z214
    """Complete function analysis.

    Function analysis includes:
        - special points
        - special intervals
        - signed/unsigned area.
    """

    def __init__(self, **kwargs):
        """Initialize the object."""
        super().__init__(**kwargs)
        self._af_extrema = AnalyzedFuncExtrema(**kwargs)
        self._af_symmetry = AnalyzedFuncSymmetry(
            func=self.func_real,
            x_range=self.x_range,
            crits_wanted=self.af_crits.crits_wanted,
            crits=self._crits,
        )

    # mypy false positive: decorated property not supported.
    # See https://github.com/python/mypy/issues/1362
    # noinspection PyCallingNonCallable
    @property  # type: ignore
    @copy_docstring_from(AnalyzedFuncExtrema.relative_minima)
    def relative_minima(self) -> np.ndarray:
        """Find all the relative minima of the function.

        See Also
        --------
        AnalyzedFuncExtrema.relative_minima

        """
        return self._af_extrema.relative_minima

    # mypy false positive: decorated property not supported.
    # See https://github.com/python/mypy/issues/1362
    # noinspection PyCallingNonCallable
    @property  # type: ignore
    @copy_docstring_from(AnalyzedFuncExtrema.relative_maxima)
    def relative_maxima(self) -> np.ndarray:
        """Find all the relative maxima of the function.

        See Also
        --------
        AnalyzedFuncExtrema.relative_maxima

        """
        return self._af_extrema.relative_maxima

    # mypy false positive: decorated property not supported.
    # See https://github.com/python/mypy/issues/1362
    # noinspection PyCallingNonCallable
    @property  # type: ignore
    @copy_docstring_from(AnalyzedFuncExtrema.absolute_maximum)
    def absolute_maximum(self) -> Coordinate:
        """Find all the absolute maximum of the function.

        See Also
        --------
        AnalyzedFuncExtrema.absolute_maximum

        """
        return self._af_extrema.absolute_maximum

    # mypy false positive: decorated property not supported.
    # See https://github.com/python/mypy/issues/1362
    # noinspection PyCallingNonCallable
    @property  # type: ignore
    @copy_docstring_from(AnalyzedFuncExtrema.absolute_minimum)
    def absolute_minimum(self) -> Coordinate:
        """Find all the absolute minimum of the function.

        See Also
        --------
        AnalyzedFuncExtrema.absolute_minimum

        """
        return self._af_extrema.absolute_minimum

    @property
    def signed_area(self) -> mp.mpf:
        """Calculate the definite integral bounded by x_range.

        Returns
        -------
        signed_area : mp.mpf
            The signed area of the analyzed function relative to the
            x-axis.

        """
        return mp.quad(self.func_real, self.x_range)

    @property
    def unsigned_area(self) -> mp.mpf:
        """Calculate the geometric area bounded by x_range.

        Returns
        -------
        unsigned_area : mp.mpf
            The unsigned area of the analyzed function relative to the
            x-axis.

        """
        return mp.quad(lambda x_val: abs(self.func_real(x_val)), self.x_range)

    # noinspection PyCallingNonCallable
    @copy_docstring_from(AnalyzedFuncSymmetry.has_symmetry)
    def has_symmetry(self, axis: int) -> bool:
        """Determine function symmetry about a vertical axis.

        See Also
        --------
        AnalyzedFuncSymmetry.has_symmetry

        """
        return self._af_symmetry.has_symmetry(axis=axis)

    # mypy false positive: decorated property not supported.
    # See https://github.com/python/mypy/issues/1362
    # noinspection PyCallingNonCallable
    @property  # type: ignore
    @copy_docstring_from(AnalyzedFuncSymmetry.vertical_axis_of_symmetry)
    def vertical_axis_of_symmetry(self) -> List[mp.mpf]:
        """Find all vertical axes of symmetry.

        See Also
        --------
        AnalyzedFuncSymmetry.vertical_axis_of_symmetry

        """
        return self._af_symmetry.vertical_axis_of_symmetry
