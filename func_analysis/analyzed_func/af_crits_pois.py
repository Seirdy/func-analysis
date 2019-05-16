# -*- coding: utf-8 -*-

"""Add critical/inflection points to function analysis."""

from __future__ import annotations

from numbers import Real
from typing import Dict, Sequence

import numpy as np

from func_analysis.analyzed_func.af_base import AnalyzedFuncBase
from func_analysis.analyzed_func.af_zeros import AnalyzedFuncZeros
from func_analysis.custom_types import Func
from func_analysis.decorators import copy_metadata_from


class _AnalyzedFuncCrits(object):
    """Initialize previously-known critical points."""

    def __init__(
        self, crits_wanted: int = None, crits: Sequence[Real] = None, **kwargs
    ):
        """Initialize the critical points of an analyzed function.

        Parameters
        ----------
        crits_wanted
            Number of critical nums to calculate.
        crits
            A list of critical numbers already known, used
            as starting points for more precise calculation.
        **kwargs
            Keyword arguments to pass to ``AnalyzedFuncZeros``.

        See Also
        --------
        AnalyzedFuncZeros : embedded via composition.

        """
        self.af_zeros = AnalyzedFuncZeros(**kwargs)
        if crits_wanted:
            self.crits_wanted = crits_wanted
        else:
            self.crits_wanted = max(self.af_zeros.zeros_wanted - 1, 0)
        self.crits = crits


class AnalyzedFuncSpecialPts(AnalyzedFuncBase):
    """A RootedFunction with additional properties (critical Real).

    This object includes a function and its properties. If those
    properties are not provided, they will be calculated and saved.

    The function properties included:
        - derivatives
        - zeros
        - critical points
        - inflection points.
    """

    def __init__(
        self, pois_wanted: int = None, pois: Sequence[Real] = None, **kwargs
    ):
        """Initialize a CriticalFunction.

        Parameters
        ----------
        pois_wanted
            Real of points of inflection to calculate.
        pois
            A list of points of inflection already known, used
            as starting points for more precise calculation.
        **kwargs
            Keyword arguments to pass to super. See doc for
            AnalyzedFuncZeros.__init__()

        """
        super().__init__(**kwargs)
        self._af_crits = _AnalyzedFuncCrits(**kwargs)
        self._crits = self._af_crits.crits
        if pois_wanted is None:
            self.pois_wanted = max(self._af_crits.crits_wanted - 1, 0)
        else:
            self.pois_wanted = pois_wanted
        self._pois = pois

    # mypy false positive: decorated property not supported.
    # See https://github.com/python/mypy/issues/1362
    @property  # type: ignore
    @copy_metadata_from(AnalyzedFuncZeros.zeros)
    def zeros(self):
        """List all zeros wanted in x_range.

        See Also
        --------
        AnalyzedFuncZeros.zeros

        """
        return self._af_crits.af_zeros.zeros

    # pylint and flake8 don't yet recognize postponed evaluation of
    # annotations.
    @property
    def rooted_first_derivative(self) -> AnalyzedFuncSpecialPts:  # noqa: F821
        """Analyze self.func's 1st derivative.

        Returns
        -------
        fprime : AnalyzedFuncSpecialPts
            Analyzed 1st derivative of the function, complete with
            zeros, crits, and an iterable func.

        """
        derivatives_of_fprime: Dict[int, Func] = {
            nth - 1: self._af_crits.af_zeros.derivatives[nth]
            for nth in self._af_crits.af_zeros.derivatives.keys()
        }
        return AnalyzedFuncSpecialPts(
            func=self._af_crits.af_zeros.nth_derivative(1),
            zeros_wanted=max(self._af_crits.crits_wanted, 1),
            zeros=self._crits,
            derivatives=derivatives_of_fprime,
            x_range=self._af_crits.af_zeros.x_range,
            crits_wanted=self.pois_wanted,
            crits=self._pois,
        )

    @property
    def rooted_second_derivative(self) -> AnalyzedFuncZeros:
        """Analyze self.func's 1st derivative.

        Returns
        -------
        fprime2 : AnalyzedFuncZeros
            Analyzed 2nd derivative of the function, complete with
            zeros and an iterable func.

        """
        # This doesn't use the obvious approach of
        # self.rooted_first_derivative.rooted_first_derivative because
        # doing so could re-calculate known values.
        derivatives_of_fprime2: Dict[int, Func] = {
            nth - 2: self._af_crits.af_zeros.derivatives[nth]
            for nth in self._af_crits.af_zeros.derivatives.keys()
        }
        return AnalyzedFuncZeros(
            func=self._af_crits.af_zeros.nth_derivative(2),
            zeros_wanted=max(self.pois_wanted, 1),
            zeros=self._pois,
            derivatives=derivatives_of_fprime2,
            x_range=self._af_crits.af_zeros.x_range,
        )

    @property
    def crits(self) -> np.ndarray:
        """List all critical numbers wanted.

        This works by returning the zeros of the 1st derivative.

        Returns
        -------
        crits : ndarray of Reals
            An array of precise critical points for the function.

        """
        if not self._af_crits.crits_wanted:
            return np.array([])
        if (
            self._crits is None
            or len(self._crits) < self._af_crits.crits_wanted
        ):
            self._crits = self.rooted_first_derivative.zeros
        return self._crits

    @property
    def pois(self) -> np.ndarray:
        """List all points of inflection wanted.

        Returns
        -------
        pois : ndarray of Reals.
            An array of precise points of inflection for the function.

        """
        if not self.pois_wanted:
            return np.array([])
        if self._pois is None or len(self._pois) < self.pois_wanted:
            fp2_zeros = self.rooted_second_derivative.zeros
            self._pois = fp2_zeros[
                np.nonzero(
                    self.rooted_first_derivative.func_iterable(fp2_zeros)
                )
            ]
        return self._pois
