"""Add critical/inflection points to function analysis."""

from __future__ import annotations

from numbers import Real
from typing import Dict, List, Optional, Tuple

import mpmath as mp
import numpy as np

from func_analysis.analyzed_func.af_zeros import AnalyzedFuncZeros
from func_analysis.util import Func


class _AnalyzedFuncCrits(AnalyzedFuncZeros):
    """Initialize previously-known critical points."""

    def __init__(
        self,
        crits_wanted: int = None,
        crits: Tuple[Real, ...] = None,
        **kwargs,
    ):
        """Initialize the critical points of an analyzed function.

        Parameters
        ----------
        crits_wanted
            Real of critical nums to calculate.
        crits
            A list of critical numbers already known, used
            as starting points for more precise calculation.

        """
        super().__init__(**kwargs)
        if not crits_wanted:
            self.crits_wanted = max(self.zeros_wanted - 1, 0)
        else:
            self.crits_wanted = crits_wanted
        self._crits = crits


class AnalyzedFuncSpecialPts(_AnalyzedFuncCrits):
    """A RootedFunction with additional properties (critical Real).

    This object includes a function and its properties. If those
    properties are not provided, they will be calculated and saved.

    The function properties included:
        - generator of derivatives
        - zeros
        - critical Real
    """

    def __init__(
        self, pois_wanted: int = None, pois: Tuple[Real, ...] = None, **kwargs
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
        if pois_wanted is None:
            self.pois_wanted = max(self.crits_wanted - 1, 0)
        else:
            self.pois_wanted = pois_wanted
        self._pois = pois

    # pylint: disable=undefined-variable
    @property
    def rooted_first_derivative(self) -> AnalyzedFuncSpecialPts:  # noqa: F821
        """Return FuncZeros object for self.func's 1st derivative.

        Returns
        -------
        fprime : AnalyzedFuncZeros
            Analyzed 1st derivative of self.func, complete with zeros,
            crits, and an iterable func.

        """
        # pylint: enable=undefined-variable
        derivatives_of_fprime: Optional[Dict[int, Func]] = {
            nth - 1: self.derivatives[nth] for nth in self.derivatives.keys()
        }
        return AnalyzedFuncSpecialPts(
            func=self.nth_derivative(1),
            zeros_wanted=max(self.crits_wanted, 1),
            zeros=self._crits,
            derivatives=derivatives_of_fprime,
            x_range=self.x_range,
            crits_wanted=self.pois_wanted,
            crits=self._pois,
        )

    @property
    def rooted_second_derivative(self) -> AnalyzedFuncZeros:
        """Return FuncZeros object for self.func's 2nd derivative.

        Returns
        -------
        fprime2 : AnalyzedFuncZeros
            Analyzed 2nd derivative of self.func, complete with zeros
            and an iterable func.

        """
        derivatives_of_fprime2: Optional[Dict[int, Func]] = {
            nth - 2: self.derivatives[nth] for nth in self.derivatives.keys()
        }
        return AnalyzedFuncZeros(
            func=self.nth_derivative(2),
            zeros_wanted=max(self.pois_wanted, 1),
            zeros=self._pois,
            derivatives=derivatives_of_fprime2,
            x_range=self.x_range,
        )

    @property
    def crits(self) -> np.ndarray:
        """List all critical numbers wanted.

        This works by returning the zeros of the 1st derivative.

        Returns
        -------
        np.ndarray
            An array of precise critical points for self.func.

        """
        if not self.crits_wanted:
            return np.array([])
        if self._crits is None or len(self._crits) < self.crits_wanted:
            self._crits = self.rooted_first_derivative.zeros
        return self._crits

    @property
    def pois(self) -> np.ndarray:
        """List all points of inflection wanted.

        Returns
        -------
        np.ndarray
            An array of precise points of inflection for self.func.

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

    @property
    def vertical_axis_of_symmetry(self) -> List[mp.mpf]:
        """Find all vertical axes of symmetry.

        Returns
        -------
        list_of_axes : List[mpf]
            A list of x-values for vertical lines about which self.func
            has symmetry.

        """
        return [crit for crit in self.crits if self.has_symmetry(axis=crit)]
