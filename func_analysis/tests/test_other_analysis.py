#!/usr/bin/env python3.7
# -*- coding: utf-8 -*-
# pylint: disable=comparison-with-callable
"""Tests func_analysis.

This deliberately uses a function requiring a high degree of precision
"""

from decimal import Decimal

import numpy as np

from func_analysis.analyzed_func import AnalyzedFunc
from pytest import raises


def test_zeroth_derivative_is_itself(all_analyzed_funcs):
    """Check that nth_derivative(0) returns the unaltered function."""
    for analyzed_func in all_analyzed_funcs:
        assert analyzed_func.nth_derivative(0) == analyzed_func.func


def test_func_raises_error(all_analyzed_funcs):
    """Check that AnalyzedFunc.func raises exception.

    AnalyzedFunc.func raises a special TypeError for unregistered types.
    """
    for analyzed_func in all_analyzed_funcs:
        with raises(TypeError) as excinfo:
            analyzed_func.func(Decimal(2))

        assert (
            str(excinfo.value)
            == "Unsupported type '<class 'decimal.Decimal'>'"
        )
        assert excinfo.typename == "TypeError"


def test_parabola_has_symmetry(analyzed_parab):
    """Check analyzed_parab's symmetry functions."""
    assert analyzed_parab.has_symmetry(axis=0)
    np.testing.assert_equal(
        analyzed_parab.vertical_axis_of_symmetry, analyzed_parab.crits
    )
    analyzed_parab_new = AnalyzedFunc(
        func=analyzed_parab.func, x_range=(-8, 8), zeros_wanted=2
    )
    np.testing.assert_equal(
        analyzed_parab_new.vertical_axis_of_symmetry, analyzed_parab.crits
    )
