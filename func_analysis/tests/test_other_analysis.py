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


def test_zeroth_derivative_is_itself(analyzed_trig_func):
    """Check that nth_derivative(0) returns the unaltered function."""
    assert analyzed_trig_func.nth_derivative(0) == analyzed_trig_func.func


def test_func_raises_error(analyzed_trig_func):
    """Check that AnalyzedFunc.func raises exception.

    AnalyzedFunc.func raises a special TypeError for unregistered types.
    """
    with raises(TypeError) as excinfo:
        analyzed_trig_func.func(Decimal(2))

    assert str(excinfo.value) == "Unsupported type '<class 'decimal.Decimal'>'"
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
