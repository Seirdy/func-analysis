#!/usr/bin/env python3.7
# -*- coding: utf-8 -*-
# pylint: disable=comparison-with-callable
"""Tests func_analysis.

This deliberately uses a function requiring a high degree of precision
"""

import numpy as np

from func_analysis.analysis_classes import AnalyzedFunc
from func_analysis.tests import testing_utils


def test_zeroth_derivative_is_itself(analyzed_trig_func):
    """Check that nth_derivative(0) returns the unaltered function."""
    assert analyzed_trig_func.nth_derivative(0) == analyzed_trig_func.func


def test_trig_func_has_correct_relative_extrema(analyzed_trig_func):
    """Test correctness of analyzed_trig_func's relative extrema.

    More specifically, test correctness of
    analyzed_trig_func.relative_maxima() and
    analyzed_trig_func.relative_minima().
    Since this is a wave function, critical points alternate between
    relative minima and relative maxima.
    """
    maxima = analyzed_trig_func.relative_maxima
    minima = analyzed_trig_func.relative_minima
    testing_utils.typecheck_zcp(maxima)
    testing_utils.typecheck_zcp(minima)
    np.testing.assert_equal(maxima, analyzed_trig_func.crits[::2])
    np.testing.assert_equal(minima, analyzed_trig_func.crits[1::2])


def test_trig_func_has_correct_abs_max(analyzed_trig_func):
    """Test that absolute_maximum() returns correct value.

    First, make sure that its approximation is correct. Then, compare
    the exact values.
    """
    trig_abs_max = analyzed_trig_func.absolute_maximum
    approximate_expected_max = [-46.35559793676238, 1.013176643861527]
    np.testing.assert_allclose(
        np.float128(trig_abs_max), approximate_expected_max
    )
    exact_expected_max = analyzed_trig_func.relative_maxima[10]
    np.testing.assert_equal(
        trig_abs_max,
        [exact_expected_max, analyzed_trig_func.func(exact_expected_max)],
    )


def test_trig_func_has_correct_abs_min(analyzed_trig_func):
    """Test that absolute_minimum() returns correct value."""
    expected_min = analyzed_trig_func.relative_minima[0]
    np.testing.assert_equal(
        analyzed_trig_func.absolute_minimum,
        [expected_min, analyzed_trig_func.func(expected_min)],
    )


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
