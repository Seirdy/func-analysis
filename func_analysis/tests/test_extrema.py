# -*- coding: utf-8 -*-

"""Test AnalyzedFunc's relative and absolute extrema."""

import numpy as np

from func_analysis.tests import typechecking


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
    typechecking.typecheck_zcp(maxima)
    typechecking.typecheck_zcp(minima)
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
