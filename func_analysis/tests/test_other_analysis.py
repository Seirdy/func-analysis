#!/usr/bin/env python3.7
# -*- coding: utf-8 -*-
# pylint: disable=comparison-with-callable
"""Tests func_analysis.

This deliberately uses a function requiring a high degree of precision
"""
from typing import Tuple

import numpy as np

import pytest
from func_analysis._analysis_classes import AnalyzedFunc
from func_analysis.tests import testing_utils
from func_analysis.tests.call_counting import (
    total_counts_pre_analysis,
    workout_analyzed_func,
)


def test_analyzedfunc_has_no_throwaways(analyzed_trig_func):
    """Ensure that the throwaway overloading functions are removed."""
    with pytest.raises(AttributeError) as errinfo:
        error_supplement = analyzed_trig_func.func_iterable
    if "has no attribute" not in str(errinfo.value):
        raise AttributeError(
            "analyzed_trig_func._ should have been deleted, "
            + "but found it to be "
            + str(error_supplement)
        )


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
    maxima = analyzed_trig_func.relative_maxima()
    minima = analyzed_trig_func.relative_minima()
    testing_utils.typecheck_zcp(maxima)
    testing_utils.typecheck_zcp(minima)
    np.testing.assert_equal(maxima, analyzed_trig_func.crits[::2])
    np.testing.assert_equal(minima, analyzed_trig_func.crits[1::2])


def test_trig_func_has_correct_abs_max(analyzed_trig_func):
    """Test that absolute_maximum() returns correct value.

    First, make sure that its approximation is correct. Then, compare
    the exact values.
    """
    trig_abs_max = analyzed_trig_func.absolute_maximum()
    approximate_expected_max = [-46.35559793676238, 1.013176643861527]
    np.testing.assert_allclose(
        np.float128(trig_abs_max), approximate_expected_max
    )
    exact_expected_max = analyzed_trig_func.relative_maxima()[10]
    np.testing.assert_equal(
        trig_abs_max,
        [exact_expected_max, analyzed_trig_func.func(exact_expected_max)],
    )


def test_trig_func_has_correct_abs_min(analyzed_trig_func):
    """Test that absolute_minimum() returns correct value."""
    expected_min = analyzed_trig_func.relative_minima()[0]
    np.testing.assert_equal(
        analyzed_trig_func.absolute_minimum(),
        [expected_min, analyzed_trig_func.func(expected_min)],
    )


def test_parabola_has_symmetry(analyzed_parab):
    """Check analyzed_parab's symmetry functions."""
    assert analyzed_parab.has_symmetry(axis=0)
    np.testing.assert_equal(
        analyzed_parab.vertical_axis_of_symmetry(), analyzed_parab.crits
    )
    analyzed_parab_new = AnalyzedFunc(
        func=analyzed_parab.func, x_range=(-8, 8), zeros_wanted=2
    )
    np.testing.assert_equal(
        analyzed_parab_new.vertical_axis_of_symmetry(), analyzed_parab.crits
    )


def test_call_counting(analyzed_trig_func):
    """Check and print call all_counts for each executed function."""
    assert total_counts_pre_analysis() == 0
    counts = workout_analyzed_func(analyzed_trig_func)
    original_vals: Tuple[int, ...] = tuple(counts[0].values())
    deduped_vals: Tuple[int, ...] = tuple(counts[1].values())
    uniqueness = np.divide(deduped_vals, original_vals)
    # Ensure that memoized func isn't called again for repeat vals.
    counts_after_repeat = counts[0]["dupe"]
    assert original_vals.count(counts_after_repeat) > 1
    assert counts_after_repeat < 1700
    assert np.amin(uniqueness) > 0.8