#!/usr/bin/env python3.7
# -*- coding: utf-8 -*-
# pylint: disable=comparison-with-callable
"""Tests func_analysis.

This deliberately uses a function requiring a high degree of precision
"""
import mpmath as mp
import numpy as np

from .._analysis_classes import AnalyzedFunc
from .._util import make_intervals
from .helpers import (
    CountCalls,
    assert_output_lessthan,
    mpf_assert_allclose,
    typecheck_intervals,
    typecheck_zcp,
)

EPSILON_0 = 1e-20
EPSILON_1 = 3.05e-15
EPSILON_2 = 1.196_789_1e-6


def test_analyzedfunc_has_no_throwaways(analyzed_trig_func):
    """Ensure that the throwaway overloading functions are removed."""
    assert not hasattr(analyzed_trig_func, "_")


def test_zeroth_derivative_is_itself(analyzed_trig_func):
    """Check that nth_derivative(0) returns the unaltered function."""
    assert analyzed_trig_func.nth_derivative(0) == analyzed_trig_func.func


def test_trig_func_has_correct_zeros(analyzed_trig_func):
    """Test the correctness of analyzed_trig_func.zeros."""
    typecheck_zcp(analyzed_trig_func.zeros)
    expected_zeros = [
        -47.038_289_673_236_13,
        -47.018_473_233_395_28,
        -46.972_318_087_653_95,
        -46.950_739_626_397_91,
        -46.906_204_518_117_63,
        -46.882_958_270_910_02,
        -46.839_955_720_658_34,
        -46.815_121_707_485,
        -46.773_576_011_368_88,
        -46.747_224_922_729_01,
        -46.707_068_062_964_04,
        -46.679_264_553_080_85,
        -46.640_433_373_296_69,
        -46.611_238_416_225_63,
        -46.573_672_554_670_36,
        -46.543_145_221_101_68,
        -46.506_785_519_620_84,
        -46.474_984_380_574_83,
        -46.439_771_604_599_5,
        -46.406_755_885_040_05,
        -46.372_629_655_875_1,
    ]
    # approximate accuracy
    np.testing.assert_allclose(
        np.float64(analyzed_trig_func.zeros), expected_zeros, rtol=EPSILON_1
    )
    # Does the function evaluate to 0 at its zeros?
    assert_output_lessthan(
        func=analyzed_trig_func.func,
        x_vals=analyzed_trig_func.zeros,
        max_y=3.5692e-19,
    )


def test_trig_func_has_correct_crits(analyzed_trig_func):
    """Test the correctness of analyzed_trig_func.crits."""
    typecheck_zcp(analyzed_trig_func.crits)
    # approximate accuracy
    expected_crits = [
        -47.028_400_867_252_13,
        -46.995_216_177_440_79,
        -46.961_552_135_996_854,
        -46.928_318_300_227_15,
        -46.894_608_617_023_465,
        -46.861_324_416_365_34,
        -46.827_569_901_478_4,
        -46.794_234_116_960_42,
        -46.760_435_575_283_8,
        -46.727_046_992_482_36,
        -46.693_205_219_082_984,
        -46.659_762_632_756_91,
        -46.625_878_408_195_66,
        -46.592_380_626_945_825,
        -46.558_454_712_583_135,
        -46.524_900_563_516_226,
        -46.490_933_696_823_73,
        -46.457_322_030_198_725,
        -46.423_314_920_098_63,
        -46.389_644_613_934_3,
        -46.355_597_936_188_15,
    ]
    np.testing.assert_allclose(
        np.float64(analyzed_trig_func.crits), expected_crits, rtol=EPSILON_1
    )
    assert_output_lessthan(
        func=analyzed_trig_func.rooted_first_derivative().func,
        x_vals=analyzed_trig_func.crits,
        max_y=EPSILON_1,
    )


def assert_trig_func_pois_are_accurate(analyzedfunc, pois_found: np.ndarray):
    """Test pois() accuracy."""
    assert (
        np.float128(pois_found[3] + 46.944_940_655_832_212_248_274_091_985_22)
        < EPSILON_1
    )
    assert_output_lessthan(
        func=analyzedfunc.rooted_second_derivative().func,
        x_vals=analyzedfunc.pois,
        max_y=EPSILON_1,
    )


def pois_stay_close_when_given_fp2(analyzedfunc, fp2_zeros):
    """Test pois() when providing second derivative.

    This makes sure that it is possible to provide a second derivative
    to AnalyzedFunc instances and that it gets used to improve
    accuracy.
    """
    analyzed_trig_func_with_fp2 = AnalyzedFunc(
        func=analyzedfunc.func,
        x_range=(-47.05, -46.3499),
        zeros_wanted=21,
        crits_wanted=21,
        known_zeros=[-47.038_289_673_236_127, -46.406_755_885_040_056],
        derivatives={2: fp2_zeros.func},
    )
    # make sure fp2_zeros.func() is actually used by tracking its call count
    fp2_zeros_func_counts_before = len(fp2_zeros.plotted_points)

    more_exact_pois = analyzed_trig_func_with_fp2.pois
    fp2_zeros_func_counts_after = len(fp2_zeros.plotted_points)
    assert fp2_zeros_func_counts_after - fp2_zeros_func_counts_before > 50

    typecheck_zcp(more_exact_pois)
    mpf_assert_allclose(fp2_zeros.zeros, more_exact_pois, EPSILON_0)
    mpf_assert_allclose(more_exact_pois, analyzedfunc.pois, EPSILON_2)


def test_trig_func_has_correct_pois(analyzed_trig_func, fp2_zeros):
    """Test the correctness of analyzed_trig_func.pois.

    First, compare the output with approximate floating-point values.
    Then, compare the output with the pois found from its exact second
    derivative.
    """
    # typechecking
    typecheck_zcp(analyzed_trig_func.pois)
    assert_trig_func_pois_are_accurate(
        analyzed_trig_func, analyzed_trig_func.pois
    )
    mpf_assert_allclose(fp2_zeros.zeros, analyzed_trig_func.pois, EPSILON_2)
    pois_stay_close_when_given_fp2(analyzed_trig_func, fp2_zeros)


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
    typecheck_zcp(maxima)
    typecheck_zcp(minima)
    np.testing.assert_equal(maxima, analyzed_trig_func.crits[::2])
    np.testing.assert_equal(minima, analyzed_trig_func.crits[1::2])


def test_trig_func_has_correct_abs_max(analyzed_trig_func):
    """Test that absolute_maximum() returns correct value.

    First, make sure that its approximation is correct. Then, compare
    the exact values.
    """
    trig_abs_max = analyzed_trig_func.absolute_maximum()
    approximate_expected_max = [-46.355_597_936_762_38, 1.013_176_643_861_527]
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


def test_trig_func_has_correct_concavity_convexity(analyzed_trig_func):
    """Test analyzed_trig_func.concave() and .convex().

    It alternates between intervals of concavity and convexity.
    """
    all_pts = list(analyzed_trig_func.pois)
    all_pts.insert(0, analyzed_trig_func.min_x)
    all_pts.append(analyzed_trig_func.max_x)
    all_intervals = make_intervals(all_pts)

    np.testing.assert_array_equal(
        np.array(analyzed_trig_func.concave()), all_intervals[::2]
    )
    np.testing.assert_array_equal(
        np.array(analyzed_trig_func.convex()), all_intervals[1::2]
    )


def test_parabola_has_correct_zeros(analyzed_parab):
    """Check that analyzed_parab.zeros returns correct value."""
    np.testing.assert_equal(analyzed_parab.zeros, np.array([-2, 2]))


def test_parabola_has_correct_crits(analyzed_parab):
    """Check that analyzed_parab.crits returns correct value."""
    assert analyzed_parab.crits == [0]


def test_parabola_has_correct_concavity(analyzed_parab):
    """Test analyzed_parab.concave() returns correct value.

    parab_func is concave across its entire x_range.
    """
    assert analyzed_parab.concave() == [analyzed_parab.x_range]


def test_parabola_has_correct_convexity(analyzed_parab):
    """Test analyzed_parab.convex() returns correct value.

    parab_func is concave across its entire x_range.
    """
    assert analyzed_parab.convex() == []


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


def test_analyzed_incdecfunc_has_correct_decreasing(analyzed_incdecfunc):
    """Test accuracy of analyzed_incdecfunc.decreasing().

    This works really well because in x_range, incdecfunc decreases
    across (-3, -e). Comparing with an irrational constant really
    pushes the boundaries of the precision of func_analysis.
    """
    mpf_assert_allclose(
        analyzed_incdecfunc.decreasing(), [(-3, mp.fneg(mp.e))], EPSILON_1 / 11
    )


def test_analyzed_incdecfunc_has_correct_increasing_decreasing(
    analyzed_incdecfunc
):
    """Test FuncIntervals' increasing() and decreasing() methods."""
    analyzed_incdecfunc_increasing = analyzed_incdecfunc.increasing()
    analyzed_incdecfunc_decreasing = analyzed_incdecfunc.decreasing()

    typecheck_intervals(analyzed_incdecfunc_increasing)
    typecheck_intervals(analyzed_incdecfunc_decreasing)
    assert (
        analyzed_incdecfunc_increasing[0][0]
        == analyzed_incdecfunc_decreasing[0][1]
    )
    mpf_assert_allclose(
        analyzed_incdecfunc.increasing(),
        [(mp.fneg(mp.e), -0.001)],
        EPSILON_1 / 10,
    )


def test_incdecfunc_has_correct_zeros(analyzed_incdecfunc):
    """Test analyzed_incdecfunc.zeros returns correct value."""
    assert analyzed_incdecfunc.zeros == [-1]


def test_incdecfunc_has_correct_concavity(analyzed_incdecfunc):
    """Test analyzed_incdecfunc.concave() returns correct value.

    inc_dec_func is concave across its entire x_range.
    """
    assert analyzed_incdecfunc.concave() == [analyzed_incdecfunc.x_range]


def test_incdecfunc_has_correct_convexity(analyzed_incdecfunc):
    """Test analyzed_incdecfunc.convex() returns correct value.

    inc_dec_func is concave across its entire x_range.
    """
    assert analyzed_incdecfunc.convex() == []


def test_call_counting():
    """Check and print call counts for each executed function."""
    # assert trig_func.call_count < 2000
    # assert trig_func.call_count > 10
    print("\ncall counts\n===========")
    for counted_func in CountCalls.functions:
        print(counted_func.func.__name__ + ": " + str(counted_func.call_count))
