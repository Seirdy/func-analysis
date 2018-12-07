#!/usr/bin/env python3.7
# -*- coding: utf-8 -*-
# pylint: disable=comparison-with-callable
"""Tests func_analysis.

This deliberately uses a function requiring a high degree of precision
"""
from __future__ import annotations

from functools import update_wrapper
from typing import Iterable, List, Union

import mpmath as mp
import numpy as np

from func_analysis.func_analysis import (
    FuncIntervals,
    FuncSpecialPts,
    _decreasing_intervals,
    _increasing_intervals,
    _make_intervals,
)

BuiltinFloat = Union[np.float128, np.float64, float]
Number = Union[mp.mpf, BuiltinFloat]

EPSILON_0 = 1e-20
EPSILON_1 = 3.05e-15
EPSILON_2 = 1.196_789_1e-6


def mpf_assert_allclose(actual, desired, atol=1e-3):
    """Assert that the two arrays are close enough.

    Similar to numpy.testing.assert_allclose().
    """
    assert np.amax(np.abs(np.subtract(actual, desired))) < atol


def assert_output_lessthan(func, x_vals, max_y):
    """Assert that func(x) < max_y for all x_vals."""
    y_vals = func(x_vals)
    assert np.amax(np.abs(y_vals)) < max_y


# pylint: disable = too-few-public-methods
class CountCalls:
    """Class decorator for tracking state."""

    # pylint: disable=undefined-variable
    functions: List[CountCalls] = []  # NOQA: F821
    # pylint: enable=undefined-variable

    def __init__(self, func):
        """Initialize the object."""
        update_wrapper(self, func)
        self.func = func
        CountCalls.functions.append(self)
        self.call_count = 0

    def __call__(self, *args):
        """Increment counter each time func is called."""
        self.call_count += 1
        return self.func(*args)


# pylint: enable = too-few-public-methods


def typecheck_multi(item, *args) -> bool:
    """Check if item is instance of anything in *args."""
    return any(isinstance(item, type) for type in args)


def typecheck_number(num):
    """Assert that item is a Number."""
    assert typecheck_multi(num, mp.mpf, float, np.float64, int)


def typecheck_iterable(items: Iterable, *args):
    """Typecheck items in an Iterable.

    Assert each item in items is an instance of something in *args.
    Since all items in numpy arrays share the same type, only the first
    item needs to be checked if items is an array.
    """
    if isinstance(items, np.ndarray):
        assert typecheck_multi(items[0], args)
    assert all(typecheck_multi(item, args) for item in items)


@CountCalls
def trig_func(x_val: mp.mpf) -> mp.mpf:
    """Define a test function requiring high precision for analysis.

    cos(x^2)-sin(x)+x/68
    """
    return mp.cos(mp.power(x_val, 2)) - mp.sin(x_val) + (x_val / 68)


analyzed_trig_func = FuncIntervals(
    func=trig_func,
    x_range=(-47.05, -46.3499),
    zeros_wanted=21,
    crits_wanted=21,
    known_zeros=[-47.038_289_673_236_127, -46.406_755_885_040_056],
)
ANALYZED_TRIG_FUNC_ZEROS = analyzed_trig_func.zeros()
ANALYZED_TRIG_FUNC_CRITS = analyzed_trig_func.crits()
ANALYZED_TRIG_FUNC_POIS = analyzed_trig_func.pois()


def test_analyzedfunc_has_no_throwaways():
    """Ensure that the throwaway overloading functions are removed."""
    assert not hasattr(analyzed_trig_func, "_")


def test_zeroth_derivative_is_itself():
    """Check that nth_derivative(0) returns the unaltered function."""
    assert analyzed_trig_func.nth_derivative(0) == analyzed_trig_func.func


def typecheck_zcp(points):
    """Typecheck functions returning arrays of points.

    Such functions include zeros(), crits(), pois(),
    relative_maxima(), relative_minima().
    """
    assert isinstance(points, np.ndarray)
    typecheck_iterable(points, mp.mpf)


def test_trig_func_has_correct_zeros():
    """Test the correctness of analyzed_trig_func.zeros()."""
    typecheck_zcp(ANALYZED_TRIG_FUNC_ZEROS)
    # approximate accuracy
    np.testing.assert_allclose(
        np.float128(ANALYZED_TRIG_FUNC_ZEROS),
        [
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
        ],
        rtol=EPSILON_1,
    )
    # Does the function evaluate to 0 at its zeros?
    assert_output_lessthan(
        func=analyzed_trig_func.func,
        x_vals=ANALYZED_TRIG_FUNC_ZEROS,
        max_y=3.5692e-19,
    )


def test_trig_func_has_correct_crits():
    """Test the correctness of analyzed_trig_func.crits()."""
    typecheck_zcp(ANALYZED_TRIG_FUNC_CRITS)
    # approximate accuracy
    np.testing.assert_allclose(
        np.float128(ANALYZED_TRIG_FUNC_CRITS),
        [
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
        ],
        rtol=EPSILON_1,
    )
    assert_output_lessthan(
        func=analyzed_trig_func.rooted_first_derivative().func,
        x_vals=ANALYZED_TRIG_FUNC_CRITS,
        max_y=EPSILON_1,
    )


@CountCalls
def sec_der(x_val: Number) -> mp.mpf:
    """Define the actual second derivative."""
    return (
        mp.cos(x_val)
        + (-4 * (mp.power(x_val, 2))) * mp.cos(mp.power(x_val, 2))
        + -2 * mp.sin(mp.power(x_val, 2))
        + mp.sin(x_val)
    )


def assert_trig_func_pois_are_accurate(pois_found: np.ndarray):
    """Test pois() accuracy."""
    assert (
        np.float128(pois_found[3] + 46.944_940_655_832_212_248_274_091_985_22)
        < EPSILON_1
    )
    assert_output_lessthan(
        func=analyzed_trig_func.rooted_second_derivative().func,
        x_vals=ANALYZED_TRIG_FUNC_POIS,
        max_y=EPSILON_1,
    )


def pois_stay_close_when_given_fp2(fp2_zeros):
    """Test pois() when providing second derivative.

    This makes sure that it is possible to provide a second derivative
    to AnalyzedFunnc instances and that it gets used to improve
    accuracy.
    """
    analyzed_trig_func_with_fp2 = FuncIntervals(
        func=trig_func,
        x_range=(-47.05, -46.3499),
        zeros_wanted=21,
        crits_wanted=21,
        known_zeros=[-47.038_289_673_236_127, -46.406_755_885_040_056],
        derivatives={2: sec_der},
    )
    # make sure sec_der() is actually used by tracking its call count
    sec_der_counts_before = sec_der.call_count
    more_exact_pois = analyzed_trig_func_with_fp2.pois()
    assert sec_der.call_count - sec_der_counts_before > 50

    typecheck_zcp(more_exact_pois)
    mpf_assert_allclose(fp2_zeros, more_exact_pois, EPSILON_0)
    mpf_assert_allclose(more_exact_pois, ANALYZED_TRIG_FUNC_POIS, EPSILON_2)


def test_trig_func_has_correct_pois():
    """Test the correctness of analyzed_trig_func.pois().

    First, compare the output with approximate floating-point values.
    Then, compare the output with the pois found from its exact second
    derivative.
    """
    # typechecking
    typecheck_zcp(ANALYZED_TRIG_FUNC_POIS)
    assert_trig_func_pois_are_accurate(ANALYZED_TRIG_FUNC_POIS)
    fp2_zeros = FuncSpecialPts(
        func=sec_der, x_range=(-47.05, -46.35), zeros_wanted=21
    ).zeros()
    mpf_assert_allclose(fp2_zeros, ANALYZED_TRIG_FUNC_POIS, EPSILON_2)
    pois_stay_close_when_given_fp2(fp2_zeros)


def test_trig_func_has_correct_relative_extrema():
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
    np.testing.assert_equal(maxima, ANALYZED_TRIG_FUNC_CRITS[::2])
    np.testing.assert_equal(minima, ANALYZED_TRIG_FUNC_CRITS[1::2])


def test_trig_func_has_correct_abs_max():
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


def test_trig_func_has_correct_abs_min():
    """Test that absolute_minimum() returns correct value."""
    expected_min = analyzed_trig_func.relative_minima()[0]
    np.testing.assert_equal(
        analyzed_trig_func.absolute_minimum(),
        [expected_min, analyzed_trig_func.func(expected_min)],
    )


@CountCalls
def parab_func(x_val: Number) -> mp.mpf:
    """Define a simple parabola.

    It is concave and symmetric about the y-axis.
    """
    return mp.power(x_val, 2) - 4


analyzed_parab = FuncIntervals(
    func=parab_func, x_range=(-8, 8), zeros_wanted=2
)


def test_parabola_has_correct_zeros():
    """Check that analyzed_parab.zeros() returns correct value."""
    np.testing.assert_equal(analyzed_parab.zeros(), np.array([-2, 2]))


def test_parabola_has_correct_crits():
    """Check that analyzed_parab.crits() returns correct value."""
    assert analyzed_parab.crits() == [0]


def test_parabola_has_symmetry():
    """Check analyzed_parab's symmetry functions."""
    assert analyzed_parab.has_symmetry(axis=0)
    np.testing.assert_equal(
        analyzed_parab.vertical_axis_of_symmetry(), analyzed_parab.crits(), [0]
    )
    delattr(analyzed_parab, "plotted_points")
    np.testing.assert_equal(
        analyzed_parab.vertical_axis_of_symmetry(), analyzed_parab.crits(), [0]
    )


@CountCalls
def inc_dec_func(x_val):
    """Define a function to test increasing/decreasing intervals.

    ln(x^2)/x subtly switches from decreasing to increasing at x=-e
    """
    return mp.fdiv(mp.log(mp.power(x_val, 2)), x_val)


analyzed_incdecfunc = FuncIntervals(
    func=inc_dec_func, x_range=(-3, -0.001), crits_wanted=0, zeros_wanted=1
)


def test_interval_helpers_work_correctly():
    """Test many helper functions that FuncIntervals leverages.

    These functions include _make_intervals(), _increasing_intervals(),
    and _decreasing_intervals()/
    """
    points = [-2.0, 8, -3, -4, -9, 12, 18, 4, 0]
    expected_intervals: List = [
        (-2, 8),
        (8, -3),
        (-3, -4),
        (-4, -9),
        (-9, 12),
        (12, 18),
        (18, 4),
        (4, 0),
    ]
    assert _make_intervals(points) == expected_intervals

    def dummy_func(x_val):
        """Return input.

        Used to test _increasing_intervals()
        and _decreasing_intervals().
        """
        return x_val

    assert _increasing_intervals(dummy_func, _make_intervals(points)) == [
        expected_intervals[0],
        expected_intervals[4],
        expected_intervals[5],
    ]
    assert _decreasing_intervals(dummy_func, _make_intervals(points)) == [
        expected_intervals[1],
        expected_intervals[2],
        expected_intervals[3],
        expected_intervals[6],
        expected_intervals[7],
    ]


def test_analyzed_incdecfunc_has_correct_decreasing():
    """Test accuracy of analyzed_incdecfunc.decreasing().

    This works really well because in x_range, incdecfunc decreases
    across (-3, -e). Comparing with an irrational constant really
    pushes the boundaries of the precision of func_analysis.
    """
    mpf_assert_allclose(
        analyzed_incdecfunc.decreasing(), [(-3, mp.fneg(mp.e))], EPSILON_1 / 11
    )


def typecheck_intervals(intervals):
    """Typecheck of all functions with return type List[Interval]."""
    assert isinstance(intervals, List)
    for interval in intervals:
        assert isinstance(interval, tuple)
        typecheck_number(interval[0])
        typecheck_number(interval[1])


def test_analyzed_incdecfunc_has_correct_increasing_decreasing():
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


def test_incdecfunc_has_correct_zeros():
    """Test analyzed_incdecfunc.zeros() returns correct value."""
    assert analyzed_incdecfunc.zeros() == [-1]


def test_call_counting():
    """Check and print call counts for each executed function."""
    assert trig_func.call_count < 2000
    assert trig_func.call_count > 10
    print("\ncall counts\n===========")
    for counted_func in CountCalls.functions:
        print(counted_func.func.__name__ + ": " + str(counted_func.call_count))
