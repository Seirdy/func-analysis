#!/usr/bin/env python3.7
# -*- coding: utf-8 -*-
# pylint: disable=comparison-with-callable
"""Tests func_analysis.

This deliberately uses a function requiring a high degree of precision
"""
from __future__ import annotations

from typing import Iterable, List, Union

import mpmath as mp
import numpy as np

from func_analysis import (
    FuncIntervals,
    FuncSpecialPts,
    decreasing_intervals,
    increasing_intervals,
    make_intervals,
)

BuiltinFloat = Union[np.float128, np.float64, float]
Number = Union[mp.mpf, BuiltinFloat]

EPSILON_0 = 1e-20
EPSILON_1 = 3.05e-15
EPSILON_2 = 1.196_789_1e-6


def mpf_maxerror(arr1, arr2):
    return np.amax(np.abs(np.subtract(arr1, arr2)))


# pylint: disable = too-few-public-methods
class CountCalls:
    """Class decorator for tracking state."""

    # pylint: disable=undefined-variable
    functions: List[CountCalls] = []  # NOQA: F821
    # pylint: enable=undefined-variable

    def __init__(self, func):
        """Initialize the object."""
        self.func = func
        CountCalls.functions.append(self)
        self.call_count = 0
        self.__name__ = self.func.__name__

    def __call__(self, *args):
        """Increment counter each time func is called."""
        self.call_count += 1
        return self.func(*args)


# pylint: enable = too-few-public-methods


def typecheck_multi(item, *args) -> bool:
    """Assert that item is instance of anything in *args."""
    return any(isinstance(item, type) for type in args)


def typecheck_isnumber(num):
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
analyzed_trig_func_zeros = analyzed_trig_func.zeros()
analyzed_trig_func_crits = analyzed_trig_func.crits()
analyzed_trig_func_pois = analyzed_trig_func.pois()


def test_zeroth_derivative_is_itself():
    assert analyzed_trig_func.nth_derivative(0) == analyzed_trig_func.func


def typecheck_zcp(points):
    assert isinstance(points, np.ndarray)
    typecheck_iterable(points, mp.mpf)


def test_trig_func_has_correct_zeros():
    typecheck_zcp(analyzed_trig_func_zeros)


def test_trig_func_has_correct_crits():
    # typechecks
    typecheck_zcp(analyzed_trig_func_crits)
    # approximate accuracy
    np.testing.assert_allclose(
        np.float128(analyzed_trig_func_crits),
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


@CountCalls
def sec_der(x_val: Number) -> mp.mpf:
    """Define the actual second derivative."""
    return (
        mp.cos(x_val)
        + (-4 * (mp.power(x_val, 2))) * mp.cos(mp.power(x_val, 2))
        + -2 * mp.sin(mp.power(x_val, 2))
        + mp.sin(x_val)
    )


def trig_func_pois_match_imprecise_expectation(pois_found: np.ndarray):
    assert (
        np.float128(pois_found[3] + 46.944_940_655_832_212_248_274_091_985_22)
        < EPSILON_1
    )


def pois_stay_close_when_given_fp2(fp2_zeros):
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
    assert mpf_maxerror(fp2_zeros, more_exact_pois) < EPSILON_0
    assert mpf_maxerror(more_exact_pois, analyzed_trig_func_pois) < EPSILON_2


def test_trig_func_has_correct_pois():
    """Test the correctness of analyzed_trig_func.pois().

    First, compare the output with approximate floating-point values.
    Then, compare the output with the pois found from its exact second
    derivative. Try to use Hypothesis for typechecking.
    """
    # typechecking
    typecheck_zcp(analyzed_trig_func_pois)
    trig_func_pois_match_imprecise_expectation(analyzed_trig_func_pois)
    fp2_zeros = FuncSpecialPts(
        func=sec_der, x_range=(-47.05, -46.35), zeros_wanted=21
    ).zeros()
    assert mpf_maxerror(fp2_zeros, analyzed_trig_func_pois) < EPSILON_2
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
    np.testing.assert_equal(maxima, analyzed_trig_func_crits[::2])
    np.testing.assert_equal(minima, analyzed_trig_func_crits[1::2])


def test_trig_func_has_correct_abs_max():
    """Test correctness of analyzed_trig_func.absolute_maximum().

    First, make sure that its approximation is correct. Then, compare
    the exact values.
    """
    trig_abs_max = analyzed_trig_func.absolute_maximum()
    appproximate_expected_max = [-46.355_597_936_762_38, 1.013_176_643_861_527]
    np.testing.assert_allclose(
        np.float128(trig_abs_max), appproximate_expected_max
    )
    exact_expected_max = analyzed_trig_func.relative_maxima()[10]
    np.testing.assert_equal(
        trig_abs_max,
        [exact_expected_max, analyzed_trig_func.func(exact_expected_max)],
    )


def test_trig_func_has_correct_abs_min():
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


ANALYZED_PARAB = FuncIntervals(
    func=parab_func, x_range=(-8, 8), zeros_wanted=2
)


def test_parabola_has_correct_zeros():
    np.testing.assert_equal(ANALYZED_PARAB.zeros(), np.array([-2, 2]))


def test_parabola_has_correct_crits():
    assert ANALYZED_PARAB.crits() == [0]


def test_parabola_has_symmetry():
    assert ANALYZED_PARAB.has_symmetry(axis=0)
    np.testing.assert_equal(
        ANALYZED_PARAB.vertical_axis_of_symmetry(), ANALYZED_PARAB.crits(), [0]
    )
    delattr(ANALYZED_PARAB, "plotted_points")
    np.testing.assert_equal(
        ANALYZED_PARAB.vertical_axis_of_symmetry(), ANALYZED_PARAB.crits(), [0]
    )


@CountCalls
def inc_dec_func(x_val):
    """Define a function to test increasing/decreasing intervals.

    ln(x^2)/x subtly switches from decreasing to increasing at x=-e
    """
    return mp.fdiv(mp.log(mp.power(x_val, 2)), x_val)


ANALYZED_INCDECFUNC = FuncIntervals(
    func=inc_dec_func, x_range=(-3, -0.001), crits_wanted=0, zeros_wanted=1
)


def test_interval_helpers_work_correctly():
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
    assert make_intervals(points) == expected_intervals

    def dummy_func(x_val):
        return x_val

    assert increasing_intervals(dummy_func, make_intervals(points)) == [
        expected_intervals[0],
        expected_intervals[4],
        expected_intervals[5],
    ]
    assert decreasing_intervals(dummy_func, make_intervals(points)) == [
        expected_intervals[1],
        expected_intervals[2],
        expected_intervals[3],
        expected_intervals[6],
        expected_intervals[7],
    ]


def test_analyzed_incdecfunc_has_correct_decreasing():
    assert (
        mpf_maxerror(ANALYZED_INCDECFUNC.decreasing(), [(-3, mp.fneg(mp.e))])
        < EPSILON_1 / 11
    )


def typecheck_intervals(intervals):
    assert isinstance(intervals, List)
    for interval in intervals:
        assert isinstance(interval, tuple)
        typecheck_isnumber(interval[0])
        typecheck_isnumber(interval[1])


def test_analyzed_incdecfunc_has_correct_increasing():
    analyzed_incdecfunc_increasing = ANALYZED_INCDECFUNC.increasing()
    analyzed_incdecfunc_decreasing = ANALYZED_INCDECFUNC.decreasing()
    typecheck_intervals(analyzed_incdecfunc_increasing)
    typecheck_intervals(analyzed_incdecfunc_decreasing)
    # assert isinstance(analyzed_incdecfunc_increasing[0], tuple)
    assert (
        analyzed_incdecfunc_increasing[0][0]
        == analyzed_incdecfunc_decreasing[0][1]
    )
    assert (
        mpf_maxerror(
            ANALYZED_INCDECFUNC.increasing(), [(mp.fneg(mp.e), -0.001)]
        )
        < EPSILON_1 / 10
    )


def test_correct_incdecfunc_zeros():
    assert ANALYZED_INCDECFUNC.zeros() == [-1]


# def test_trig_func_has_correct_concavity():
#     print(analyzed_trig_func.concave())


# def test_trig_func_has_correct_convexity():
#     print(analyzed_trig_func.convex())


# @CountCalls
# def poi_func(x_val):
#     """Define a function with no points of inflection.

#     Its second and third derivatives at the origin are zero.
#     """
#     return x_val ** 3 / (x_val + 2)


# AnalyzedPoiFunc = FuncIntervals(
#     func=poi_func,
#     x_range=(-2 + 1e-6, 6),
#     zeros_wanted=1,
#     known_zeros=[0],
#     crits_wanted=1,
#     pois_wanted=1,
# )

# POI_ZEROS = AnalyzedPoiFunc.zeros()
# POI_POIS = AnalyzedPoiFunc.pois()
# POI = POI_POIS[0]
# X_VALS = mp.linspace(POI * (1 - 1e-6), POI * (1 + 1e-6), 11)
# assert X_VALS.index(POI) == 5
# Y_VALS = AnalyzedPoiFunc.rooted_first_derivative().func(X_VALS)
# print(AnalyzedPoiFunc.rooted_first_derivative().func(0))
# # print(np.fabs(Y_VALS))
# print(np.argmin(np.fabs(Y_VALS)))

# print(AnalyzedPoiFunc.rooted_first_derivative().func(POI_POIS))
# assert POI_ZEROS.size, POI_POIS.size == 1
# assert POI_ZEROS[0] < EPSILON_2


def test_call_counting():
    """Check and print call counts for each executed function."""
    assert trig_func.call_count < 1700
    assert trig_func.call_count > 10
    print("\ncall counts\n===========")
    for counted_func in CountCalls.functions:
        print(counted_func.func.__name__ + ": " + str(counted_func.call_count))
