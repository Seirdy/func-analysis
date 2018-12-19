#!/usr/bin/env python3.7
# -*- coding: utf-8 -*-
# pylint: disable=comparison-with-callable
"""Tests interval-finding algos in func_analysis.

These include intervals of increase/decrease and concavity/convexity.

This deliberately uses a function requiring a high degree of precision
"""

import mpmath as mp
import numpy as np

from .._util import make_intervals
from .helpers import mpf_assert_allclose, typecheck_intervals

EPSILON_0 = 1e-20
EPSILON_1 = 3.05e-15
EPSILON_2 = 1.196_789_1e-6


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
