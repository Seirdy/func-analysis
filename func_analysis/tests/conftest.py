"""Fixtures to represent sample AnalyzedFunc instances."""

import pytest
from func_analysis.tests.call_counting import AnalyzedFuncCounted
from func_analysis.tests.funcs_to_analyze import (
    inc_dec_func,
    parab_func,
    sec_der,
    trig_func,
)


@pytest.fixture
def analyzed_trig_func():
    """Fixture for an AnalyzedFunc describing trig_func."""
    analyzed_func = AnalyzedFuncCounted(
        func=trig_func,
        x_range=(-47.05, -46.3499),
        zeros_wanted=21,
        crits_wanted=21,
        known_zeros=[-47.038289673236127, -46.406755885040056],
    )
    return analyzed_func


@pytest.fixture
def fp2_zeros():
    """Fixture for an AnalyzedFuncCounted describing sec_der."""
    return AnalyzedFuncCounted(
        func=sec_der, x_range=(-47.05, -46.35), zeros_wanted=21
    )


@pytest.fixture
def analyzed_parab():
    """Fixture for an AnalyzedFuncCounted describing parab_func."""
    return AnalyzedFuncCounted(
        func=parab_func, x_range=(-8, 8), zeros_wanted=2
    )


@pytest.fixture
def analyzed_incdecfunc():
    """Fixture for an AnalyzedFuncCounted describing inc_dec_func."""
    return AnalyzedFuncCounted(
        func=inc_dec_func, x_range=(-3, -0.001), crits_wanted=1, zeros_wanted=1
    )
