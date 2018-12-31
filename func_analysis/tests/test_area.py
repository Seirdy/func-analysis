"""Test area-finding methods."""

from func_analysis.tests import constants
from func_analysis.tests.testing_utils import assert_error_lessthan


def test_trig_func_signed_area(analyzed_trig_func):
    """Test definite integral of trig_func is accurate."""
    assert_error_lessthan(
        analyzed_trig_func.signed_area,
        constants.TRIG_FUNC_SIGNED_AREA,
        5.81425e-5,
    )


def test_trig_func_unsigned_area(analyzed_trig_func):
    """Test definite integral of trig_func is accurate."""
    assert_error_lessthan(
        analyzed_trig_func.unsigned_area,
        constants.TRIG_FUNC_UNSIGNED_AREA,
        5.55874e-4,
    )
