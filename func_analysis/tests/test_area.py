"""Test area-finding methods."""

from func_analysis.tests import constants
from func_analysis.tests.testing_utils import calculate_error


def test_trig_func_signed_area(analyzed_trig_func):
    """Test definite integral of trig_func is accurate."""
    error = calculate_error(
        analyzed_trig_func.signed_area, constants.TRIG_FUNC_SIGNED_AREA
    )
    assert error < 5.81425e-5


def test_trig_func_unsigned_area(analyzed_trig_func):
    """Test definite integral of trig_func is accurate."""
    error = calculate_error(
        analyzed_trig_func.unsigned_area, constants.TRIG_FUNC_UNSIGNED_AREA
    )
    assert error < 5.55874e-4
