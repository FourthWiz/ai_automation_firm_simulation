"""T-09 + T-10: app.py unit tests.

Tests _PARAM_FIELDS index positions and enable_training_delay checkbox default.
Uses AppTest for live Streamlit widget checks.
"""
import pytest

from app import _PARAM_FIELDS, params_to_key
from firm_ai_abm.config import FirmParams


def _active_key():
    """Build a sample key from default FirmParams for index tests."""
    p = FirmParams()
    return params_to_key(p, 0)


def test_param_fields_index_positions():
    """T-09: Verify _PARAM_FIELDS index contract."""
    key = _active_key()
    assert key[0] == FirmParams().N, f"key[0] should be N, got {key[0]}"
    assert key[20] == FirmParams().T_review, f"key[20] should be T_review, got {key[20]}"
    assert key[21] == FirmParams().firing_threshold, f"key[21] should be firing_threshold, got {key[21]}"
    assert key[-1] == 0, f"key[-1] should be seed=0, got {key[-1]}"


def test_enable_training_delay_checkbox_defaults_true():
    """T-09: dashboard checkbox for enable_training_delay defaults to True (UX default)."""
    try:
        from streamlit.testing.v1 import AppTest
    except ImportError:
        pytest.skip("streamlit AppTest not available")

    at = AppTest.from_file("app.py", default_timeout=30)
    at.run()

    # Find the enable_training_delay checkbox
    checkboxes = {cb.key: cb for cb in at.checkbox}
    assert "enable_training_delay" in checkboxes, (
        "enable_training_delay checkbox not found in sidebar. "
        f"Available checkboxes: {list(checkboxes.keys())}"
    )
    assert checkboxes["enable_training_delay"].value is True, (
        "enable_training_delay checkbox should default to True in the dashboard "
        "(FirmParams default is False — deliberate two-defaults seam per D-01)."
    )
