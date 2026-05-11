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


def test_cache_key_length():
    """T-12: cache key has exactly 28 elements; _PARAM_FIELDS has exactly 27."""
    assert len(_PARAM_FIELDS) == 27, (
        f"Expected 27 fields in _PARAM_FIELDS, got {len(_PARAM_FIELDS)}"
    )
    key = params_to_key(FirmParams(), 0)
    assert len(key) == 28, (
        f"Expected 28-tuple from params_to_key, got {len(key)}"
    )
    # Spot-check: index 26 = enable_hiring, index 27 = seed
    assert key[26] == FirmParams().enable_hiring, f"key[26] should be enable_hiring, got {key[26]}"
    assert key[-1] == 0, f"key[-1] should be seed=0, got {key[-1]}"


def test_enable_hiring_checkbox_defaults_false():
    """T-12: enable_hiring checkbox defaults to False (dormant/opt-in semantics)."""
    try:
        from streamlit.testing.v1 import AppTest
    except ImportError:
        pytest.skip("streamlit AppTest not available")

    at = AppTest.from_file("app.py", default_timeout=30)
    at.run()

    checkboxes = {cb.key: cb for cb in at.checkbox}
    assert "enable_hiring" in checkboxes, (
        "enable_hiring checkbox not found in sidebar. "
        f"Available checkboxes: {list(checkboxes.keys())}"
    )
    assert checkboxes["enable_hiring"].value is False, (
        "enable_hiring checkbox should default to False (opt-in per D-02)."
    )
