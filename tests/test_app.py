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
    """T-16: cache key has exactly 35 elements; _PARAM_FIELDS has exactly 34.

    Updated from 30/31 to 34/35 after beta-dist-task-attrs added 4 fields:
    alpha_mean (30), alpha_concentration (31), beta_mean (32), beta_concentration (33).
    seed is still key[-1] at position 34.
    """
    assert len(_PARAM_FIELDS) == 34, (
        f"Expected 34 fields in _PARAM_FIELDS, got {len(_PARAM_FIELDS)}"
    )
    key = params_to_key(FirmParams(), 0)
    assert len(key) == 35, (
        f"Expected 35-tuple from params_to_key, got {len(key)}"
    )
    # Spot-check: index 26 = enable_hiring (unchanged), 27 = enable_replenish_hiring,
    # 30 = alpha_mean, 33 = beta_concentration, -1 = seed at position 34.
    assert key[26] == FirmParams().enable_hiring, f"key[26] should be enable_hiring, got {key[26]}"
    assert key[27] == FirmParams().enable_replenish_hiring, f"key[27] should be enable_replenish_hiring, got {key[27]}"
    assert key[30] == FirmParams().alpha_mean, f"key[30] should be alpha_mean, got {key[30]}"
    assert key[33] == FirmParams().beta_concentration, f"key[33] should be beta_concentration, got {key[33]}"
    assert key[-1] == 0, f"key[-1] should be seed=0, got {key[-1]}"


def test_hiring_mode_radio_defaults_off():
    """D-01: hiring_mode radio replaces two checkboxes; defaults to 'off'."""
    try:
        from streamlit.testing.v1 import AppTest
    except ImportError:
        pytest.skip("streamlit AppTest not available")

    at = AppTest.from_file("app.py", default_timeout=30).run()
    radios = {r.key: r for r in at.radio}
    assert "hiring_mode" in radios, (
        f"hiring_mode radio not found. Available radio keys: {list(radios.keys())}"
    )
    assert radios["hiring_mode"].value == "off", (
        f"hiring_mode should default to 'off', got {radios['hiring_mode'].value}"
    )


def test_replenish_hiring_toggle_changes_cache_key():
    """D-01: toggling hiring_mode to enable_replenish_hiring maps correctly via DRAFT_PARAMS_DEBUG.

    5-step approach: radio toggle → DRAFT_PARAMS_DEBUG read (inline mapping) → tuple-position check.
    A typo swapping enable_hiring_val and enable_replenish_hiring_val in _build_controls
    would produce dp_after.enable_hiring is True and fail Step 4.
    """
    try:
        from streamlit.testing.v1 import AppTest
    except ImportError:
        pytest.skip("streamlit AppTest not available")

    at = AppTest.from_file("app.py", default_timeout=30).run()

    # Step 1: assert default radio state
    radios = {r.key: r for r in at.radio}
    assert "hiring_mode" in radios, (
        f"hiring_mode radio not found. Keys: {list(radios.keys())}"
    )
    assert radios["hiring_mode"].value == "off"

    # Step 2: read default DRAFT_PARAMS_DEBUG — inline mapping yields both False
    dp_default = at.session_state["DRAFT_PARAMS_DEBUG"]
    assert dp_default.enable_hiring is False, (
        f"default enable_hiring should be False, got {dp_default.enable_hiring}"
    )
    assert dp_default.enable_replenish_hiring is False, (
        f"default enable_replenish_hiring should be False, got {dp_default.enable_replenish_hiring}"
    )

    # Step 3: toggle to enable_replenish_hiring and re-run
    at.radio(key="hiring_mode").set_value("enable_replenish_hiring").run()

    # Step 4: read post-toggle DRAFT_PARAMS_DEBUG — verifies _build_controls inline mapping
    dp_after = at.session_state["DRAFT_PARAMS_DEBUG"]
    assert dp_after.enable_hiring is False, (
        f"post-toggle enable_hiring should be False, got {dp_after.enable_hiring}"
    )
    assert dp_after.enable_replenish_hiring is True, (
        f"post-toggle enable_replenish_hiring should be True, got {dp_after.enable_replenish_hiring}"
    )

    # Step 5: verify tuple position invariant via params_to_key
    from app import params_to_key
    key_after = params_to_key(dp_after, 0)
    assert key_after[26] is False, f"key[26] (enable_hiring) should be False, got {key_after[26]}"
    assert key_after[27] is True, f"key[27] (enable_replenish_hiring) should be True, got {key_after[27]}"
