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
    """T-16: cache key has exactly 38 elements; _PARAM_FIELDS has exactly 37.

    Updated from 36/37 to 37/38 after splitting the prior field into dp_prior_alpha (index 35)
    and dp_prior_beta (index 36):
    max_hire_per_step (29), hire_delay_periods (30), alpha_mean (31),
    alpha_concentration (32), beta_mean (33), beta_concentration (34),
    dp_prior_alpha (35), dp_prior_beta (36). seed is still key[-1] at position 37.
    """
    assert len(_PARAM_FIELDS) == 37, (
        f"Expected 37 fields in _PARAM_FIELDS, got {len(_PARAM_FIELDS)}"
    )
    key = params_to_key(FirmParams(), 0)
    assert len(key) == 38, (
        f"Expected 38-tuple from params_to_key, got {len(key)}"
    )
    # Spot-check: index 26 = enable_hiring (unchanged), 27 = enable_replenish_hiring,
    # 29 = max_hire_per_step, 31 = alpha_mean, 34 = beta_concentration,
    # 35 = dp_prior_alpha, 36 = dp_prior_beta, -1 = seed (position 37).
    assert key[26] == FirmParams().enable_hiring, f"key[26] should be enable_hiring, got {key[26]}"
    assert key[27] == FirmParams().enable_replenish_hiring, f"key[27] should be enable_replenish_hiring, got {key[27]}"
    assert key[29] == FirmParams().max_hire_per_step, f"key[29] should be max_hire_per_step, got {key[29]}"
    assert key[31] == FirmParams().alpha_mean, f"key[31] should be alpha_mean, got {key[31]}"
    assert key[34] == FirmParams().beta_concentration, f"key[34] should be beta_concentration, got {key[34]}"
    assert key[35] == FirmParams().dp_prior_alpha, f"key[35] should be dp_prior_alpha, got {key[35]}"
    assert key[36] == FirmParams().dp_prior_beta, f"key[36] should be dp_prior_beta, got {key[36]}"
    assert key[-1] == 0, f"key[-1] should be seed=0, got {key[-1]}"


def test_hiring_mode_radio_defaults_enable_hiring():
    """hiring_mode radio defaults to 'enable_hiring' (index=1, per Scenario B recalibration 99ddaea)."""
    try:
        from streamlit.testing.v1 import AppTest
    except ImportError:
        pytest.skip("streamlit AppTest not available")

    at = AppTest.from_file("app.py", default_timeout=30).run()
    radios = {r.key: r for r in at.radio}
    assert "hiring_mode" in radios, (
        f"hiring_mode radio not found. Available radio keys: {list(radios.keys())}"
    )
    assert radios["hiring_mode"].value == "enable_hiring", (
        f"hiring_mode should default to 'enable_hiring', got {radios['hiring_mode'].value}"
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

    # Step 1: assert default radio state (enable_hiring per Scenario B recalibration 99ddaea)
    radios = {r.key: r for r in at.radio}
    assert "hiring_mode" in radios, (
        f"hiring_mode radio not found. Keys: {list(radios.keys())}"
    )
    assert radios["hiring_mode"].value == "enable_hiring"

    # Step 2: read default DRAFT_PARAMS_DEBUG — inline mapping: enable_hiring=True, replenish=False
    dp_default = at.session_state["DRAFT_PARAMS_DEBUG"]
    assert dp_default.enable_hiring is True, (
        f"default enable_hiring should be True (Scenario B), got {dp_default.enable_hiring}"
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
