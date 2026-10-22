"""Tests for the three-mode theme system (src/theme.py + app.py integration)."""

import os
import tempfile

import pytest

from src.theme import (
    THEME_OPTIONS,
    THEME_MODES,
    PALETTES,
    SYSTEM,
    LIGHT,
    DARK,
    build_theme_css,
    build_native_bridge_html,
    native_label,
    normalize_mode,
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP_PATH = os.path.join(PROJECT_ROOT, "app.py")


def _srgb_channel(value: float) -> float:
    return value / 12.92 if value <= 0.03928 else ((value + 0.055) / 1.055) ** 2.4


def _relative_luminance(hex_color: str) -> float:
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i:i + 2], 16) / 255.0 for i in (0, 2, 4))
    return 0.2126 * _srgb_channel(r) + 0.7152 * _srgb_channel(g) + 0.0722 * _srgb_channel(b)


def _contrast_ratio(fg: str, bg: str) -> float:
    l1, l2 = _relative_luminance(fg), _relative_luminance(bg)
    hi, lo = max(l1, l2), min(l1, l2)
    return (hi + 0.05) / (lo + 0.05)


# ---------------------------------------------------------------------------
# Pure helper tests
# ---------------------------------------------------------------------------
def test_exactly_three_modes_in_order():
    assert THEME_MODES == [SYSTEM, LIGHT, DARK]
    assert len(THEME_OPTIONS) == 3
    labels = [label for _, label in THEME_OPTIONS]
    assert labels[0].endswith("System Default")
    assert len(set(labels)) == 3


def test_normalize_mode_and_native_labels():
    assert normalize_mode("DARK") == DARK
    assert normalize_mode(" light ") == LIGHT
    assert normalize_mode("system") == SYSTEM
    assert normalize_mode("nonsense") == SYSTEM  # safe default
    assert normalize_mode(None) == SYSTEM
    assert native_label(DARK) == "Dark"
    assert native_label(LIGHT) == "Light"
    assert native_label(SYSTEM) == "System"


def test_light_and_dark_css_are_theme_specific():
    light = build_theme_css(LIGHT)
    dark = build_theme_css(DARK)
    assert "color-scheme: light" in light
    assert "color-scheme: dark" in dark
    assert PALETTES[LIGHT]["surface"] in light
    assert PALETTES[DARK]["surface"] in dark
    assert PALETTES[DARK]["surface"] not in light
    # Component classes are present in both.
    for css in (light, dark):
        for cls in (".metric-card", ".metric-card-danger", ".metric-card-warning",
                    ".badge-blocked", ".badge-allowed", ".overview-card"):
            assert cls in css


def test_component_rules_use_tokens_not_hardcoded_colors():
    """Rule bodies must reference --cad-* tokens (single source of truth)."""
    for css in (build_theme_css(LIGHT), build_theme_css(DARK)):
        # Every status surface resolves through a variable.
        assert "background-color: var(--cad-success-surface)" in css
        assert "background-color: var(--cad-danger-surface)" in css
        assert "background-color: var(--cad-warning-surface)" in css
        assert "background-color: var(--cad-surface)" in css
        # Defensive: cancel any stale gradient from an older stylesheet.
        assert "background-image: none" in css
        assert "var(--cad-" in css


def test_status_surfaces_are_opaque():
    """Regressions: translucent tints turned dark over a dark page surface."""
    for mode in (LIGHT, DARK):
        p = PALETTES[mode]
        for key in ("surface", "success_surface", "danger_surface", "warning_surface"):
            assert p[key].startswith("#"), f"{mode}.{key} must be an opaque hex color"


def test_system_css_follows_os_preference_when_active_theme_unknown():
    css = build_theme_css(SYSTEM)
    assert "prefers-color-scheme: dark" in css
    assert "color-scheme: light dark" in css
    # Both palettes must be present so the media query can switch.
    assert PALETTES[LIGHT]["surface"] in css
    assert PALETTES[DARK]["surface"] in css
    # The media query must re-declare the dark surface token.
    assert f"--cad-surface: {PALETTES[DARK]['surface']};" in css


def test_system_css_uses_active_native_theme_when_known():
    """System Default must match the *actual* active native theme.

    This is the fix for cards following the OS media query while Streamlit's
    native surface followed a different stored preference.
    """
    dark = build_theme_css(SYSTEM, active_theme=DARK)
    light = build_theme_css(SYSTEM, active_theme=LIGHT)

    assert PALETTES[DARK]["surface"] in dark
    assert PALETTES[LIGHT]["surface"] not in dark
    assert PALETTES[LIGHT]["surface"] in light
    assert PALETTES[DARK]["surface"] not in light

    # No OS media query when the active theme is known: the palette is pinned
    # to what Streamlit is actually rendering.
    assert "prefers-color-scheme: dark" not in dark
    assert "prefers-color-scheme: dark" not in light
    assert f"color-scheme: {PALETTES[DARK]['scheme']}" in dark
    assert f"color-scheme: {PALETTES[LIGHT]['scheme']}" in light


def test_system_css_ignores_unknown_active_theme():
    css = build_theme_css(SYSTEM, active_theme="sepia")
    assert "prefers-color-scheme: dark" in css  # falls back to OS media query


def test_native_bridge_targets_streamlit_theme_key_and_guards():
    html = build_native_bridge_html(DARK)
    assert "Dark" in html
    assert "stActiveTheme-" in html
    assert "cadThemeBridgeGuard" in html
    assert "location.reload" in html
    # Avoid reload loops for the system default when nothing is stored.
    assert 'desired === "System" && !hasKey' in html


@pytest.mark.parametrize("mode", [LIGHT, DARK])
def test_status_colors_have_accessible_contrast(mode):
    p = PALETTES[mode]
    # Body text on the neutral surface and on each opaque status surface.
    assert _contrast_ratio(p["text"], p["surface"]) >= 4.5
    assert _contrast_ratio(p["text"], p["success_surface"]) >= 4.5
    assert _contrast_ratio(p["text"], p["danger_surface"]) >= 4.5
    assert _contrast_ratio(p["text"], p["warning_surface"]) >= 4.5
    # Status badges (text is never conveyed by color alone, but must be legible).
    assert _contrast_ratio(p["badge_blocked_fg"], p["badge_blocked_bg"]) >= 4.5
    assert _contrast_ratio(p["badge_allowed_fg"], p["badge_allowed_bg"]) >= 4.5
    # Status foreground accents on their own surfaces.
    assert _contrast_ratio(p["success_fg"], p["success_surface"]) >= 4.5
    assert _contrast_ratio(p["danger_fg"], p["danger_surface"]) >= 4.5
    # Secondary/muted text.
    assert _contrast_ratio(p["muted"], p["surface"]) >= 3.0
    assert _contrast_ratio(p["muted"], p["success_surface"]) >= 3.0
    assert _contrast_ratio(p["muted"], p["danger_surface"]) >= 3.0


# ---------------------------------------------------------------------------
# App integration tests (isolated temp database)
# ---------------------------------------------------------------------------
@pytest.fixture()
def app_test_env(monkeypatch):
    """Points the app at a throwaway database and returns the DB path."""
    from streamlit.testing.v1 import AppTest

    tmpdir = tempfile.mkdtemp(prefix="cad_theme_")
    db_path = os.path.join(tmpdir, "theme.db")
    monkeypatch.setenv("CAD_DB_PATH", db_path)
    return AppTest, db_path


def test_app_starts_with_three_theme_options(app_test_env):
    AppTest, _ = app_test_env
    at = AppTest.from_file(APP_PATH, default_timeout=90).run()
    assert not at.exception
    assert len(at.radio) == 1
    assert at.radio[0].options == [label for _, label in THEME_OPTIONS]
    assert at.radio[0].value == "🖥️ System Default"
    # Three-tab structure is preserved.
    assert len(at.tabs) == 3


def test_theme_selection_persists_across_reruns(app_test_env):
    AppTest, _ = app_test_env
    at = AppTest.from_file(APP_PATH, default_timeout=90).run()

    at.radio[0].set_value("🌙 Dark Mode").run()
    assert not at.exception
    assert at.radio[0].value == "🌙 Dark Mode"
    assert at.query_params.get("theme") == "dark"

    # A plain rerun (no new selection) keeps the choice.
    at.run()
    assert at.radio[0].value == "🌙 Dark Mode"
    assert at.query_params.get("theme") == "dark"

    at.radio[0].set_value("☀️ Light Mode").run()
    assert at.query_params.get("theme") == "light"
    assert at.radio[0].value == "☀️ Light Mode"


def test_theme_choice_survives_reload_via_query_params(app_test_env):
    """A fresh session that receives ?theme=dark should start in Dark mode."""
    AppTest, _ = app_test_env
    at = AppTest.from_file(APP_PATH, default_timeout=90)
    at.query_params["theme"] = "dark"
    at.run()
    assert not at.exception
    assert at.radio[0].value == "🌙 Dark Mode"


def test_theme_change_does_not_reset_memory_or_history(app_test_env):
    AppTest, db_path = app_test_env
    from src.memory_engine import RemediationMemoryEngine, normalize_context
    from src.history_store import IncidentHistoryStore

    # Seed one constraint and one history row in the isolated database.
    engine = RemediationMemoryEngine(db_path=db_path)
    engine.record_negative_constraint(
        failure_condition="DB_CONN_EXHAUSTION",
        observed_context_raw={"cpu": 42.0},
        observed_context_vector=normalize_context({"cpu": 42.0}),
        action="restart_service",
        verification_evidence="e",
        failure_reason="r",
        negative_constraint="c",
        timestamp=0.0,
    )
    IncidentHistoryStore(db_path=db_path).record_incident(_make_record())

    at = AppTest.from_file(APP_PATH, default_timeout=90).run()
    assert not at.exception
    at.radio[0].set_value("🌙 Dark Mode").run()
    assert not at.exception

    reopened = RemediationMemoryEngine(db_path=db_path)
    assert len(reopened.negative_memory) == 1
    assert IncidentHistoryStore(db_path=db_path).count_incidents() == 1


class _Rec:
    def __init__(self):
        self.incident_id = 1
        self.condition = "DB_CONN_EXHAUSTION"
        self.controller_name = "Proposed (Negative+Positive Memory)"
        self.telemetry_raw = {"cpu": 42.0}
        self.telemetry_vector = [0.4, 0.6, 1.0, 0.85, 0.4]
        self.attempts = 2
        self.tried_actions = ["restart_service", "connection_capacity_remediation"]
        self.successful_action = "connection_capacity_remediation"
        self.blocked_actions = []
        self.first_attempt_success = False
        self.repeated_failures = 0
        self.mttr_seconds = 20.0
        self.toxic_failures_encountered = 0
        self.timestamp = 1.0
        self.execution_trace = []


def _make_record():
    return _Rec()
