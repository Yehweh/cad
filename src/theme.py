"""
Theme system for the CAD dashboard.

Streamlit 1.66 has no Python API to change the *native* app theme at runtime.
Internally the browser resolves the active theme (light / dark / "Use system
setting") and persists the user's choice in ``localStorage`` under a
versioned key (``stActiveTheme-<pathname>-vN``), while "System" tracks the OS
via ``prefers-color-scheme``.

This module provides two complementary, best-effort mechanisms:

1. :func:`build_theme_css` - a *single* set of component rules written entirely
   against ``--cad-*`` design tokens, plus the token values for the requested
   mode. Status surfaces are **opaque** per palette so a card never depends on,
   or blends with, the surrounding page background (an earlier revision used
   translucent ``rgba`` tints, which turned dark when composited over a dark
   surface).

   The palette for "system" mode is taken from the **actual active native
   theme** (``st.context.theme['type']``), not from the OS
   ``prefers-color-scheme`` media query. This is the important consistency fix:
   the custom cards and Streamlit's native surface must agree. Streamlit's
   native theme can differ from the OS (a stored preference, or a choice made
   in Streamlit's own Settings menu), and keying the cards to the OS media
   query made them disagree with the page. The media query is kept only as a
   fallback for the rare case where the active theme cannot be determined.

2. :func:`build_native_bridge_html` - a tiny, guarded browser-side bridge
   (rendered through ``st.iframe``) that mirrors the selected mode into
   Streamlit's own ``localStorage`` theme key and, only when it actually
   changes, triggers a reload so Streamlit's native widgets and charts switch
   too. If the internal key ever changes, the bridge degrades to a no-op and
   the CSS from (1) still applies.

Both builders are pure functions (no Streamlit imports) so they are unit
testable without a browser.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

# Canonical mode ids.
SYSTEM = "system"
LIGHT = "light"
DARK = "dark"

# Sidebar options: (mode id, label). Order is the display order.
THEME_OPTIONS: List[Tuple[str, str]] = [
    (SYSTEM, "🖥️ System Default"),
    (LIGHT, "☀️ Light Mode"),
    (DARK, "🌙 Dark Mode"),
]
THEME_LABELS: Dict[str, str] = {mode: label for mode, label in THEME_OPTIONS}
THEME_MODES: List[str] = [mode for mode, _ in THEME_OPTIONS]

# Streamlit internally stores the theme name under this value.
_NATIVE_NAMES: Dict[str, str] = {SYSTEM: "System", LIGHT: "Light", DARK: "Dark"}

# Token keys that are emitted as CSS custom properties. ``scheme`` is special:
# it becomes the ``color-scheme`` property rather than a ``--cad-*`` variable.
_TOKEN_KEYS = (
    "surface",
    "surface_alt",
    "text",
    "muted",
    "border",
    "primary",
    "radius",
    "shadow",
    "success",
    "success_surface",
    "success_border",
    "success_fg",
    "danger",
    "danger_surface",
    "danger_border",
    "danger_fg",
    "warning",
    "warning_surface",
    "warning_border",
    "warning_fg",
    "badge_blocked_bg",
    "badge_blocked_fg",
    "badge_allowed_bg",
    "badge_allowed_fg",
)


def normalize_mode(value: object) -> str:
    """Coerces ``value`` to a known mode id, defaulting to ``system``."""
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in (SYSTEM, LIGHT, DARK):
            return lowered
    return SYSTEM


def native_label(mode: str) -> str:
    """Returns the label Streamlit's client uses for ``mode``."""
    return _NATIVE_NAMES[normalize_mode(mode)]


# ---------------------------------------------------------------------------
# Design tokens
# ---------------------------------------------------------------------------
# Every value is opaque so components never depend on the page background.
# Status surfaces are subtle, readable tints rather than saturated fills.
PALETTES: Dict[str, Dict[str, str]] = {
    LIGHT: {
        "scheme": "light",
        "surface": "#ffffff",
        "surface_alt": "#f6f8fb",
        "text": "#1f2430",
        "muted": "#5b6472",
        "border": "#d8dee7",
        "primary": "#2563eb",
        "radius": "10px",
        "shadow": "0 1px 2px rgba(15, 23, 42, 0.08), 0 1px 3px rgba(15, 23, 42, 0.06)",
        "success": "#1a7f37",
        "success_surface": "#e9f6ee",
        "success_border": "#b9e0c6",
        "success_fg": "#0f5132",
        "danger": "#c0392b",
        "danger_surface": "#fdecea",
        "danger_border": "#f2c4bf",
        "danger_fg": "#8a2a20",
        "warning": "#9a6700",
        "warning_surface": "#fff5e0",
        "warning_border": "#ecdab2",
        "warning_fg": "#7a5200",
        "badge_blocked_bg": "#c0392b",
        "badge_blocked_fg": "#ffffff",
        "badge_allowed_bg": "#1a7f37",
        "badge_allowed_fg": "#ffffff",
    },
    DARK: {
        "scheme": "dark",
        "surface": "#1a1d24",
        "surface_alt": "#161a20",
        "text": "#e8eaed",
        "muted": "#9aa3b2",
        "border": "#2f3541",
        "primary": "#60a5fa",
        "radius": "10px",
        "shadow": "0 1px 3px rgba(0, 0, 0, 0.45)",
        "success": "#3fb950",
        "success_surface": "#14251a",
        "success_border": "#265c34",
        "success_fg": "#8ce0aa",
        "danger": "#f85149",
        "danger_surface": "#2a1416",
        "danger_border": "#6e2a28",
        "danger_fg": "#ff9d96",
        "warning": "#d29922",
        "warning_surface": "#2a2110",
        "warning_border": "#6b5320",
        "warning_fg": "#f0c86a",
        "badge_blocked_bg": "#f85149",
        "badge_blocked_fg": "#1b0d0c",
        "badge_allowed_bg": "#3fb950",
        "badge_allowed_fg": "#08130b",
    },
}


def _token_block(palette: Dict[str, str], indent: str = "    ") -> str:
    """Renders the ``--cad-*`` custom-property declarations for a palette."""
    lines = [f"{indent}--cad-{key.replace('_', '-')}: {palette[key]};" for key in _TOKEN_KEYS]
    return "\n".join(lines)


def _component_rules(indent: str = "        ") -> str:
    """
    Returns the component rules. Colors come exclusively from ``--cad-*``
    tokens, so there is exactly one copy of each rule and no hardcoded colors.
    """
    i = indent
    return f"""
{i}.metric-card, {i}.metric-card-danger, {i}.metric-card-warning {{
{i}    background-color: var(--cad-success-surface);
{i}    background-image: none;
{i}    color: var(--cad-text);
{i}    border: 1px solid var(--cad-success-border);
{i}    border-left: 5px solid var(--cad-success);
{i}    border-radius: var(--cad-radius);
{i}    padding: 14px 16px;
{i}    margin-bottom: 12px;
{i}    box-shadow: var(--cad-shadow);
{i}}}
{i}.metric-card-danger {{
{i}    background-color: var(--cad-danger-surface);
{i}    border-color: var(--cad-danger-border);
{i}    border-left-color: var(--cad-danger);
{i}}}
{i}.metric-card-warning {{
{i}    background-color: var(--cad-warning-surface);
{i}    border-color: var(--cad-warning-border);
{i}    border-left-color: var(--cad-warning);
{i}}}
{i}.metric-card b, {i}.metric-card-danger b, {i}.metric-card-warning b {{
{i}    color: var(--cad-text);
{i}}}
{i}.metric-card small, {i}.metric-card-danger small, {i}.metric-card-warning small {{
{i}    color: var(--cad-muted);
{i}}}
{i}.badge-blocked, {i}.badge-allowed {{
{i}    display: inline-block;
{i}    padding: 3px 8px;
{i}    border-radius: 4px;
{i}    font-size: 12px;
{i}    font-weight: 700;
{i}    letter-spacing: 0.02em;
{i}}}
{i}.badge-blocked {{
{i}    background-color: var(--cad-badge-blocked-bg);
{i}    color: var(--cad-badge-blocked-fg);
{i}}}
{i}.badge-allowed {{
{i}    background-color: var(--cad-badge-allowed-bg);
{i}    color: var(--cad-badge-allowed-fg);
{i}}}
{i}.overview-card {{
{i}    background-color: var(--cad-surface);
{i}    background-image: none;
{i}    color: var(--cad-text);
{i}    border: 1px solid var(--cad-border);
{i}    border-left: 5px solid var(--cad-primary);
{i}    border-radius: var(--cad-radius);
{i}    padding: 14px 16px;
{i}    margin-bottom: 10px;
{i}    box-shadow: var(--cad-shadow);
{i}}}
{i}.overview-card--primary {{ border-left-color: var(--cad-primary); }}
{i}.overview-card--success {{ border-left-color: var(--cad-success); }}
{i}.overview-card--warning {{ border-left-color: var(--cad-warning); }}
{i}.overview-card--danger {{ border-left-color: var(--cad-danger); }}
{i}.overview-label {{
{i}    font-size: 0.75rem;
{i}    text-transform: uppercase;
{i}    letter-spacing: 0.05em;
{i}    color: var(--cad-muted);
{i}}}
{i}.overview-value {{
{i}    font-size: 1.55rem;
{i}    font-weight: 700;
{i}    line-height: 1.25;
{i}    color: var(--cad-text);
{i}}}
{i}.overview-sub {{
{i}    font-size: 0.78rem;
{i}    color: var(--cad-muted);
{i}}}
{i}.cad-caption {{
{i}    font-size: 0.82rem;
{i}    color: var(--cad-muted);
{i}}}
""".rstrip()


def build_theme_css(mode: str, active_theme: "str | None" = None) -> str:
    """
    Returns a ``<style>`` bundle for the requested mode.

    ``active_theme`` is the *actual* native theme Streamlit is rendering
    (``st.context.theme['type']``, i.e. ``"light"`` or ``"dark"``).

    - ``light`` / ``dark``: a single explicit palette; the user's explicit
      choice overrides the system preference.
    - ``system`` with a known ``active_theme``: the palette matching the active
      native theme. This is the fix for the divergence where the custom cards
      used the OS ``prefers-color-scheme`` while Streamlit's native surface used
      a *different* stored preference (e.g. a stale ``"Dark"`` left from the
      earlier forced-dark config). Keying the cards to the active theme keeps
      the two layers consistent.
    - ``system`` without an ``active_theme`` (e.g. bare/unit context): fall back
      to the light palette plus a ``prefers-color-scheme: dark`` media query so
      the components still follow the OS live.
    """
    mode = normalize_mode(mode)
    lines = ["<style>"]

    if mode == SYSTEM:
        resolved = active_theme if active_theme in (LIGHT, DARK) else None
        if resolved is not None:
            palette = PALETTES[resolved]
            lines.append(f"        :root {{ color-scheme: {palette['scheme']}; }}")
            lines.append("        :root {")
            lines.append(_token_block(palette))
            lines.append("        }")
            lines.append(_component_rules())
        else:
            lines.append("        :root { color-scheme: light dark; }")
            lines.append("        :root {")
            lines.append(_token_block(PALETTES[LIGHT]))
            lines.append("        }")
            lines.append(_component_rules())
            lines.append("        @media (prefers-color-scheme: dark) {")
            lines.append("            :root {")
            lines.append(_token_block(PALETTES[DARK], indent="                "))
            lines.append("            }")
            lines.append("        }")
    else:
        palette = PALETTES[LIGHT] if mode == LIGHT else PALETTES[DARK]
        lines.append(f"        :root {{ color-scheme: {palette['scheme']}; }}")
        lines.append("        :root {")
        lines.append(_token_block(palette))
        lines.append("        }")
        lines.append(_component_rules())

    lines.append("</style>")
    return "\n".join(lines)


def build_native_bridge_html(mode: str) -> str:
    """
    Returns the HTML/JS for the browser-side native-theme bridge.

    The rendered ``data-cad-theme`` attribute changes with the mode, so
    Streamlit re-renders the iframe and the script re-runs when the selection
    changes. The script:
      * computes Streamlit's persistence key from the *parent* pathname,
      * does nothing when the stored native theme already matches,
      * never forces a value for ``system`` when no preference is stored
        (Streamlit already defaults to "Use system setting"),
      * uses a session guard to prevent reload loops,
      * is fully wrapped in try/catch, so a Streamlit internals change or a
        sandboxed iframe degrades to a no-op rather than breaking the app.
    """
    desired = native_label(mode)
    return f"""
<div data-cad-theme="{desired}" style="display:none"></div>
<script>
(function () {{
    try {{
        var desired = "{desired}";
        var parentWin = window.parent;
        var ls = parentWin.localStorage;
        var path = parentWin.location.pathname || "/";
        var prefix = "stActiveTheme-" + path;
        var want = JSON.stringify(desired);

        var keys = [];
        for (var i = 0; i < ls.length; i++) {{
            var k = ls.key(i);
            if (k && k.indexOf(prefix) === 0) {{ keys.push(k); }}
        }}
        var hasKey = keys.length > 0;
        var mismatch = false;
        for (var j = 0; j < keys.length; j++) {{
            if (ls.getItem(keys[j]) !== want) {{ mismatch = true; }}
        }}

        // Streamlit already defaults to "Use system setting" when nothing is stored.
        if (desired === "System" && !hasKey) {{ return; }}
        if (hasKey && !mismatch) {{ return; }}

        var guardKey = "cadThemeBridgeGuard";
        if (parentWin.sessionStorage.getItem(guardKey) === want) {{ return; }}
        parentWin.sessionStorage.setItem(guardKey, want);

        if (hasKey) {{
            for (var m = 0; m < keys.length; m++) {{ ls.setItem(keys[m], want); }}
        }} else {{
            ls.setItem(prefix + "-v2", want);
        }}
        parentWin.location.reload();
    }} catch (err) {{
        /* No-op: CSS-based theming in build_theme_css still applies. */
    }}
}})();
</script>
""".strip()
