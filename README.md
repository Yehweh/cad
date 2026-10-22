# Negative Remediation Memory for Autonomous Cloud Self-Healing

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Framework-Streamlit](https://img.shields.io/badge/dashboard-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![Status-Verified](https://img.shields.io/badge/verification-100%25%20passed-brightgreen.svg)]()

> **Core Research Premise:** Current cloud self-healing systems discard failed remediation attempts as disposable events, causing them to blindly retry known-bad actions when identical incidents recur. This project transforms verified failed remediation attempts into reusable decision constraints (**Negative Remediation Memory**), pruning ineffective or toxic actions from future decision spaces, preventing collateral damage, and slashing Mean Time To Recovery (MTTR).

---

## 1. Problem Statement & Research Objectives

### The Linear Recovery Defect
Traditional autonomous self-healing engines operate under a naive linear control loop:
$$\text{Failure Detected} \longrightarrow \text{Select Action} \longrightarrow \text{Execute} \longrightarrow \text{Verify}$$

When a remediation action (such as `restart_service`) fails to restore service health (e.g., because the root cause is an external locked database connection pool), existing systems:
1. Blindly repeat the same failed action on future recurring incidents.
2. Rely exclusively on **Positive Memory** ("what worked previously on average"), which fails when a globally popular action is toxic under a specific operational telemetry context.
3. Fail to prevent cascading catastrophic outages (e.g., cache stampedes triggered by cold restarts).

### Core Objectives
1. **Record verified failed remediation outcomes** alongside high-dimensional operating telemetry context.
2. **Associate recurring failures** with multi-dimensional operational states using Cosine Similarity.
3. **Prune ineffective candidate actions** dynamically to prevent repeated recovery failures.
4. **Implement self-forgetting confidence decay** to prevent permanent over-constraining when underlying application code or infrastructure is patched.
5. **Differentiate Neutral vs. Toxic failures** to penalize destructive actions and preserve system stability.

---

## 2. 3-Part Novelty Engine & Mathematical Formulation

### 2.1 Fuzzy Context Fingerprinting (Cosine Similarity)
Cloud metrics are noisy and non-identical across incidents (e.g., Incident 1 has CPU at 42.0%, Incident 2 has CPU at 44.5%). The system projects observed cloud telemetry into a normalized 5-dimensional state vector:
$$V_{\text{state}} = [\text{CPU\_Usage}, \text{Memory\_Usage}, \text{DB\_Connections}, \text{Network\_Latency}, \text{Error\_Rate}]$$
where each dimension is normalized to $[0.0, 1.0]$.

When a new incident arises with telemetry vector $V_{\text{current}}$, the Constraint Engine computes the Cosine Similarity against stored failure vectors $V_{\text{stored}}$:
$$\text{Similarity}(V_{\text{current}}, V_{\text{stored}}) = \frac{V_{\text{current}} \cdot V_{\text{stored}}}{\|V_{\text{current}}\| \|V_{\text{stored}}\|}$$

If $\text{Similarity} \ge \theta_{\text{sim}}$ (default $\theta_{\text{sim}} = 0.85$) and the failure condition matches, the negative constraint matches.

### 2.2 Ebbinghaus Confidence Decay (Self-Forgetting Memory)
Permanent blocking leads to system brittleness—an action that fails today might succeed next week following a bug patch or infrastructure upgrade. Each negative constraint is initialized with confidence $C_0 \in (0, 1]$ and decays exponentially over elapsed incident cycles $\Delta t = t - t_0$:
$$C(t) = C_0 \cdot e^{-\lambda (t - t_0)}$$

**Constraint Activation Rule:**
An action is pruned if and only if:
$$\text{Similarity}(V_{\text{current}}, V_{\text{stored}}) \ge \theta_{\text{sim}} \quad \text{AND} \quad C(t) \ge \theta_{\text{conf}}$$
where default $\theta_{\text{conf}} = 0.40$. Once $C(t) < \theta_{\text{conf}}$, the constraint expires ("self-forgets"), re-enabling safe testing.

### 2.3 Collateral Damage Scoring (Action Toxicity)
- **Neutral Failure:** The action failed to restore health but caused no additional harm (e.g., restarting an app while Postgres connection pool is full).
  - $C_0 = 0.86$, $\lambda = 0.05$ (normal decay).
- **Toxic Failure:** The action induced catastrophic collateral damage (e.g., flushing an active cache during a stampede, crashing downstream DBs).
  - $C_0 = 0.98$, $\lambda = 0.015$ (prolonged protection).

**Configurable decay (dashboard slider):** the **Decay Constant (λ)** slider sets the *base*
decay rate applied to newly recorded **neutral** constraints. Toxic constraints always decay at
`0.3 × base` (preserving the `0.015 / 0.05` ratio), so destructive actions remain protected
longer. Each constraint stores the λ it was created with, so changing the slider affects **new**
constraints only and keeps historical confidence reproducible. The same stored-λ calculation is
used both for the active/expired blocking decision and for the confidence shown in the UI.

---

## 3. System Architecture & Project Structure

```text
negative-remediation-self-healing/
├── requirements.txt         # Runtime dependencies
├── requirements-dev.txt     # Development dependencies (pytest)
├── README.md                # Comprehensive documentation
├── demo_cli.py              # CLI walkthrough reproducing Slide 5 & Slide 7 sequence
├── app.py                   # Multi-tab Streamlit Live Demonstration & Evaluation Portal
├── .streamlit/config.toml   # Streamlit config (no theme pinned; System Default)
├── src/
│   ├── __init__.py
│   ├── paths.py             # Project-root-relative path resolution
│   ├── theme.py             # Three-mode theme CSS + native-theme bridge
│   ├── memory_engine.py     # NegativeConstraint schema, Cosine Similarity, Ebbinghaus Decay, SQLite DB
│   ├── history_store.py     # Persistent incident execution history (SQLite table + indexes)
│   ├── cloud_simulator.py   # Cloud Environment Simulator with 3 failure modes & telemetry physics
│   ├── healing_controller.py # Baseline A (Fixed Rule), Baseline B (Positive-Only), Proposed System
│   └── evaluate.py          # Batch benchmark & 300 DPI publication chart generator
├── tests/                   # pytest suite (temporary DBs only)
└── outputs/                 # Evaluation CSV summaries, SQLite DB, and high-res plots
    ├── evaluation_summary.csv
    ├── remediation_memory.db      # learned memory + incident_history tables
    ├── mttr_comparison.png
    ├── attempts_and_failures.png
    └── confidence_decay_curve.png
```

---

## 4. Quickstart & Verification Instructions

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Run the Formatted CLI Walkthrough
Demonstrates the exact two-incident sequence from Slide 5 and the confidence decay breakdown from Slide 7:
```bash
python demo_cli.py
```
**Verification Evidence:**
- **Incident 1:** `restart_service` fails verification (HTTP 500). Negative constraint recorded with $C_0 = 0.86$. System recovers via `connection_capacity_remediation` in 2 attempts (MTTR: 20.0s).
- **Incident 2:** Context vector matches stored failure ($\text{Similarity} = 0.9998 \ge 0.85$, $C(t) = 0.8181 \ge 0.40$). `restart_service` is **BLOCKED**. System directly executes `connection_capacity_remediation`, resolving in **1 attempt** (MTTR: 5.0s vs 20.0s — a 75% reduction *in this illustrative single incident*).
- **Decay Demo:** Confidence decays below $\theta_{\text{conf}} = 0.40$ at $t \approx 16.5$ steps, demonstrating self-forgetting recovery.

> The single-incident numbers above are an **illustrative walkthrough**. Aggregate performance
> claims should come from the benchmark in Step 3 / Section 5, which reports measured results.

### Step 3: Run the 60-Incident Automated Benchmark Suite
```bash
python -m src.evaluate
```
Generates 300 DPI publication plots and updates `outputs/evaluation_summary.csv`.

### Step 4: Run the Automated Test Suite
```bash
pip install -r requirements-dev.txt   # adds pytest
pytest -q
```
Tests use temporary SQLite databases and never touch your real database or benchmark outputs.

### Step 5: Launch the Interactive Streamlit Dashboard
```bash
streamlit run app.py
```
Opens the interactive web dashboard on `http://localhost:8501`.
- **Appearance selector** (top of sidebar): **System Default / Light Mode / Dark Mode** — see
  section 7.
- **Overview:** live metrics derived from persisted history (incidents processed, recovery
  success rate, mean recovery time, actions blocked by negative memory).
- **Tab 1:** Live telemetry injection, candidate filtering badges, execution trace logs, and a
  persisted incident-history explorer with filters.
- **Tab 2:** Memory repository inspector, decay-trajectory projection, and interactive Cosine
  Similarity vector calculator.
- **Tab 3:** Benchmark runner, comparative metric tables, measured takeaways, and
  high-resolution plots.

Resetting/clearing is split into two confirmed actions in the sidebar: **Reset Learned Memory**
(negative constraints + positive outcomes) and **Delete Incident History** (execution logs).
Each preserves the other, and neither touches benchmark files. Switching the theme never
clears memory or history.

---

## 5. Benchmark Results (Measured)

> All values below are **measured** by `python -m src.evaluate` (60 incidents, seed `42`) and
> written to `outputs/evaluation_summary.csv`. The benchmark runs against an isolated
> **in-memory** memory engine, so it never modifies the live dashboard database. Re-run the
> command (or use the dashboard's **Run Benchmark Suite**) to reproduce tables and plots; the
> dashboard always displays whatever the latest run actually produced.

| Metric | Baseline A (Fixed Rule) | Baseline B (Positive-Only) | Proposed System (Negative Memory) |
| :--- | :---: | :---: | :---: |
| **Incidents (sample size)** | 60 | 60 | 60 |
| **Successful Recoveries** | 60 | 60 | 60 |
| **Recovery Success Rate** | 100.0% | 100.0% | 100.0% |
| **Average Attempts / Incident** | 2.20 | 1.78 | **1.07** |
| **First-Attempt Success Rate** | 0.0% | 23.3% | **95.0%** |
| **Repeated Failures (count)** | 68 | 43 | **0** |
| **Repeated Unsuccessful Rate** | 51.5% | 40.2% | **0.0%** |
| **Mean MTTR (seconds)** | 23.0s | 16.8s | **6.0s** |
| **MTTR Reduction vs Baseline A** | 0.0% | 27.2% | **73.9%** |
| **Toxic Collateral Events** | 24 | 13 | 2 |
| **Prevented Unnecessary Actions** | 0 | 0 | 33 |
| **Blocked-Action Precision** | n/a | n/a | 100.0% |

`Blocked-Action Precision` is the fraction of memory-blocked actions that would genuinely have
failed, verified by a side-effect-free simulator probe; it is reported as `n/a` when nothing
was blocked (undefined, never assumed to be 100%). `MTTR Reduction vs Baseline A` is computed
relative to Baseline A as the explicit reference controller.

---

## 6. Mathematical Formulas Summary

1. **Normalized State Vector:**
   $$V_{\text{state}} = [v_{\text{cpu}}, v_{\text{mem}}, v_{\text{db}}, v_{\text{lat}}, v_{\text{err}}], \quad v_i \in [0.0, 1.0]$$

2. **Fuzzy Cosine Similarity:**
   $$\text{Similarity}(V_1, V_2) = \frac{\sum_{i=1}^{5} V_{1,i} V_{2,i}}{\sqrt{\sum_{i=1}^{5} V_{1,i}^2} \cdot \sqrt{\sum_{i=1}^{5} V_{2,i}^2}}$$

3. **Ebbinghaus Confidence Decay:**
   $$C(t) = C_0 \cdot e^{-\lambda (t - t_0)}$$

4. **Negative Constraint Activation Condition:**
   $$\mathbb{I}_{\text{blocked}}(\text{Action}) = \left(\text{Similarity}(V_{\text{current}}, V_{\text{stored}}) \ge \theta_{\text{sim}}\right) \land \left(C(t) \ge \theta_{\text{conf}}\right)$$

---

## 7. Theme System (System Default / Light / Dark)

The sidebar exposes exactly three appearance modes, persisted across reruns and reloads via the
URL query string (`?theme=system|light|dark`):

| Mode | Native Streamlit theme | Custom components |
| :--- | :--- | :--- |
| **System Default** | Follows the OS via Streamlit's own theme handling | Matches the **actual active native theme** (`st.context.theme`) |
| **Light Mode** | Forced to Streamlit's Light theme | Light palette |
| **Dark Mode** | Forced to Streamlit's Dark theme | Dark palette |

**How it works.** Streamlit 1.66 has **no Python API to change the native app theme at
runtime** (`st.set_option` only accepts `client.*`; there is no `st.set_theme`). The theme is
resolved client-side: "System" tracks the OS through `prefers-color-scheme`, and the chosen
theme is persisted in `localStorage` under a versioned key (`stActiveTheme-<pathname>-vN`).

`src/theme.py` therefore provides two mechanisms:

1. `build_theme_css(mode, active_theme)` emits **one** set of component rules written entirely
   against `--cad-*` design tokens (surface, text, muted, border, status backgrounds/text,
   shadow), plus the token values for the requested mode and an explicit `color-scheme` for
   native form controls. Status surfaces are **opaque** per palette (e.g. light `#fdecea` vs dark
   `#2a1416`), so a card's colour never depends on — or blends with — the surrounding page
   surface. An earlier revision used translucent `rgba(...)` tints, which composited over a dark
   page and looked like dark maroon/green cards in Light Mode.

   For **System Default**, the palette is taken from `active_theme` — the theme Streamlit is
   *actually* rendering (`st.context.theme['type']`) — rather than from the OS media query. This
   is the key consistency fix: the custom cards and Streamlit's native surface must agree, and
   Streamlit's active theme can differ from the OS (a stored preference, or a choice made in
   Streamlit's own Settings menu). The OS `prefers-color-scheme` media query is kept only as a
   fallback for the rare case where the active theme cannot be determined.
2. `build_native_bridge_html(mode)` renders a tiny, guarded browser-side bridge (via
   `st.iframe`, same-origin with scripts enabled). It mirrors the selection into Streamlit's own
   `localStorage` theme key and, only when the value actually changes, triggers a single reload
   so Streamlit's **native widgets and charts** switch too. It is wrapped in `try/catch`, uses a
   session guard to prevent reload loops, and never forces a value for System Default when no
   preference is stored (Streamlit already defaults to "Use system setting").

The configuration in `.streamlit/config.toml` intentionally pins **no** theme, which is what
makes System Default work and the selector meaningful.

**Compromises / limitations (honest):**

- Because there is no official runtime API, the native switch requires one page refresh per
  explicit-mode change; the in-app selection itself is preserved through the URL query string.
- The bridge reads Streamlit's internal `stActiveTheme-*` `localStorage` key. If a future
  Streamlit version renames it, the bridge degrades to a no-op (custom CSS still applies) and
  will not break the app.
- The benchmark's matplotlib figures are static PNGs rendered server-side, so their colors are
  fixed at generation time. The dashboard renders them in the currently resolved palette
  (explicit Light/Dark, or the active native palette for System Default); the interactive
  `st.line_chart` follows the native theme.
- **Live OS changes while in System Default:** `st.context.theme` is resolved when the script
  runs. Streamlit's native surface updates live when the OS preference changes, but the custom
  cards refresh on the next rerun (any interaction). Switching themes with the in-app selector
  always updates the cards immediately, and explicit Light/Dark are never affected by this.
- Browser-level verification **was** performed with a headless Chromium (Playwright, installed
  outside the project — not added to `requirements.txt`). It reproduces the original
  inconsistency (custom cards following the OS while the native surface followed a different
  stored preference) and confirms the fix: across System Default with light/dark OS, **stale**
  stored preferences (`"Dark"` on a light OS and `"Light"` on a dark OS), and explicit Light/Dark
  overriding the OS, the custom cards and the native app surface now resolve to the **same**
  palette. The reproducible checks remain in the test-suite (`tests/test_theme.py`).
