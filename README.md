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

---

## 3. System Architecture & Project Structure

```text
negative-remediation-self-healing/
├── requirements.txt         # Project dependencies
├── README.md                # Comprehensive documentation
├── demo_cli.py              # CLI walkthrough reproducing Slide 5 & Slide 7 sequence
├── app.py                   # Multi-tab Streamlit Live Demonstration & Evaluation Portal
├── src/
│   ├── __init__.py
│   ├── memory_engine.py     # NegativeConstraint schema, Cosine Similarity, Ebbinghaus Decay, SQLite DB
│   ├── cloud_simulator.py   # Cloud Environment Simulator with 3 failure modes & telemetry physics
│   ├── healing_controller.py# Baseline A (Fixed Rule), Baseline B (Positive-Only), Proposed System
│   └── evaluate.py          # 60-incident batch benchmark & 300 DPI publication chart generator
└── outputs/                 # Evaluation CSV summaries, SQLite DB, and high-res plots
    ├── evaluation_summary.csv
    ├── remediation_memory.db
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
- **Incident 2:** Context vector matches stored failure ($\text{Similarity} = 0.9998 \ge 0.85$, $C(t) = 0.8181 \ge 0.40$). `restart_service` is **BLOCKED**. System directly executes `connection_capacity_remediation`, resolving in **1 attempt** (MTTR: 5.0s, **75% reduction**).
- **Decay Demo:** Confidence decays below $\theta_{\text{conf}} = 0.40$ at $t \approx 16.5$ steps, demonstrating self-forgetting recovery.

### Step 3: Run the 60-Incident Automated Benchmark Suite
```bash
python -m src.evaluate
```
Generates 300 DPI publication plots and updates `outputs/evaluation_summary.csv`.

### Step 4: Launch the Interactive Streamlit Dashboard
```bash
streamlit run app.py
```
Opens the interactive web dashboard on `http://localhost:8501`.
- **Tab 1:** Live telemetry injection, candidate filtering badges, and execution trace logs.
- **Tab 2:** Memory repository inspector and interactive Cosine Similarity vector calculator.
- **Tab 3:** Benchmark runner, comparative metric tables, and high-resolution plots.

---

## 5. Benchmark Results (60 Incidents)

| Metric | Baseline A (Fixed Rule) | Baseline B (Positive-Only) | Proposed System (Negative Memory) | Relative Improvement |
| :--- | :---: | :---: | :---: | :---: |
| **Average Attempts / Incident** | 2.20 | 1.78 | **1.07** | **-51.4% attempts** |
| **First-Attempt Success Rate** | 0.0% | 23.3% | **95.0%** | **+95.0% 1st-try success** |
| **Repeated Unsuccessful Rate** | 51.5% | 40.2% | **0.0%** | **100% elimination** |
| **Mean MTTR (seconds)** | 23.0s | 16.8s | **6.0s** | **-73.9% MTTR reduction** |
| **Toxic Collateral Events** | 24 | 13 | **2** (initial discovery only) | **-91.7% toxic incidents** |
| **Unnecessary Actions Prevented** | 0 | 0 | **33** | **33 wasted actions blocked** |
| **Memory Precision (%)** | 0.0% | 0.0% | **100.0%** | **Zero harmful blocks** |

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
