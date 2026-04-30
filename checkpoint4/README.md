# Checkpoint 4 — Final Evaluation & Findings

## What We Built

CP4 extended the multi-agent carbon optimization pipeline with:

1. **Narrative-first dashboard** (`dashboard.py`) — 8-page Streamlit UI designed to answer the 4 CP4 evaluation questions
2. **Single-model baseline** — A/B architecture for comparison against multi-agent
3. **Governance stress test** — Runs both pipelines on adversarial workloads to expose the quality gap
4. **Architecture comparison** — Real token/time/energy metrics from actual runs

---

## The 4 CP4 Evaluation Questions

| # | Question | Dashboard Page |
|---|---|---|
| Q1 | Why do we need a system like this? | 🌍 Why This Matters |
| Q2 | Can it provide insights conversationally? | 💬 Ask the Agent |
| Q3 | Show with graphs why multi-agent decided better | ⚖️ Multi-Agent vs Single |
| Q4 | Thousands of kgs of carbon savings at no cost | 💡 The Opportunity |

---

## Running the Dashboard

```bash
streamlit run dashboard.py
```

Requires a prior pipeline run. If no data exists:

```bash
python run_pipeline.py          # generate baseline data
python run_comparison.py        # A/B comparison (multi-agent vs single-model)
python run_stress_test.py       # governance stress test (60% production workloads)
```

---

## Key Files Added / Modified for CP4

### New files
| File | Purpose |
|---|---|
| `run_comparison.py` | Runs both pipelines and computes architecture comparison |
| `run_pipeline_single.py` | Entry point for single-model pipeline only |
| `run_stress_test.py` | A/B governance stress test with adversarial config |
| `src/agents/single_model.py` | Single-model agent (merges Planner+Governance+Executor into one LLM call) |
| `src/single_model_orchestrator.py` | Orchestrator for single-model pipeline |
| `src/data/electricity_maps.py` | Real-time grid carbon intensity via Electricity Maps API |
| `src/data/aws_pricing.py` | AWS on-demand pricing for cost estimation |
| `prompts/single_model.md` | System prompt for the single-model architecture |

### Modified files
| File | Changes |
|---|---|
| `dashboard.py` | Full CP4 rewrite — 8-page radio nav, architecture comparison, reasoning trace |
| `src/orchestrator.py` | Added `preflight_real_data_check()`, LLM token budget, negotiation dialogues |
| `src/agents/base.py` | Retry logic (5 attempts, exponential backoff), token tracking |
| `src/agents/governance.py` | `SIMULATED_HIGH_RISK_APPROVAL_RATE = 0.85` hard enforcement |
| `src/data/azure_traces.py` | `_inject_production_workloads()` for stress test reclassification |
| `src/data/carbon_intensity_real.py` | EIA + Ember real grid data integration |
| `config.py` | `REAL_DATA_ONLY`, `MAX_LLM_*` budget controls, stress test env vars |
| `.github/workflows/ci.yml` | Added `REAL_DATA_ONLY=false` for CI synthetic-data test mode |

---

## Stress Test Config

The stress test deliberately skews the workload to force governance decisions:

```
STRESS_TEST_PRODUCTION_FRACTION = 0.60   # 60% production workloads
MAX_JOBS_PER_REGION              = 8     # tighter concentration limit
MAX_COST_INCREASE_PCT            = 5     # tighter cost guardrail
MAX_AZURE_JOBS                   = 800   # 800 jobs, 5-day window
```

**Result:**
- Multi-agent: 46/50 approved (92%) — rejected 4 high-risk recs with recorded reasons
- Single-model: 50/50 approved (100%) — rubber-stamped everything, no audit trail

---

## Architecture Comparison (real run numbers)

| Metric | Multi-Agent | Single-Model |
|---|---|---|
| LLM calls | 21 | 1 |
| Total tokens | 7,938 | 9,778 |
| Wall-clock time | 25s | 37s |
| Estimated energy | 1.16 Wh | 0.62 Wh |
| Governance rejections | 4 (with reasons) | 0 (no audit trail) |
| Negotiation rounds | 3 | 0 |

Source: `data/architecture_comparison.json` and `data/stress_test/architecture_comparison.json`

---

## Dashboard Pages

| Page | Description |
|---|---|
| 🌍 Why This Matters | Regional carbon intensity bar chart (real EIA/Ember data), hourly variance, 6-step pipeline diagram, 4 hero metrics |
| 💡 The Opportunity | Scatter of all recommendations (cost vs carbon), free-wins quadrant, annualized projection from verified savings |
| ⚡ Carbon Analysis | Emissions by region, workload type, carbon intensity heatmap, daily trend |
| ✅ Verification (MRV) | Counterfactual MRV charts, 90% CI confidence intervals, evidence chain explorer |
| 🤝 The Debate | Live negotiation transcript between Planner and Governance agents |
| ⚖️ Multi-Agent vs Single | Architecture diagram + 4 tabs: Efficiency, Governance, Reasoning Trace, Verdict |
| 🏆 Team Leaderboard | Points awarded only for verified savings |
| 💬 Ask the Agent | LLM chatbot with full agent reasoning traces + governance decisions injected into context |

---

## Data Sources

| Region | Carbon Intensity Source |
|---|---|
| us-east-1 | EIA API (PJM grid) — real-time |
| us-west-2 | EIA API (BPAT grid) — real-time |
| ap-south-1 | Ember Climate 2023 — cited baseline |
| eu-north-1 | Swedish Energy Agency 2023 — synthetic fallback |
| eu-west-1 | EirGrid 2023 — synthetic fallback |

Workload data: Azure VM Traces (publicly available research dataset)
