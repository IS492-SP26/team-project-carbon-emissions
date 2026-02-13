# sust-AI-naible: Multi-Agent Cloud Carbon Optimization System

## System Design Document — v0.1 (Brainstorm)

---

## 0. Problem Reframing

The core insight: **cloud carbon optimization is a closed-loop control problem**, not a dashboard.

```
Effective Cost = Cloud Bill ($) + (tCO₂e × $75/ton)
```

The system must continuously sense workload patterns, model their carbon+cost impact, decide on shifts, act on those decisions, verify the outcomes, and learn from the delta. Every claim of "carbon saved" must have a traceable evidence chain — not estimates alone, but counterfactual reasoning: *what would have happened if we hadn't intervened?*

**Key tension**: carbon reduction often trades off against latency, cost, and developer velocity. The system must navigate this Pareto frontier, not collapse it into a single score.

---

## 1. Agent Roster

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        GOVERNANCE AGENT                             │
│              (approval gates, policy enforcement)                    │
│─────────────────────────────────────────────────────────────────────│
│                                                                     │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐        │
│  │ INGESTOR │──▶│ CARBON   │──▶│ PLANNER  │──▶│ EXECUTOR │        │
│  │ AGENT    │   │ ACCNTANT │   │ AGENT    │   │ AGENT    │        │
│  └──────────┘   └──────────┘   └──────────┘   └──────────┘        │
│       │              │              │              │                 │
│       │              ▼              │              ▼                 │
│       │     ┌──────────────┐       │     ┌──────────────┐          │
│       │     │  EMISSIONS   │       │     │  VERIFIER    │          │
│       │     │  FACTOR DB   │       │     │  AGENT       │◀─┐      │
│       │     └──────────────┘       │     └──────────────┘  │      │
│       │                            │              │         │      │
│       ▼                            ▼              ▼         │      │
│  ┌─────────────────────────────────────────────────────┐   │      │
│  │            SHARED MEMORY (World Model)              │───┘      │
│  │  Activity Ledger │ Decision Log │ Outcome Log       │          │
│  └─────────────────────────────────────────────────────┘          │
│                            │                                       │
│                            ▼                                       │
│                    ┌──────────────┐                                │
│                    │  DEV COPILOT │                                │
│                    │  (optional)  │                                │
│                    └──────────────┘                                │
└─────────────────────────────────────────────────────────────────────┘
```

---

### Agent 1: Ingestor Agent

- **Purpose**: Collect, normalize, and validate cloud usage data from heterogeneous sources into a canonical activity ledger.
- **Type**: Mostly deterministic, LLM-assisted for schema mapping of new sources.

| Aspect | Detail |
|---|---|
| **Inputs** | Cloud billing APIs (AWS CUR, GCP Billing Export, Azure Cost Mgmt), K8s metrics (Prometheus), CI/CD logs (GitHub Actions, Jenkins), scheduler metadata |
| **Outputs** | Normalized activity records: `{job_id, service, region, timestamp_start, timestamp_end, resource_type, resource_qty, cost_usd, workload_category, team_id}` |
| **Tools** | Cloud provider SDKs, SQL queries on billing exports, Prometheus API, GitHub API, cron scheduler |
| **NOT allowed to** | Modify any cloud resources. Make cost or carbon attributions (that's the Accountant's job). Directly contact developers. |
| **Success metrics** | Data freshness < 1 hour, schema validation pass rate > 99.5%, zero silent data drops |
| **Failure modes** | (1) API rate limits → stale data → stale decisions. (2) Schema drift from provider updates → parsing failures. (3) Missing data for spot instances or preemptible VMs. (4) Double-counting shared resources. |

---

### Agent 2: Carbon Accountant Agent

- **Purpose**: Compute location-based and market-based carbon emissions for every activity record, with uncertainty bounds and full provenance.
- **Type**: **Strictly deterministic** — no LLM in the calculation path.

| Aspect | Detail |
|---|---|
| **Inputs** | Activity ledger records, emissions factor store, grid carbon intensity data (WattTime, Electricity Maps, or static IPCC/EPA factors) |
| **Outputs** | Emissions records: `{activity_id, scope, kgCO2e, kgCO2e_lower, kgCO2e_upper, emission_factor_id, methodology, timestamp}` |
| **Tools** | Emissions factor DB (versioned), grid intensity API or cached time series, PUE lookup tables, embodied carbon amortization calculator |
| **NOT allowed to** | Round or suppress uncertainty. Use unversioned emission factors. Retroactively modify past records. |
| **Success metrics** | 100% of activity records have emissions attributed within 2 hours. Uncertainty bounds cover actual ≥ 95% of the time (calibration). Audit trail for every number. |
| **Failure modes** | (1) Stale grid intensity data → wrong marginal signal. (2) PUE assumptions wrong for specific data centers. (3) Scope 3 (embodied, networking) omitted → systematic undercount. (4) Uncertainty bounds too wide to be useful → planner ignores them. |

**Critical design choice**: Emissions math MUST be deterministic + auditable. The formulas are:

```
kgCO₂e = resource_qty × duration_hrs × PUE × grid_intensity_kgCO2_per_kWh × TDP_kW
```

Uncertainty propagated via interval arithmetic (not Monte Carlo for MVP — too slow for every record).

---

### Agent 3: Planner Agent

- **Purpose**: Generate cost+carbon-optimal workload placement and scheduling recommendations that respect SLA constraints.
- **Type**: Hybrid — deterministic solver core, LLM for constraint interpretation and explanation generation.

| Aspect | Detail |
|---|---|
| **Inputs** | Activity ledger (historical + predicted), emissions forecasts by region/time, cost model, SLA definitions, workload category labels, team preferences |
| **Outputs** | Recommendation set: `{recommendation_id, workload_id, action_type, current_config, proposed_config, est_carbon_delta_kgCO2e, est_cost_delta_usd, est_latency_delta_ms, confidence, rationale_text}` |
| **Tools** | OR-Tools / PuLP solver, time-series forecaster (Prophet or simple ARIMA for grid intensity), cost calculator, SLA constraint checker |
| **NOT allowed to** | Execute any changes. Override SLA constraints. Make recommendations without cost AND carbon estimates. Generate recommendations with confidence < threshold without flagging. |
| **Success metrics** | ≥ 80% of recommendations accepted by teams. Actual savings within ±20% of estimates. No SLA violations from accepted recommendations. |
| **Failure modes** | (1) Solver infeasible due to over-constrained SLAs → outputs nothing → system stalls. (2) Optimizes for carbon but costs spike → trust erosion. (3) Stale forecasts → recommendations shift workloads to currently-dirty grids. (4) Doesn't account for data gravity / egress costs. (5) "Rebound effect" — frees capacity that gets consumed. |

---

### Agent 4: Executor Agent

- **Purpose**: Translate approved recommendations into concrete infrastructure changes (tickets, PRs, scheduler config) and track their execution status.
- **Type**: LLM-heavy for ticket/PR generation, deterministic for scheduler API calls.

| Aspect | Detail |
|---|---|
| **Inputs** | Approved recommendations from Planner (with governance sign-off), infrastructure templates, team routing rules |
| **Outputs** | Jira tickets, GitHub PRs (Terraform/K8s manifests), scheduler API calls, status updates to Decision Log |
| **Tools** | Jira/Linear API, GitHub API (PR creation), Terraform Cloud API, K8s API (CronJob rescheduling), Slack/Teams notifications |
| **NOT allowed to** | Execute without approval for HIGH-impact changes (> $100/day cost change or production services). Self-approve. Modify production routing without canary. Make changes outside the approved recommendation scope. |
| **Success metrics** | Execution success rate > 95%. Mean time from approval to execution < 4 hours (automated) / < 48 hours (manual). Zero unauthorized changes. |
| **Failure modes** | (1) PR merge conflicts with developer changes → stale. (2) Scheduler change breaks dependent jobs → cascade failure. (3) Partial execution (moved region but didn't update DNS). (4) Notification fatigue → teams ignore alerts. |

---

### Agent 5: Verifier Agent

- **Purpose**: Measure actual outcomes of executed recommendations against counterfactual baselines and detect drift.
- **Type**: **Strictly deterministic** for measurement, LLM-assisted for report generation.

| Aspect | Detail |
|---|---|
| **Inputs** | Pre-action baseline snapshots, post-action activity records, emissions records, decision log |
| **Outputs** | Verification records: `{recommendation_id, measured_carbon_delta, counterfactual_carbon, confidence_interval, measured_cost_delta, sla_compliance, verification_status, evidence_chain}` |
| **Tools** | Statistical comparison engine, A/B or before/after analysis, SLA monitoring queries, anomaly detection |
| **NOT allowed to** | Verify its own recommendations (separation of concerns). Cherry-pick time windows. Report savings without confidence intervals. |
| **Success metrics** | 100% of executed recommendations verified within 7 days. Calibration: 90% CIs contain actual 90% of the time. Drift detection latency < 24 hours. |
| **Failure modes** | (1) Counterfactual model wrong → inflated or deflated savings claims. (2) Confounders (business growth, seasonality) not controlled → spurious attribution. (3) Not enough post-period data → wide CIs → useless. (4) Verification lag too long → system can't learn. |

**Counterfactual approach** (MVP): Simple difference-in-differences. For a job moved from region A to B at time t:
- Counterfactual emissions = (actual resource usage in B) × (emission factor of A at actual times)
- Actual emissions = (actual resource usage in B) × (emission factor of B at actual times)
- Delta = counterfactual - actual

This is conservative and auditable. More sophisticated: synthetic control methods (stretch goal).

---

### Agent 6: Developer Copilot (Optional, recommended)

- **Purpose**: Surface contextual carbon/cost insights to developers at decision time (PR review, CI config, resource provisioning) and manage gamification.
- **Type**: LLM-driven for interactions, deterministic backend for points.

| Aspect | Detail |
|---|---|
| **Inputs** | PR diffs, CI config changes, resource provisioning requests, team leaderboard data, verified outcomes |
| **Outputs** | PR comments, Slack nudges, dashboard widgets, carbon "receipts" per deployment, points/badges |
| **Tools** | GitHub PR API (comments), Slack API, dashboard API, points ledger |
| **NOT allowed to** | Block PRs (advisory only). Award points for unverified savings. Share individual developer data publicly (team-level only). Nag more than 2x/week per developer. |
| **Success metrics** | Developer engagement rate > 40%. Nudge-to-action conversion > 15%. NPS from developers > 0 (not negative). |
| **Failure modes** | (1) Alert fatigue → ignored → wasted effort. (2) Gamification gamed (see §7). (3) Inaccurate nudges → trust destroyed. (4) Perceived as "carbon police" → political backlash. |

---

### Agent 7: Governance Agent

- **Purpose**: Enforce approval policies, rate-limit system actions, maintain audit trail integrity, and flag anomalies for human review.
- **Type**: Mostly deterministic rule engine, LLM for policy interpretation.

| Aspect | Detail |
|---|---|
| **Inputs** | All recommendations before execution, all verification reports, system health metrics, organizational policies (text) |
| **Outputs** | Approval/rejection decisions, escalation notices, audit reports, policy violation alerts |
| **Tools** | Policy rule engine (OPA/Rego or simple rule set), approval workflow (Jira/Slack), audit log, anomaly detector |
| **NOT allowed to** | Override human rejection. Auto-approve HIGH-risk changes. Modify policies without human sign-off. Suppress alerts. |
| **Success metrics** | Zero unauthorized executions. 100% of HIGH-risk changes human-approved. Audit trail completeness = 100%. Mean approval latency < 8 hours. |
| **Failure modes** | (1) Overly conservative → bottleneck → recommendations rot. (2) Policy rules stale → blocks valid optimizations. (3) Single approver → bus factor / vacation delays. |

---

## 2. World Model + Memory

### Shared Memory Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    PostgreSQL + TimescaleDB              │
│                                                         │
│  ┌─────────────────┐  ┌──────────────────────────┐     │
│  │ activity_ledger │  │ emissions_factors         │     │
│  │ (TimescaleDB    │  │ (versioned, append-only)  │     │
│  │  hypertable)    │  │                           │     │
│  └─────────────────┘  └──────────────────────────┘     │
│                                                         │
│  ┌─────────────────┐  ┌──────────────────────────┐     │
│  │ decision_log    │  │ outcome_log              │     │
│  │ (append-only)   │  │ (append-only)            │     │
│  └─────────────────┘  └──────────────────────────┘     │
│                                                         │
│  ┌─────────────────┐  ┌──────────────────────────┐     │
│  │ points_ledger   │  │ policy_store             │     │
│  │                 │  │ (versioned)              │     │
│  └─────────────────┘  └──────────────────────────┘     │
└─────────────────────────────────────────────────────────┘
```

### Schema Definitions

#### 2.1 Activity Ledger

Written by: **Ingestor Agent** (every ingestion cycle, ~hourly)

```sql
CREATE TABLE activity_ledger (
    activity_id         UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    job_id              TEXT NOT NULL,          -- external job identifier
    service_name        TEXT NOT NULL,
    team_id             TEXT NOT NULL,
    cloud_provider      TEXT NOT NULL,          -- aws | gcp | azure
    region              TEXT NOT NULL,          -- e.g., us-east-1
    instance_type       TEXT,
    resource_type       TEXT NOT NULL,          -- compute | storage | network | gpu
    resource_qty        FLOAT NOT NULL,         -- vCPU-hours, GB-hours, etc.
    resource_unit       TEXT NOT NULL,
    cost_usd            FLOAT NOT NULL,
    workload_category   TEXT NOT NULL,          -- urgent | balanced | sustainable
    started_at          TIMESTAMPTZ NOT NULL,
    ended_at            TIMESTAMPTZ,
    ingested_at         TIMESTAMPTZ DEFAULT NOW(),
    source              TEXT NOT NULL,          -- billing_api | k8s_metrics | ci_logs
    raw_record_ref      TEXT                    -- pointer to raw data for audit
);
-- TimescaleDB hypertable on started_at
```

#### 2.2 Emissions Factors Store

Written by: **Carbon Accountant Agent** (on update, typically weekly; grid intensity hourly)

```sql
CREATE TABLE emission_factors (
    factor_id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    factor_type         TEXT NOT NULL,          -- grid_intensity | pue | embodied
    region              TEXT,
    provider            TEXT,
    value               FLOAT NOT NULL,         -- kgCO2e per kWh (grid) or multiplier (PUE)
    value_lower         FLOAT NOT NULL,         -- uncertainty bound
    value_upper         FLOAT NOT NULL,
    unit                TEXT NOT NULL,
    source              TEXT NOT NULL,          -- watttime | electricity_maps | epa_egrid | ipcc
    source_url          TEXT,
    valid_from          TIMESTAMPTZ NOT NULL,
    valid_to            TIMESTAMPTZ,
    version             INT NOT NULL,
    created_at          TIMESTAMPTZ DEFAULT NOW()
);
-- UNIQUE constraint on (factor_type, region, provider, version) — append-only, never UPDATE

CREATE TABLE grid_intensity_timeseries (
    ts                  TIMESTAMPTZ NOT NULL,
    region              TEXT NOT NULL,
    intensity_kgco2_kwh FLOAT NOT NULL,
    intensity_lower     FLOAT,
    intensity_upper     FLOAT,
    source              TEXT NOT NULL,
    PRIMARY KEY (ts, region)
);
-- TimescaleDB hypertable on ts
```

#### 2.3 Decision Log

Written by: **Planner Agent** (on recommendation), **Governance Agent** (on approval/rejection), **Executor Agent** (on execution)

```sql
CREATE TABLE decision_log (
    decision_id         UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    recommendation_id   UUID NOT NULL,
    workload_id         TEXT NOT NULL,
    action_type         TEXT NOT NULL,          -- region_shift | time_shift | right_size | spot_convert
    current_config      JSONB NOT NULL,
    proposed_config     JSONB NOT NULL,
    est_carbon_delta_kg FLOAT NOT NULL,         -- negative = reduction
    est_cost_delta_usd  FLOAT NOT NULL,
    est_latency_delta   FLOAT,                  -- ms, nullable
    confidence          FLOAT NOT NULL,         -- 0-1
    workload_category   TEXT NOT NULL,
    rationale           TEXT NOT NULL,           -- LLM-generated explanation
    status              TEXT NOT NULL DEFAULT 'proposed',
                        -- proposed | approved | rejected | executing | executed | failed | verified
    proposed_at         TIMESTAMPTZ DEFAULT NOW(),
    approved_at         TIMESTAMPTZ,
    approved_by         TEXT,                    -- human or governance_agent
    executed_at         TIMESTAMPTZ,
    execution_ref       TEXT,                    -- PR URL, ticket ID, etc.
    risk_level          TEXT NOT NULL            -- low | medium | high
);
```

#### 2.4 Outcome Log

Written by: **Verifier Agent** (post-execution, within 7 days)

```sql
CREATE TABLE outcome_log (
    outcome_id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    decision_id             UUID REFERENCES decision_log(decision_id),
    recommendation_id       UUID NOT NULL,
    measured_carbon_delta_kg FLOAT NOT NULL,
    counterfactual_carbon_kg FLOAT NOT NULL,
    actual_carbon_kg         FLOAT NOT NULL,
    carbon_ci_lower          FLOAT NOT NULL,     -- 90% confidence interval
    carbon_ci_upper          FLOAT NOT NULL,
    measured_cost_delta_usd  FLOAT NOT NULL,
    sla_compliant            BOOLEAN NOT NULL,
    sla_detail               JSONB,
    verification_method      TEXT NOT NULL,       -- diff_in_diff | before_after | synthetic_control
    evidence_chain           JSONB NOT NULL,      -- array of {source, query, timestamp, hash}
    verified_at              TIMESTAMPTZ DEFAULT NOW(),
    verification_status      TEXT NOT NULL        -- confirmed | partial | refuted | inconclusive
);
```

#### 2.5 Points Ledger (Gamification)

Written by: **Developer Copilot** (after Verifier confirms)

```sql
CREATE TABLE points_ledger (
    entry_id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    team_id             TEXT NOT NULL,
    outcome_id          UUID REFERENCES outcome_log(outcome_id),
    points              INT NOT NULL,           -- 1 point per verified kgCO2e avoided
    reason              TEXT NOT NULL,
    awarded_at          TIMESTAMPTZ DEFAULT NOW()
);
-- Points ONLY awarded after verification. No points for unverified claims.
```

### Write Permissions Matrix

| Store | Ingestor | Accountant | Planner | Executor | Verifier | Copilot | Governance |
|-------|----------|------------|---------|----------|----------|---------|------------|
| activity_ledger | **WRITE** | read | read | read | read | read | read |
| emission_factors | read | **WRITE** | read | — | read | — | read |
| grid_intensity_ts | read | **WRITE** | read | — | read | — | read |
| decision_log | — | — | **WRITE** (propose) | **WRITE** (execute status) | **WRITE** (verify status) | read | **WRITE** (approve/reject) |
| outcome_log | — | — | read | — | **WRITE** | read | read |
| points_ledger | — | — | — | — | — | **WRITE** | read |

**Critical invariant**: All write tables are append-only in production. No UPDATEs except status transitions on `decision_log` (which are logged).

---

## 3. The Agentic Loop (Closed-Loop Control)

### Step-by-Step Loop

```
    ┌──────────────────────────────────────────────────────────────┐
    │                                                              │
    ▼                                                              │
┌────────┐    ┌────────┐    ┌────────┐    ┌────────┐    ┌────────┐│   ┌────────┐
│ SENSE  │───▶│ MODEL  │───▶│ DECIDE │───▶│  ACT   │───▶│ VERIFY ││──▶│ LEARN  │
│        │    │        │    │        │    │        │    │        ││   │        │
│Ingestor│    │Carbon  │    │Planner │    │Executor│    │Verifier││   │Planner │
│        │    │Acct.   │    │+Govnce │    │        │    │        ││   │retrain │
└────────┘    └────────┘    └────────┘    └────────┘    └────────┘│   └────────┘
                                                                   │        │
                                                                   └────────┘
```

### Detailed Steps

| Step | Agent(s) | Frequency | Human Approval? | Description |
|------|----------|-----------|-----------------|-------------|
| **1. SENSE** | Ingestor | Hourly (billing), 5-min (K8s metrics) | No | Pull cloud usage data. Validate schemas. Write to activity_ledger. Flag anomalies (10x cost spikes, missing regions). |
| **2. MODEL** | Carbon Accountant | Hourly (on new activity data) | No | Compute emissions for new activity records. Fetch latest grid intensity. Propagate uncertainty. Write emissions records. |
| **3. DECIDE** | Planner + Governance | Daily (batch), on-demand (triggered) | **YES for HIGH-risk** | Planner generates recommendations. Governance evaluates risk level. LOW-risk auto-approved. MEDIUM-risk: team lead notified, 24h auto-approve if no objection. HIGH-risk: explicit human approval required. |
| **4. ACT** | Executor | On approval (event-driven) | No (approval already granted) | Create tickets/PRs/scheduler changes. Record execution_ref in decision_log. Notify affected teams. Snapshot pre-action baseline. |
| **5. VERIFY** | Verifier | 3-7 days post-execution | No | Compute counterfactual. Compare actual vs. counterfactual emissions + cost. Check SLA compliance. Write outcome_log. Flag refuted recommendations. |
| **6. LEARN** | Planner (feedback loop) | Weekly | No | Analyze verification results. Update confidence calibration. Adjust recommendation thresholds. Feed Copilot for points and nudges. |

### Timing Summary

```
Continuous:  K8s metrics ingestion (5-min)
Hourly:      Billing ingestion, emissions computation
Daily:       Planner batch run (03:00 UTC), governance review window
On-demand:   Triggered recommendations (new workload, cost spike)
Weekly:      Learning cycle, leaderboard update, drift report
Monthly:     Full system audit report, factor store review
```

### Human-in-the-Loop Gates

1. **HIGH-risk recommendation approval** — team lead + platform team sign-off
2. **Emission factor updates** — data team review before activation
3. **Policy changes** — governance committee (monthly)
4. **System parameter changes** (carbon price, thresholds) — leadership approval

---

## 4. Optimization Framing

### Workload Categories

| Category | Deferral Window | Region Flexibility | Typical Workloads | % of fleet (Assumption) |
|----------|----------------|--------------------|--------------------|------------------------|
| **Urgent** | 0 (no deferral) | Same continent only | Production APIs, real-time serving, on-call jobs | ~30% |
| **Balanced** | ≤ 4 hours | Any region, same provider | CI/CD, scheduled reports, non-critical batch | ~45% |
| **Sustainable** | ≤ 24 hours | Any region, any provider | Model training, analytics backfill, dev/test, nightly ETL | ~25% |

### Formulation A: MVP Rule-Based Planner

**Approach**: Priority queue with simple scoring. No solver needed.

```
For each pending workload W in category C:
    1. Get candidate (region, time_slot) pairs within C's constraints
    2. For each candidate, compute:
         score = α × carbon_cost + β × dollar_cost + γ × latency_penalty
         where:
           carbon_cost  = est_kgCO2e × $75/ton × (1/1000)   [convert kg to tons]
           dollar_cost  = est_cloud_cost_usd
           latency_penalty = max(0, est_latency_ms - sla_target_ms) × penalty_rate
    3. Rank candidates by score (lower = better)
    4. Select top candidate IF score < current_config_score
    5. Else: no recommendation (status quo is optimal)
```

**Weights** (Assumption — tunable):
- α = 1.0 (carbon valued at internal price)
- β = 1.0 (dollar cost weighted equally)
- γ = 10.0 (strong penalty for SLA violation)

**Why this works for MVP**: Defensible, explainable, auditable. Every recommendation has a clear score breakdown. No "black box" optimization.

**Limitation**: Doesn't consider cross-workload interactions (e.g., moving 50 jobs to one region might overwhelm capacity).

### Formulation B: Multi-Objective Constrained Optimizer

**Objective** (minimize):

```
min  Σ_i [ cost_i(region_i, time_i) + carbon_price × emissions_i(region_i, time_i) ]

subject to:
  latency_i(region_i) ≤ sla_max_latency_i          ∀i ∈ Urgent
  |time_i - time_i_requested| ≤ defer_max_i          ∀i
  availability(region_i) ≥ sla_availability_i         ∀i ∈ Urgent ∪ Balanced
  Σ_i resource_demand(region_r, time_t) ≤ capacity_r  ∀r,t  [capacity constraint]
  egress_cost_i(region_i) ≤ egress_budget_i            ∀i    [data gravity]
```

**Variables**: For each workload i: `region_i ∈ R_i` (feasible regions), `time_i ∈ T_i` (feasible time window)

**Solver**: OR-Tools CP-SAT (constraint programming) or PuLP (LP relaxation with rounding)

**Why CP-SAT**: Workload scheduling is naturally a constraint satisfaction problem. CP-SAT handles discrete choices (regions) + time windows natively. Or-Tools is free, fast, and well-documented.

**Practical consideration**: For ~100 devs generating ~500-2000 jobs/day, CP-SAT solves in seconds. No scalability concern for v1.

### Gamification ↔ Verified Savings Connection

```
Points = f(verified_kgCO2e_avoided)

Specifically:
  1 point = 1 kg CO₂e avoided (verified by Verifier Agent)
  
  Points are ONLY awarded when:
    (a) Recommendation was executed
    (b) Verifier confirmed savings (status = "confirmed" or "partial")
    (c) SLA was not violated
  
  If verification_status = "refuted" → 0 points, team notified
  If verification_status = "partial" → points = measured_savings × 0.5 (conservative)
```

**Anti-gaming**: Points tied to verified outcomes, not actions. You don't get points for "marking a job as sustainable" — you get points when the system verifies that your job actually ran in a lower-carbon config and the measured delta is positive.

**Leaderboards**: Team-level only (not individual) to prevent perverse incentives. Monthly reset with rolling annual tally.

---

## 5. AI vs Deterministic Boundaries

### The Bright Line

```
┌───────────────────────────────────────────────────────────────┐
│                  LLM-DRIVEN (soft, advisory)                  │
│                                                               │
│  • Parsing unstructured policies into rule-engine configs      │
│  • Generating PR descriptions and ticket bodies               │
│  • Explaining recommendations in human language               │
│  • Summarizing verification reports for stakeholders          │
│  • Developer Copilot nudges and contextual suggestions        │
│  • Interpreting new/ambiguous workload categories             │
│  • Generating audit report narratives                         │
│                                                               │
├───────────────────────────────────────────────────────────────┤
│             DETERMINISTIC (hard, authoritative)               │
│                                                               │
│  • Emissions calculations (kgCO₂e = f(resource, grid, PUE))  │
│  • Optimization solver (OR-Tools / PuLP)                      │
│  • Cost calculations                                          │
│  • SLA compliance checks                                      │
│  • Counterfactual estimation                                  │
│  • Points calculation                                         │
│  • Approval policy evaluation (OPA/Rego)                      │
│  • Data validation and schema enforcement                     │
│  • Uncertainty propagation (interval arithmetic)              │
│                                                               │
└───────────────────────────────────────────────────────────────┘
```

### Justification

| Boundary | Why |
|----------|-----|
| Emissions math = deterministic | **Auditability**. If an auditor asks "where did 42.7 kgCO₂e come from?", we need a traceable formula, not "the LLM said so." GHG Protocol requires reproducible calculations. |
| Optimization solver = deterministic | **Reliability**. LLMs hallucinate constraints. A CP-SAT solver either finds a feasible solution or proves infeasibility. No ambiguity. |
| Counterfactuals = deterministic | **Credibility**. Claimed savings must be reproducible. Two runs with same inputs must produce same outputs. LLM outputs are stochastic. |
| Policy interpretation = LLM | **Flexibility**. Organizational policies are written in natural language. An LLM can parse "no workloads in China-region for ITAR-controlled projects" into a constraint. But the parsed constraint is then enforced deterministically. |
| Explanations = LLM | **UX**. Developers don't want to read "recommendation_id: abc123, action_type: region_shift, est_carbon_delta: -12.3." They want "Moving your nightly ETL from us-east-1 to us-west-2 would save ~12 kg CO₂e/week because Oregon's grid is 3x cleaner than Virginia's right now." |

### The Handoff Pattern

```
LLM parses → Deterministic validates → Deterministic computes → LLM explains
         ↑                                                          ↓
         └──── Human reviews if parse confidence < 0.9 ─────────────┘
```

The LLM is never in the critical path of any calculation. It's a translator at the edges.

---

## 6. Realistic Numbers + Simulation Plan

### Synthetic Workload Model

**Assumption**: 100 developers, generating workloads across 5 categories.

| Workload Type | Jobs/Day | Avg Duration | Avg vCPUs | Avg Cost/Job | Category Split |
|---------------|----------|-------------|-----------|-------------|----------------|
| CI/CD builds | 800 | 12 min | 4 | $0.15 | 60% balanced, 30% urgent, 10% sustainable |
| Batch analytics | 50 | 2 hours | 16 | $2.40 | 20% balanced, 80% sustainable |
| Model training | 10 | 6 hours | 8 (+ 1 GPU) | $18.00 | 100% sustainable |
| Dev/test envs | 100 | 8 hours | 2 | $0.80 | 100% sustainable |
| Production services | 30 | 24/7 | 4 | $3.50/day | 100% urgent |

**Total daily jobs**: ~990 (Assumption)
**Total daily cloud spend**: ~$600/day → ~$18,000/month (Assumption, plausible for 100-dev org)

### Carbon Intensity Model

**Known** (from public sources — Electricity Maps, WattTime, EPA eGRID):

| Region | Avg Grid Intensity (gCO₂/kWh) | Range | Source |
|--------|-------------------------------|-------|--------|
| us-east-1 (Virginia) | 350 | 250–450 | EPA eGRID 2022 |
| us-west-2 (Oregon) | 90 | 50–150 | EPA eGRID 2022 (hydro-heavy) |
| eu-west-1 (Ireland) | 300 | 150–450 | EirGrid data |
| eu-north-1 (Stockholm) | 30 | 10–80 | Swedish Energy Agency |
| ap-south-1 (Mumbai) | 700 | 600–800 | CEA India |

**Assumption**: PUE = 1.1 (hyperscaler average, per Google/AWS sustainability reports)
**Assumption**: Average server TDP ≈ 0.005 kW per vCPU (simplified)

### Baseline Emissions Calculation

```
For a single job:
  kgCO₂e = vCPUs × duration_hrs × TDP_per_vCPU × PUE × grid_intensity_kg/kWh

Example CI/CD job (us-east-1):
  = 4 × 0.2hrs × 0.005kW × 1.1 × 0.350 kgCO₂/kWh
  = 0.00154 kgCO₂e per job
  × 800 jobs/day = 1.23 kgCO₂e/day from CI/CD

Example model training (us-east-1, with GPU):
  = (8×0.005 + 1×0.3)kW × 6hrs × 1.1 × 0.350
  = 0.34 kW × 6 × 1.1 × 0.350
  = 0.79 kgCO₂e per training job
  × 10 jobs/day = 7.9 kgCO₂e/day from training
```

**Estimated baseline** (all workloads, Assumption):

| Metric | Daily | Monthly | Annual |
|--------|-------|---------|--------|
| Cloud spend | $600 | $18,000 | $216,000 |
| Emissions | ~15 kgCO₂e | ~450 kgCO₂e | ~5.4 tCO₂e |
| Carbon cost (@$75/t) | $1.13 | $33.75 | $405 |
| **Effective cost** | **$601.13** | **$18,034** | **$216,405** |

Note: Carbon cost is <1% of cloud spend at this scale. This is realistic — carbon pricing only significantly changes decisions when:
(a) carbon price is much higher ($200+/ton), or
(b) the organization is large enough that small % = large absolute numbers, or
(c) there are zero-cost carbon reduction opportunities (region shifts that don't increase cloud cost).

**This is an important honesty check**: at $75/ton, the system's value proposition is primarily (c) — finding carbon wins that are free or cheap.

### Optimized Scenario (Assumption)

**Conservative target**: 25-40% reduction of flexible workload emissions via region + time shifting.

- ~70% of workloads are flexible (balanced + sustainable)
- Of those, ~50% can meaningfully benefit from region/time shifts
- Average carbon reduction per shifted workload: 50-70% (e.g., Virginia → Oregon)

```
Flexible fleet emissions: ~10.5 kgCO₂e/day (70% of 15)
Shiftable: ~5.25 kgCO₂e/day (50% of flexible)
Reduction: ~3.15 kgCO₂e/day (60% avg reduction)

Optimized total: 15 - 3.15 = ~11.85 kgCO₂e/day
Reduction: ~21% overall
```

| Metric | Baseline | Optimized | Delta |
|--------|----------|-----------|-------|
| Daily emissions | 15 kgCO₂e | 11.85 kgCO₂e | -21% |
| Monthly emissions | 450 kgCO₂e | 356 kgCO₂e | -94 kgCO₂e |
| Annual emissions | 5.4 tCO₂e | 4.3 tCO₂e | -1.1 tCO₂e |
| Annual carbon cost | $405 | $323 | -$82 |
| Cloud cost change | — | +$0 to +$500/yr | (egress, possible region cost diffs) |

**Honest assessment**: The annual carbon cost saving (~$82) is tiny. The real value is:
1. Organizational learning and carbon literacy
2. Building the measurement infrastructure for when carbon prices rise
3. Demonstrating MRV capability
4. Identifying zero-regret optimizations (things that save both money AND carbon)

### Simulation Plan (No External Access Required)

```python
# simulation_plan.py — what we'd build

# 1. Workload Generator
#    - Poisson arrivals for each workload type
#    - Duration: log-normal distribution
#    - Resource: categorical (tied to workload type)
#    - Time-of-day patterns (CI/CD peaks during work hours)

# 2. Carbon Intensity Simulator
#    - Hourly time series per region
#    - Base: sinusoidal (day/night cycle) + noise
#    - Ranges from Known table above
#    - Option: replay real Electricity Maps data if available

# 3. Cost Model
#    - Lookup table: {region, instance_type} → $/hour
#    - Egress: $0.01-0.09/GB cross-region (Known, from AWS pricing)
#    - Spot discount: 60-70% (Known, from AWS)

# 4. Simulator Loop
#    for day in simulation_period:
#        workloads = generate_workloads(day)
#        baseline_result = run_baseline(workloads)     # no optimization
#        optimized_result = run_optimized(workloads)   # with planner
#        record(baseline_result, optimized_result)

# 5. Output: CSV with daily/hourly granularity
#    - baseline_emissions, optimized_emissions
#    - baseline_cost, optimized_cost
#    - sla_violations, deferred_jobs
#    - trade-off curves: sweep carbon_price from $0 to $500/ton
```

### Uncertainty Sources

| Source | Impact | How Reported |
|--------|--------|-------------|
| Grid intensity data staleness | Emissions could be off by ±30% in volatile grids | Confidence intervals on each emissions record |
| PUE assumption (1.1) | Could be 1.05–1.4 depending on data center | Sensitivity analysis: report range |
| Server power model (linear vCPU→kW) | Nonlinear in reality; idle power significant | Flag as known limitation; bound with min/max TDP |
| Workload duration prediction | If a job runs 2x longer, emissions double | Use actual (not estimated) for verification |
| Cost model simplification | Ignores reserved instances, savings plans | Flag as assumption; use actual billing for verification |
| Counterfactual model | "What would have happened" is inherently uncertain | Report 90% CIs; use conservative (lower-bound) for official claims |

---

## 7. Failure Modes + Attack Surface

| # | Failure Mode | Severity | Likelihood | Mitigation |
|---|-------------|----------|------------|------------|
| 1 | **Region shifting increases latency beyond SLA** | HIGH | Medium | Hard SLA constraint in solver. Verifier checks post-action latency. Auto-rollback trigger if p99 latency > threshold for 15 min. |
| 2 | **Stale or wrong emission factors** | HIGH | Medium | Version all factors. Alert if factor age > 30 days. Cross-check multiple sources. Verifier flags outcomes that diverge > 3σ from estimates. |
| 3 | **Rebound effect: teams schedule more jobs because "green"** | MEDIUM | High | Track total emissions trend, not just per-job efficiency. Flag teams whose total footprint grows > 20% while claiming savings. Report absolute numbers alongside intensity metrics. |
| 4 | **Teams game points by marking jobs "sustainable" to enable deferral** | MEDIUM | High | Workload category set by policy (not self-reported) based on service tier, not developer choice. Audit random sample of category labels monthly. |
| 5 | **Carbon reduction achieved but cost spikes** | HIGH | Medium | Planner jointly optimizes cost+carbon. Hard constraint: no recommendation that increases cost by > X% without explicit approval. Dashboard shows both metrics side-by-side. |
| 6 | **Executor creates broken PRs / scheduler changes** | HIGH | Low | Canary deployments for production changes. Automated rollback on health check failure. PR requires CI pass before merge. |
| 7 | **Counterfactual inflation: system claims more savings than real** | HIGH | Medium | Use conservative counterfactual (lower bound of CI). External audit of verification methodology. Compare system-reported savings against actual billing trend. |
| 8 | **LLM hallucinates in policy parsing → wrong constraints** | HIGH | Low | LLM output → deterministic validator. Parsed constraints shown to human before activation. Confidence score; low-confidence → human review. |
| 9 | **Single point of failure: one agent down → loop breaks** | MEDIUM | Medium | Health checks on each agent. Circuit breaker: if Ingestor down > 2 hours, alert. Planner uses last-known-good data. System degrades gracefully (no optimization, but no breakage). |
| 10 | **Notification fatigue: developers ignore all nudges** | LOW | High | Rate limit: max 2 nudges/dev/week. A/B test nudge formats. Allow developers to set "do not disturb" windows. Track engagement metrics; if < 10%, redesign. |
| 11 | **Data exfiltration: activity ledger reveals business secrets** | MEDIUM | Low | Role-based access control on all stores. Team-level data visible only to team members + platform team. Aggregation for cross-team views. Audit log on all data access. |
| 12 | **Carbon price manipulation: someone changes $75/ton to $0** | HIGH | Low | Policy store is versioned and append-only. Carbon price changes require governance committee approval. Alert on any parameter change. |
| 13 | **Solver finds "optimal" solution that's actually corner case** | MEDIUM | Medium | Bound all decision variables. Sanity checks on solver output (e.g., can't move all jobs to one region). Human review of first 20 recommendations. |
| 14 | **Provider-specific outage makes "optimal" region unavailable** | HIGH | Low | Real-time availability check before execution. Fallback region list. Executor retries with next-best option. |

---

## 8. "Recruiter-worthy" Differentiators

### Differentiator A: Counterfactual Verification with Evidence Chains

**What**: Every claimed carbon reduction has a machine-verifiable evidence chain:
- Pre-action snapshot (activity record, config, emission factor)
- Post-action measurement (actual usage, actual grid intensity)
- Counterfactual computation (what would have happened without intervention)
- Confidence interval (how sure we are)

**Why it's impressive**: This is MRV (Measurement, Reporting, Verification) — the gold standard in carbon markets. Most "green cloud" tools just show estimates. This system can *prove* its impact, or honestly say "we're not sure."

**AI engineering angle**: The evidence chain is a structured reasoning trace — similar to chain-of-thought but for empirical claims. Builds toward auditable AI decision-making.

---

### Differentiator B: Bandit Learning for Policy Tuning

**What**: Use a contextual multi-armed bandit to learn which recommendation strategies work best for which workload/team contexts.

- Arms: {region_shift, time_shift, right_size, spot_convert, do_nothing}
- Context: workload type, team, time of day, grid forecast confidence
- Reward: verified_kgCO₂e_saved × acceptance_rate - sla_violation_penalty

Thompson sampling with Beta priors. Update weekly from Verifier outcomes.

**Why it's impressive**: The system doesn't just optimize — it *learns what optimizations teams actually accept and that actually work.* Closes the gap between theoretical optimality and organizational reality.

**AI engineering angle**: Online learning from human feedback in a real system. Demonstrates understanding of exploration-exploitation tradeoff.

---

### Differentiator C: Uncertainty-Aware Decision Making

**What**: Propagate uncertainty through the entire pipeline — from grid intensity → emissions → counterfactuals → recommendations.

- Recommendations include confidence intervals, not point estimates
- Planner uses robust optimization: optimize for worst-case in uncertainty set
- System self-calibrates: if 90% CIs contain actual only 70% of the time, widen them

**Why it's impressive**: Most systems pretend their numbers are exact. This one is epistemically honest and self-correcting. Shows understanding of decision-making under uncertainty.

**AI engineering angle**: Combines Bayesian reasoning, robust optimization, and calibration — core ML engineering concepts applied to a real system.

---

### Recommendation for 6-10 Week School Timeline

**Build Differentiator A (Counterfactual Verification) as the core, with a taste of C (Uncertainty).**

**Rationale**:

| Option | Feasibility (6-10 wks) | Impressiveness | Risk |
|--------|----------------------|----------------|------|
| A: Counterfactual MRV | ★★★★★ | ★★★★☆ | Low — well-defined math |
| B: Bandit Learning | ★★★☆☆ | ★★★★★ | Medium — needs enough data to learn |
| C: Uncertainty-Aware | ★★★★☆ | ★★★★☆ | Low — interval arithmetic is straightforward |
| A + C combined | ★★★★☆ | ★★★★★ | Low — natural fit |

**Why A+C**: 
- Verification is the *hardest part* that most projects skip. Building it well is a genuine differentiator.
- Uncertainty propagation naturally fits into verification (confidence intervals on savings claims).
- Bandit learning (B) is sexy but needs weeks of simulated operational data to show anything meaningful — and it's hard to demo convincingly in a school setting without a running system.
- A+C gives you a system that can say: *"We moved 47 training jobs from Virginia to Oregon over 4 weeks, saving an estimated 18.3 kgCO₂e [90% CI: 12.1–24.5], verified against counterfactual emissions of 31.7 kgCO₂e. No SLA violations detected."* That's a powerful demo statement.

### Suggested 6-10 Week Timeline

| Week | Focus | Deliverable |
|------|-------|-------------|
| 1-2 | System design finalization + simulation scaffold | Design doc (this), synthetic workload generator, carbon intensity simulator |
| 3-4 | Core pipeline: Ingestor → Accountant → Planner (rule-based MVP) | Working pipeline with synthetic data, baseline emissions computed |
| 5-6 | Executor (mock) + Verifier with counterfactual engine | End-to-end loop running on simulation, verified outcomes generated |
| 7-8 | Uncertainty propagation + evidence chain generation | Confidence intervals on all claims, evidence chain JSON for audit |
| 9 | Dashboard + Developer Copilot (light version) | Visualization of trade-off curves, team leaderboard |
| 10 | Demo prep, write-up, sensitivity analysis | Final presentation, report, ablation studies (vary carbon price, workload mix) |

---

## Appendix: Key Assumptions Summary

| Item | Value | Type | Source/Justification |
|------|-------|------|---------------------|
| Organization size | 100 developers | Given | Problem statement |
| Internal carbon price | $75/ton | Given | Problem statement |
| Daily job volume | ~990 | Assumption | ~10 jobs/dev/day is typical for CI-heavy orgs |
| PUE | 1.1 | Assumption | Google, AWS sustainability reports (hyperscaler avg) |
| TDP per vCPU | 0.005 kW | Assumption | ~200W server / 40 vCPUs, simplified |
| GPU TDP | 0.3 kW | Assumption | NVIDIA A100 TDP = 300W |
| us-east-1 grid intensity | 350 gCO₂/kWh | Known | EPA eGRID 2022, PJM region |
| us-west-2 grid intensity | 90 gCO₂/kWh | Known | EPA eGRID 2022, BPA/hydro |
| eu-north-1 grid intensity | 30 gCO₂/kWh | Known | Swedish Energy Agency |
| Spot discount | 60-70% | Known | AWS spot pricing history |
| Cross-region egress | $0.01-0.09/GB | Known | AWS data transfer pricing |
| Flexible workload share | 70% | Assumption | Conservative; many orgs have more |
| Achievable reduction | 21% overall | Assumption | Conservative for MVP scope |
