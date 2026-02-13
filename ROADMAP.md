# sust-AI-naible — Practical Roadmap

## The 3-Sentence Pitch

> We built a multi-agent system that watches cloud workloads, calculates their carbon
> footprint, recommends cheaper+greener scheduling, executes those changes, and then
> *proves* the savings were real using counterfactual analysis — not vibes.

---

## Phase 0: Understand What You're Actually Building (Day 1)

We're building **5 things that talk to each other**:

```
FAKE CLOUD        →  MEASURE IT  →  PLAN BETTER  →  DO IT  →  PROVE IT WORKED
(simulator)          (accountant)    (planner)       (executor)  (verifier)
```

That's it. Everything else is details.

---

## Phase 1: The Fake Cloud (Week 1-2)

**Why first**: You can't optimize what you can't measure. You can't measure what doesn't exist.
Since you don't have a real cloud account to instrument, you build a synthetic one.

### What to build:
```
src/
  simulator/
    workload_generator.py    ← generates fake jobs (CI builds, training runs, etc.)
    carbon_intensity.py      ← generates fake grid carbon data per region per hour
    cost_model.py            ← simple lookup: region + instance type → $/hour
    clock.py                 ← simulated time (so you can fast-forward weeks in seconds)
```

### Workload Generator — what it does:
- Every simulated "hour", generate N jobs with random properties
- Each job has: name, team, duration, vCPUs, region, category (urgent/balanced/sustainable)
- Use simple distributions (Poisson for arrival rate, log-normal for duration)
- Output: list of Job objects or rows in a SQLite table

### Carbon Intensity — what it does:
- For each region, generate an hourly carbon intensity (gCO₂/kWh)
- Use a sine wave (clean during day for solar regions, clean at night for wind) + noise
- Hard-code 5 regions with different base intensities:
  - us-east-1: ~350 (dirty, coal+gas)
  - us-west-2: ~90 (clean, hydro)
  - eu-west-1: ~300 (mixed)
  - eu-north-1: ~30 (very clean, hydro+nuclear)
  - ap-south-1: ~700 (very dirty, coal)

### Cost Model — what it does:
- Lookup table: {region, instance_type} → $/hour
- Add cross-region egress cost: $0.02/GB if job moves regions
- That's it. Don't overcomplicate.

### Deliverable: 
Run the simulator for 30 simulated days → get a CSV of ~30,000 jobs with 
timestamps, costs, regions, durations. This is your "activity ledger."

---

## Phase 2: The Carbon Accountant (Week 2-3)

### What to build:
```
src/
  agents/
    carbon_accountant.py     ← takes jobs + grid intensity → emits kgCO₂e per job
```

### The one formula you need:
```
kgCO₂e = vCPUs × duration_hours × 0.005 kW/vCPU × 1.1 (PUE) × grid_intensity_kgCO₂/kWh
```

### What it does:
- Read each job from the activity ledger
- Look up the grid intensity for that job's region and time
- Compute emissions with uncertainty bounds (±20% as a starting assumption)
- Write results to an emissions table

### Deliverable:
For your 30,000 simulated jobs, you now have: total baseline emissions, 
emissions by region, emissions by team, emissions by hour-of-day.
Make a few matplotlib charts. This is already demo-worthy.

---

## Phase 3: The Planner (Week 3-5)

### What to build:
```
src/
  agents/
    planner.py               ← generates recommendations: "move job X to region Y"
```

### Start with the dumb version (Rule-Based, Formulation A):
```python
for each job in upcoming_schedule:
    if job.category == "urgent":
        skip  # don't touch
    
    current_emissions = compute_emissions(job, job.region, job.time)
    
    best_region = job.region
    best_time = job.time
    best_score = current_emissions * 75/1000 + job.cost  # effective cost
    
    for candidate_region in allowed_regions(job):
        for candidate_time in allowed_times(job):  # within deferral window
            e = compute_emissions(job, candidate_region, candidate_time)
            c = compute_cost(job, candidate_region) + egress_cost(job, candidate_region)
            score = e * 75/1000 + c
            if score < best_score:
                best_region, best_time, best_score = candidate_region, candidate_time, score
    
    if best_region != job.region or best_time != job.time:
        emit_recommendation(job, best_region, best_time, estimated_savings)
```

### Deliverable:
Run planner on the simulated workload → list of recommendations with estimated savings.
Then RE-RUN the simulation applying those recommendations → compare baseline vs optimized.

---

## Phase 4: The Verifier (Week 5-7)

### What to build:
```
src/
  agents/
    verifier.py              ← computes counterfactual: "what WOULD have happened?"
```

### The counterfactual logic:
```
For a job that was moved from region A → region B:
  
  counterfactual_emissions = actual_resource_usage × emission_factor_of_ORIGINAL_region_at_ACTUAL_time
  actual_emissions         = actual_resource_usage × emission_factor_of_NEW_region_at_ACTUAL_time
  verified_savings         = counterfactual - actual
  
  confidence_interval: propagate uncertainty from grid intensity bounds
```

### Why this matters:
- Without verification, you're just making promises
- With verification, you can say: "We claimed we'd save 18 kg. We actually saved 15.2 kg [CI: 11–19]."
- This is MRV (Measurement, Reporting, Verification) — the same concept carbon markets use

### Deliverable:
For every executed recommendation, a verification record with:
measured savings, confidence interval, evidence chain (what data went in, what formula, what came out).

---

## Phase 5: The Glue — LLM Integration (Week 7-8)

### What to build:
```
src/
  agents/
    executor.py              ← generates mock tickets/PRs from recommendations
    copilot.py               ← generates human-readable summaries + nudges
    governance.py             ← evaluates risk level, auto-approve or flag
```

### This is where the LLM comes in:
- Executor: "Take this recommendation JSON → generate a mock Jira ticket body"
- Copilot: "Take this week's verification results → write a team summary"
- Governance: "Parse this policy text → extract constraints for the solver"

### Why last?
The system works WITHOUT the LLM. The LLM makes it usable by humans.
This ordering means you always have something working, and the LLM is additive.

---

## Phase 6: Dashboard + Demo (Week 8-10)

### What to build:
- Streamlit or Gradio dashboard (simplest options)
- Show: baseline vs optimized emissions, cost, trade-off curves
- Show: recommendation list with verification status
- Show: evidence chain for any claimed saving (click to expand)
- Show: team leaderboard (gamification)

---

## The Tech Stack

| Component | Tool | Why |
|-----------|------|-----|
| Language | Python 3.11+ | You know it. Ecosystem is great. |
| Database | SQLite (→ PostgreSQL later if needed) | Zero setup. Good enough for simulation. |
| Simulation | NumPy + Pandas | Standard. |
| Optimization | OR-Tools or just Python loops (MVP) | OR-Tools is free, but start without it. |
| LLM | OpenAI API or local Ollama | Either works. Ollama = free + no API key. |
| Charts | Matplotlib + Plotly | Matplotlib for analysis, Plotly for dashboard. |
| Dashboard | Streamlit | One file, instant web app. |
| Version control | Git | Obviously. |

---

