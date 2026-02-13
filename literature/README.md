# Literature Review — sust-AI-naible

## How to use this folder
Each team member should add 2-3 papers/tools with structured notes.
Use the template below. Focus on **what you learned that changed your design thinking**.

---

## Paper 1: Carbon-Aware Computing for Datacenters
- **Authors:** Radovanović, A., et al. (Google)
- **Year:** 2022
- **Venue:** IEEE Transactions on Power Systems
- **Reviewed by:** [Team Member Name]

### Summary
Google demonstrated 10-40% carbon reduction in their data centers by shifting
compute workloads across time and regions to match renewable energy availability.
They used carbon-aware scheduling for delay-tolerant batch workloads.

### Key Findings
- Temporal shifting (time of day) gives 10-20% reduction for flexible workloads
- Spatial shifting (across regions) gives 20-40% for highly flexible workloads
- Real-time carbon signals (like Electricity Maps) are essential for decision-making
- Only works for delay-tolerant workloads — production services can't be shifted

### Gap / Limitation
- No public verification methodology — they claim savings but don't show how
  an external auditor would confirm them
- No cost analysis — unclear if carbon shifting increased cloud costs
- Single-cloud (Google only) — not generalizable to multi-cloud orgs

### Design Implication for Our Project
- We build counterfactual verification into the core (not just claim savings — prove them)
- We model cost explicitly alongside carbon (joint optimization)
- We categorize workloads by flexibility (urgent/balanced/sustainable)

---

## Paper 2: Recalibrating Global Data Center Energy-Use Estimates
- **Authors:** Masanet, E., Shehabi, A., Lei, N., Smith, S., Koomey, J.
- **Year:** 2020
- **Venue:** Science, 367(6481), 984-986
- **Reviewed by:** [Team Member Name]

### Summary
Authoritative estimate that data centers consumed ~1% of global electricity in 2018
(~205 TWh). Despite a 550% increase in compute output 2010-2018, energy use only
grew 6% due to efficiency gains. Hyperscale data centers are more efficient (PUE ~1.1)
than enterprise facilities (PUE ~1.6).

### Key Findings
- Cloud migration itself reduces carbon (hyperscale PUE 1.1 vs enterprise 1.6)
- But growth in AI/ML workloads may overwhelm efficiency gains
- Regional grid mix is the dominant factor in per-kWh emissions
- Embodied carbon (manufacturing) is 10-20% of lifecycle emissions

### Gap / Limitation
- Aggregate estimates — doesn't help individual organizations measure their footprint
- No guidance on per-workload attribution
- PUE varies significantly even within hyperscale providers

### Design Implication for Our Project
- We use PUE = 1.1 as our baseline assumption (hyperscale), with sensitivity analysis
- We acknowledge embodied carbon as out of scope for v1 (flag as limitation)
- Regional grid intensity is our primary optimization lever

---

## Paper 3: AutoGen — Enabling Next-Gen LLM Applications via Multi-Agent Conversation
- **Authors:** Wu, Q., Bansal, G., Zhang, J., et al. (Microsoft Research)
- **Year:** 2023
- **Venue:** arXiv preprint / ICLR 2024
- **Reviewed by:** [Team Member Name]

### Summary
Framework for building multi-agent systems where LLM-powered agents collaborate
through conversation. Agents have roles, tool access, and can be composed into
workflows. Key insight: separating agent responsibilities enables more reliable
and auditable AI systems.

### Key Findings
- Role specialization (each agent has one clear purpose) outperforms monolithic agents
- Human-in-the-loop gates are essential for high-stakes decisions
- Tool use (calling functions/APIs) makes agents more reliable than pure generation
- Conversation-based coordination naturally produces audit trails

### Gap / Limitation
- Focused on general-purpose tasks — no domain-specific carbon/sustainability application
- No discussion of deterministic boundaries (when should an agent NOT use the LLM?)
- Evaluation is largely qualitative

### Design Implication for Our Project
- We adopt the multi-agent pattern with role specialization (7 agents)
- We add explicit AI/deterministic boundaries: LLM for reasoning, math for computation
- Our agents have tools (emissions calculator, cost model, optimizer) not just conversation
- The orchestrator pattern coordinates agents without requiring them to "talk" to each other

---

## Paper 4: GHG Protocol Corporate Standard
- **Authors:** WRI / WBCSD
- **Year:** 2004 (revised 2015)
- **Venue:** Greenhouse Gas Protocol
- **Reviewed by:** [Team Member Name]

### Summary
The de facto global standard for corporate greenhouse gas accounting. Defines Scope 1
(direct), Scope 2 (electricity), and Scope 3 (supply chain) emissions. Cloud computing
falls under Scope 2 (location-based) and Scope 3 (market-based) for most organizations.

### Key Findings
- Location-based method: use grid average emission factors
- Market-based method: use supplier-specific or contractual factors
- Uncertainty must be reported — not optional
- "Avoided emissions" (what we call savings) need a credible baseline/counterfactual

### Gap / Limitation
- Written for traditional enterprises, not cloud-native organizations
- No guidance on per-workload attribution within a cloud account
- "Counterfactual" methodology for avoided emissions is vague

### Design Implication for Our Project
- We use location-based methodology (grid intensity × energy) as our primary approach
- We report uncertainty bounds on every emissions number (±20% from grid intensity)
- Our counterfactual verification directly addresses the GHG Protocol's "avoided emissions" 
  requirement — this is our strongest compliance argument

---

## Tool Review 1: Cloud Carbon Footprint (Thoughtworks, OSS)
- **URL:** cloudcarbonfootprint.org
- **Reviewed by:** [Team Member Name]

### What it does well
- Multi-cloud support (AWS, GCP, Azure) in one tool
- Open-source — transparent methodology
- Billing-based estimation (works without infrastructure access)
- Clean dashboard with per-service breakdown

### Where it falls short
- **Measurement only** — no optimization recommendations
- No automation — requires manual analysis of results
- No counterfactual verification — can't prove savings
- Emission factors are static (not real-time grid intensity)
- No workload-level granularity (service-level only)

### Design implication
- We go beyond measurement into optimization + verification
- We use real-time (simulated) grid intensity, not static annual factors
- We provide per-job granularity, not just per-service aggregates

---

## Tool Review 2: Electricity Maps
- **URL:** electricitymaps.com
- **Reviewed by:** [Team Member Name]

### What it does well
- Real-time grid carbon intensity for 200+ zones worldwide
- Historical data for analysis and forecasting
- API access for programmatic use
- Beautiful visualization of grid conditions

### Where it falls short
- **Signal only** — provides data, doesn't act on it
- No integration with cloud schedulers or CI/CD
- No workload-awareness (doesn't know what's flexible)
- Uncertainty of real-time estimates not well-documented

### Design implication
- Electricity Maps (or WattTime) would be our data source in production
- In simulation, we generate synthetic intensity data matching their patterns
- We add the "action layer" that Electricity Maps lacks — the agents that 
  use this signal to actually move workloads

---

## Synthesis: 4 Themes Across the Literature

### 1. Measurement exists. Optimization + verification doesn't.
Every tool and paper we reviewed can MEASURE emissions. None close the loop 
to automatically optimize and verify. That's the gap.

### 2. Multi-objective optimization is required, not optional.
Papers that optimize carbon alone report cost spikes and SLA violations.
The problem is inherently multi-objective: carbon × cost × latency.

### 3. Uncertainty is the elephant in the room.
Grid intensity varies ±30%, PUE estimates are coarse, and counterfactuals
are inherently uncertain. Systems that report point estimates without 
confidence intervals are not credible.

### 4. Agent architectures match this problem's structure.
The problem has natural role decomposition (measure, plan, approve, execute, verify).
Multi-agent systems handle this better than monolithic approaches.
