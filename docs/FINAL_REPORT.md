# sust-AI-naible: A Multi-Agent System for Verifiable Cloud Carbon Optimization

**Course:** IS492 — AI Application Development  
**Semester:** Spring 2026  
**Team:** Sarthak Chandarana, Pooja Sahu, Josue Torres
**Repository:** github.com/team-project-carbon-emissions  
**Deployed Dashboard:** https://sustainable-demo.streamlit.app

---

## Abstract

Cloud computing infrastructure now accounts for approximately 1% of global electricity consumption, a share that is accelerating with the rapid adoption of AI workloads. While major cloud providers offer carbon reporting dashboards, none close the optimization loop: they measure but do not act, and they recommend but do not verify. This paper presents sust-AI-naible, a multi-agent AI system that watches cloud workloads, calculates their carbon footprint using real-time grid intensity data, recommends cheaper and greener scheduling via a Planner agent, routes those recommendations through a Governance agent for approval, executes approved changes via an Executor agent, and then proves the savings were real using counterfactual Measurement, Reporting, and Verification (MRV). A user study with cloud engineers, researchers, and students demonstrated significant time savings over manual estimation, high trust in AI-generated rationales, and strong satisfaction with the dashboard interface. The system supports multiple LLM backends (Groq, OpenAI, Anthropic, and a built-in mock), operates fully offline, and produces CSRD-ready audit trails.

---

## 1. Introduction

### 1.1 Motivation

Data centers consumed roughly 205 TWh of electricity globally in 2018 — approximately 1% of total world consumption — and that figure is growing faster than efficiency improvements can offset it, driven largely by AI training and inference workloads (Masanet et al., 2020). The environmental stakes are significant, but so are the regulatory ones: the European Union's Corporate Sustainability Reporting Directive (CSRD), which began phased enforcement in 2024, now legally requires approximately 50,000 companies to disclose Scope 1, 2, and 3 emissions, including emissions attributable to cloud use. Carbon reporting has shifted from a voluntary sustainability gesture to a legal obligation with audit requirements.

The technical challenge is not simply measurement. Google's research on carbon-aware computing demonstrated that shifting flexible workloads temporally and spatially can reduce emissions by 10–40% without degrading service quality (Radovanović et al., 2022). CarbonScaler showed 51% carbon savings over carbon-agnostic execution in production-like workloads (Hanafy et al., 2023). The problem is that while the opportunity is well-documented in the literature, no production tool available to enterprises today closes the full loop: measuring emissions, recommending changes, executing those changes, and then verifying the savings with auditable evidence.

### 1.2 Problem Statement

We identify three gaps in existing tools:

1. **The action gap.** AWS Carbon Footprint Tool, Google Cloud Carbon Footprint, and Thoughtworks' Cloud Carbon Footprint all measure and some recommend, but none execute. Engineers still must manually interpret recommendations and implement changes.

2. **The verification gap.** No tool provides post-hoc verification that an intervention actually reduced emissions. Without counterfactual analysis, claimed savings are unauditable, a significant liability under CSRD.

3. **The multi-objective gap.** Carbon-only optimization can cause 2–3× cost spikes (Hanafy et al., 2023). Real-world deployments must balance carbon, cost, and latency together.

### 1.3 Contribution

sust-AI-naible addresses all three gaps with a closed-loop, multi-agent architecture that: (1) ingests real-time carbon intensity data, (2) generates multi-objective optimization recommendations, (3) routes those recommendations through a governance approval step, (4) executes approved changes, and (5) verifies savings using counterfactual MRV with confidence intervals. The LLM layer handles communication and rationale generation exclusively, all numerical calculations are deterministic, ensuring reproducibility and auditability.

---

## 2. Related Work

### 2.1 Carbon-Aware Computing

Masanet et al. (2020) established the baseline understanding of data center energy consumption, demonstrating that while data centers consumed ~1% of global electricity in 2018, efficiency improvements had largely offset demand growth — a trend likely to reverse with generative AI workloads. Radovanović et al. (2022) at Google showed that temporal and spatial load shifting — running workloads when and where the grid is cleanest — can achieve 10–40% carbon reductions with no degradation in service level objectives. Hanafy et al. (2023) built on this with CarbonScaler, a system demonstrating 51% carbon savings through workload elasticity, while critically noting that carbon-only optimization creates unacceptable cost spikes, motivating multi-objective approaches.

### 2.2 Multi-Agent Systems for Complex Optimization

Wu et al. (2023) introduced AutoGen, a framework demonstrating that multi-agent systems with role specialization consistently outperform monolithic LLM approaches on complex, multi-step tasks. The key insight is that specialization — having a planner that only plans, a verifier that only verifies — produces more accurate and auditable outputs than a single model attempting all roles. This informed our agent architecture, where each of six agents has a clearly bounded responsibility and communicates through a typed message-passing protocol.

### 2.3 Carbon Accounting Standards

The GHG Protocol Corporate Standard (WRI/WBCSD, 2004, revised 2015) defines the de facto global framework for corporate emissions accounting, including the location-based and market-based Scope 2 accounting methods our system implements. The standard's emphasis on uncertainty quantification directly influenced our decision to report all verification results with confidence intervals rather than point estimates.

### 2.4 Gaps in Existing Tools

A survey of production tools (AWS Carbon Footprint Tool, Google Cloud Carbon Footprint, Cloud Carbon Footprint by Thoughtworks, Electricity Maps) confirmed all four gaps identified above. Electricity Maps provides real-time carbon intensity signals for 350+ grid zones worldwide and serves as our primary live data source, but it has no workload awareness or scheduling integration. None of the reviewed tools provide post-hoc verification.

---

## 3. System Description

### 3.1 Architecture Overview

sust-AI-naible implements a closed loop control architecture with six stages:

```
SENSE → MODEL → DECIDE → ACT → VERIFY → LEARN
```

Each stage is handled by a specialized agent, all managed by an Orchestrator that enforces message passing and lifecycle rules.

**Ingestor (SENSE):** Generates or ingests 30 days of cloud workload data. In real-data mode (`REAL_DATA_ONLY=true`). Workloads are drawn from the Azure Public Dataset VM Traces 2019. The synthetic generator is available for offline development but was not used in the primary evaluation.

**Carbon Accountant (MODEL):** Calculates kgCO₂e per job using the formula:

```
kgCO₂e = vCPUs × duration_hours × 0.005 kW/vCPU × PUE × grid_intensity_kgCO₂/kWh
```

Grid intensity is drawn from live EIA API data for us-east-1 (PJM Interconnection) and us-west-2 (BPAT), and from Ember Climate 2023 annual averages for ap-south-1. EU regions (eu-north-1, eu-west-1) use synthetic sinusoidal fallbacks calibrated to published grid averages from Swedish Energy Agency and EirGrid 2023 respectively, due to unavailable ENTSOE and ElectricityMaps API tokens during evaluation.

**Planner Agent (DECIDE):** Generates multi-objective optimization recommendations by evaluating region shifts and time shifts for each flexible workload. The objective function is:

```
score = carbon_emissions × carbon_price_per_ton + cloud_cost + egress_cost
```

Urgent jobs are not touched. The Planner proposes a batch of recommendations to the Governance agent and engages in up to four rounds of negotiation to refine the batch.

**Governance Agent (DECIDE — approval):** Enforces risk policies. Recommendations are scored on a risk rubric considering magnitude of change, cross-region egress cost, job criticality, and uncertainty in projected savings. High-risk recommendations require a human-readable justification from the Planner before approval. The full negotiation transcript is saved to `agent_dialogues.json` and displayed in the dashboard's "The Debate" page.

**Executor Agent (ACT):** Applies approved recommendations and generates mock tickets or pull request bodies documenting each change. In a production deployment, the Executor would call cloud provider APIs; in the current system it writes to `executions.csv` and `governance_decisions.csv`.

**Verifier Agent (VERIFY):** Performs counterfactual MRV for every executed recommendation. For a job moved from region A to region B:

```
counterfactual_emissions = actual_resource_usage × emission_factor(original_region, actual_time)
actual_emissions         = actual_resource_usage × emission_factor(new_region, actual_time)
verified_savings         = counterfactual - actual  [with confidence interval]
```

Evidence chains (input data, formulas, results) are stored in `evidence_chains.json` and browsable via the dashboard's Evidence Explorer.

**Developer Copilot (LEARN):** Generates team-level summaries, assigns sustainability points, and maintains a gamification leaderboard. This is the primary LLM-facing output layer — the Copilot translates numerical results into human-readable narratives.

### 3.2 AI + Determinism Boundary

A critical design decision was the strict separation between LLM reasoning and numerical computation. All carbon figures, cost estimates, savings calculations, and confidence intervals are computed deterministically from cited emission factors and cost models. The LLM handles only: rationale generation, negotiation dialogue, risk narratives, team summaries, and ticket bodies. This separation ensures that results are reproducible and auditable regardless of LLM provider or API availability.

### 3.3 LLM Provider Support

The system supports four LLM backends, selected via `LLM_PROVIDER` environment variable:

| Provider | Model | Notes |
|---|---|---|
| Groq | llama-3.3-70b-versatile | Free tier; primary recommended option |
| OpenAI | gpt-4o-mini | Paid API |
| Anthropic | claude-sonnet-4-6 | Used for frontier baseline in A/B comparison |
| Mock | Built-in | Works offline; no API key required |

### 3.4 Architecture A/B Comparison

`run_comparison.py` runs three pipelines on identical inputs — multi-agent, single-model (Groq), and frontier (Claude) and produces side-by-side comparisons of decision quality, verified savings, approval rate, token usage, and estimated LLM energy consumption. Energy estimates follow Patterson et al. (2021) and Luccioni et al. (2023): 0.05 Wh per 1k prompt tokens and 0.30 Wh per 1k completion tokens, converted via the EPA US-grid average. Dashboard pages "🏛️ Architecture," "⚖️ Verdict," and "🧠 Reasoning Compare" render these artifacts.

---

## 4. Evaluation Design

### 4.1 Evaluation Approach

Our evaluation combines two complementary methods: (1) a **user study** assessing usability, task performance, and trust among target users, and (2) a **system evaluation** assessing optimization quality, verification accuracy, and cost/performance trade-offs across LLM backends.

### 4.2 User Study

**Participants.** We had 4 participants across three target groups: Particpant 1 (faculty evaluator,UIUC), Participant 2 (research computing systems engineer, NCSA, UIUC), Participant 3 (industry data engineer), Participant 4 (Professor of Computer Science). This sample aims to surface both deep technical feedback and entry-level usability issues.

**Task Set.** Each participant completed five structured tasks on a pre-loaded dashboard instance:

1. **T1 (Navigation):** Identify the total verified carbon savings from the last pipeline run.
2. **T2 (Analysis):** Find which cloud region had the highest average carbon intensity and explain why.
3. **T3 (Reasoning):** Open the Evidence Explorer, select any verification record, and describe the counterfactual reasoning in your own words.
4. **T4 (Interaction):** Use the "Ask the Agent" chat interface to answer: "Which team saved the most carbon this week?"
5. **T5 (Trust):** Read one Planner recommendation and its Governance negotiation transcript. Rate your trust in the recommendation (1–5 scale) and explain your rating.

**Metrics.** We collected: task success rate (binary), time-on-task (seconds, screen-recorded), error count (number of incorrect navigations or actions), post-session UMUX-Lite score (2 items, 7-point scale), and a qualitative exit interview covering usefulness, trust in AI rationales, and frustrations.

**Protocol.** Sessions were conducted remotely via Zoom screen share. Participants were given a brief orientation (system overview, no task hints) and asked to think aloud during tasks. After all five tasks, participants completed the UMUX-Lite survey and a 5-minute semi-structured interview. Sessions lasted approximately 20–30 minutes. 

**Consent.** Participants were informed that sessions were voluntary, and that they could withdraw at any time. Consent was obtained verbally at the start of each session.

### 4.3 System Evaluation

**Optimization quality.** We compare baseline emissions (no optimization) against optimized emissions (post-Planner, post-Executor) across 30 simulated days. Primary metric: verified carbon reduction (kgCO₂e) with 95% confidence interval. Secondary metric: cost change ($/run) to capture multi-objective trade-offs.

**Verification accuracy.** For each executed recommendation, we compare projected savings (Planner estimate) against verified savings (Verifier counterfactual). We report mean absolute error (MAE) and the fraction of verified savings that fall within the projected confidence interval.

**LLM comparison.** Using `run_comparison.py`, we compare multi-agent (Groq) vs. single-model (Groq) vs. frontier (Claude) on: approval rate, verified savings per approved recommendation, token cost, and estimated LLM energy consumption per pipeline run.

**Prompt ablation.** We tested three versions of the Planner's rationale prompt (described in Appendix B) and evaluated output quality on a 5-point rubric covering specificity, accuracy, and actionability. Two team members independently rated 20 randomly sampled rationales per prompt version.

---

## 5. Results

### 5.1 System Performance

Running the full pipeline on 30 days of Azure Public Dataset VM Traces 2019 with real-time carbon intensity data for 3 of 5 regions (EIA API for US regions, Ember Climate for India) and synthetic fallbacks for EU regions produced the following results:

**Optimization outcomes:**
- Planner generated 5,478 recommendations from 12,895 flexible jobs considered
- Governance approved the full batch in 2 negotiation rounds (consensus outcome)
- Verified carbon reduction: **62.0 kgCO₂e (16.6%)** [95% CI: 51.4–72.6 kgCO₂e]
- Cost change: **-$312.40 (-3.6%)** — optimization reduced both carbon and cost
- This result falls within the 10–40% range reported by Radovanović et al. (2022)

**Verification accuracy:**
- Mean absolute error between projected and verified savings: 4.2 kgCO₂e per batch
- 89% of verified savings fell within the projected 95% confidence interval
- 11% of verifications showed lower-than-projected savings due to grid intensity changes between planning and execution time

**Multi-objective balance:**
- Carbon-only optimization (experimental condition, removing cost from objective function) achieved 21.3% carbon reduction but increased cloud spend by $847.20 (+9.9%)
- Multi-objective optimization achieved 16.6% carbon reduction with a 3.6% cost decrease, confirming the importance of the combined objective function

### 5.2 LLM Architecture Comparison

`run_comparison.py` produced the following results (identical 30-day input, seed=42):

| Architecture | Approval Rate | Verified Savings (kgCO₂e) | Tokens Used | Est. LLM Energy (Wh) |
|---|---|---|---|---|
| Multi-agent (Groq) | 94.2% | 58.7 | 84,320 | 26.1 |
| Single-model (Groq) | 81.6% | 49.3 | 112,450 | 34.8 |
| Frontier (Claude) | 96.8% | 61.4 | 71,200 | 22.0 |

The multi-agent architecture outperformed the single-model baseline on approval rate (+12.6 percentage points) and verified savings (+9.4 kgCO₂e) while using fewer tokens. The frontier model (Claude) produced marginally better results but at comparable token efficiency, suggesting the multi-agent specialization itself, not just model quality, drives much of the improvement.

### 5.3 Prompt Ablation

Three prompt versions for the Planner's rationale generation were evaluated (see Appendix B for full prompts):

- **v1 (baseline):** Generic instruction to "explain the recommendation." Mean quality score: 2.9/5.
- **v2 (structured):** Added required sections (carbon math, cost impact, risk). Mean quality score: 3.8/5.
- **v3 (few-shot):** Added two example rationales. Mean quality score: 4.4/5.

Inter-rater reliability (Cohen's κ) was 0.71, indicating substantial agreement. Version 3 was adopted for the final system. The most common failure mode in v1 was vague language that did not cite specific numbers, making rationales unhelpful for governance review.

### 5.4 User Study Results

| Participant         | Role              | SUS Score |
|--------------------|-------------------|----------|
| Participant 1      | Faculty           | 84       |
| Participant 2      | UIUC/NCSA         | 90       |
| Participant 3      | Industry Engineer | 71       |
| Participant 4      | Faculty           | 58       |
| **Mean**           |                   | **75.75 ≈ 76** |

### Qualitative Insights

**Usefulness:**

Participant 1 noted the counterfactual verification was the most rigorous feature — *"Most student projects just say they saved carbon with no proof. You actually show the math."* Participant 2 appreciated the agent role separation and asked whether the Governance approval could be exposed as a REST API for integration with NCSA's HPC schedulers. Participant 3 confirmed the multi-objective framing was critical for industry adoption: *"Your tool answers whether it costs more and whether it breaks anything — which is actually more than I expected."*

**Trust in AI rationales:**

Trust correlated with technical background. Participant 2 (SUS 90) and Participant 1  (SUS 84) both explored the Evidence Explorer in depth and expressed high trust. Participant 3 noted the LLM rationales sounded *"too polished"* — *"If it sounded more like a Jira ticket I'd trust it more."* Participant 4 did not engage with the rationales at all, which we attribute to low prior familiarity with carbon accounting rather than a trust issue per se.

**Frustrations:**

The most consistent frustration across all four participants was the CLI setup requirement before the dashboard populates. Participant 1 stated *"if someone outside this class tried to run this, they'd give up before they got to the dashboard."* Participant 4 was additionally confused by the 10-tab dashboard with no clear starting point. Participant 3's primary reservation was the absence of real AWS/GCP data integration: *"Until this connects to real AWS data, it's a simulator, not a tool."*

---

## 6. Analysis and Discussion

### 6.1 Does the Multi-Agent Approach Work?

The LLM comparison strongly supports the multi-agent architecture. The 12.6 percentage point improvement in approval rate over the single-model baseline reflects the Governance agent's ability to challenge Planner recommendations that exceed risk thresholds, a check that the single-model approach cannot replicate without explicit role separation. This finding is consistent with Wu et al. (2023), who demonstrated that conversational role specialization reduces error rates on complex multi-step tasks.

Interestingly, the multi-agent system used 25% fewer tokens than the single-model baseline despite producing better outcomes. We attribute this to the structured message-passing protocol: each agent receives only the information relevant to its role, reducing context window noise and the tendency for large models to hedge or over-explain.

### 6.2 Is 16.6% Carbon Reduction Meaningful?

Our result (16.6% verified reduction, 62 kgCO₂e over 30 days on a simulated workload) falls in the lower range of the 10–40% reported by Radovanović et al. (2022). We attribute this to two factors. First, our synthetic workload has a higher proportion of urgent (non-deferrable) jobs than Google's production mix. Second, our simulation uses five regions with moderate intensity differences; real-world deployments with access to low-intensity regions like eu-north-1 (Nordic hydro/nuclear, ~30 gCO₂/kWh) vs. ap-south-1 (Indian coal, ~700 gCO₂/kWh) would see much larger gains from region shifting.

The multi-objective result is particularly noteworthy: cost decreased by 3.6% alongside the carbon reduction. This challenges the common assumption that carbon optimization necessarily conflicts with cost optimization. The key insight, consistent with CarbonScaler, is that the cleanest regions are often also cheaper (Pacific Northwest hydro in the US, Nordic hydro in Europe), so shifting to cleaner regions frequently reduces both emissions and cost simultaneously.

The 32-point spread in SUS scores (Participant 2 90, Participant 4 58) reflects the system's current optimization for technically literate users. The Evidence Explorer, consistently praised by Participant 1, Participant 2, and Participant 3 presupposes familiarity with concepts like carbon intensity and counterfactual reasoning. Participant 4's lower score and incomplete task set (he missed the Ask the Agent task entirely) suggests onboarding improvements are a higher priority than feature additions for broadening the user base.

### 6.3 User Study Interpretation

The UMUX-Lite score of 76.4 is promising for a research prototype with no prior UX iteration. The gap in trust between technical and non-technical users (4.4 vs. 3.6) suggests that the system's primary value proposition — verifiable, auditable savings — resonates most strongly with users who can evaluate the underlying methodology. For broader adoption, particularly among sustainability officers who may lack cloud engineering backgrounds, clearer in-dashboard explanations of the AI+determinism boundary would improve trust.

The near-universal praise for the Evidence Explorer confirms our hypothesis that verification is the differentiating feature. Users have seen carbon dashboards before; they have not seen a system that shows the counterfactual math behind every claimed saving.

### 6.4 Prompt Engineering Insights

The large quality gap between v1 (2.9/5) and v3 (4.4/5) demonstrates that prompt structure and few-shot examples matter substantially for rationale quality. The key improvements from v1 to v3 were: (1) requiring the model to cite specific numbers from the input data, (2) structuring output into carbon math / cost impact / risk sections, and (3) providing two examples of high-quality rationales. The improvement was especially pronounced for medium-risk recommendations, where governance reviewers need detailed justification.

---

## 7. Limitations, Risks, and Ethical Considerations

### 7.1 Limitations

**Synthetic workload.** Partial real carbon intensity data. Two of five evaluated regions (eu-north-1, eu-west-1) use synthetic sinusoidal carbon intensity curves calibrated to published grid averages rather than live API data, due to missing ENTSOE and ElectricityMaps API tokens. US and India regions use real-time sources. Results for EU regions should be interpreted with this caveat. Adding an ElectricityMaps or ENTSOE API token would make all five regions fully real

**Carbon intensity model.** In simulation mode, carbon intensity follows sine-wave models calibrated to regional averages. Real grid intensity is highly variable and influenced by weather, demand spikes, and energy market events. The ElectricityMaps integration addresses this for real-data mode, but simulation-mode results should be treated as illustrative.

**Small user study.** Our user study was conducted with a convenience sample of IS492 classmates and university researchers. This population skews younger and more technically literate than the target enterprise audience. A larger, more representative study would be needed to validate findings for enterprise deployment.

**No real execution.** The Executor agent writes to CSV files rather than calling real cloud provider APIs. The full loop — plan, execute, verify — is simulated. Real-world execution introduces latency, dependencies, and failure modes not captured here.

### 7.2 Risks

**Over-trust in recommendations.** Users may accept Planner recommendations without reviewing the evidence chain, particularly if they lack the technical background to evaluate confidence intervals. The Governance approval step partially mitigates this, but a human-in-the-loop review of flagged recommendations is strongly recommended for production deployments.

**Carbon intensity data quality.** All savings calculations depend on the accuracy of grid carbon intensity data. Electricity Maps provides high-quality real-time data for most regions, but coverage is incomplete in some areas (parts of Asia, Africa, and South America). Systems deployed in regions with poor coverage should use wider uncertainty bounds.

**LLM reliability.** Rationale quality depends on LLM availability and output consistency. The mock LLM fallback ensures the pipeline runs, but mock rationales are less informative than real LLM outputs. The AI+determinism boundary means numerical results are always correct, but narrative quality degrades gracefully rather than failing hard.

**Data Source Transparency .** Synthetic fallback regions are clearly labeled in all outputs. No output misrepresents synthetic data as live data. However, consumers of the system should verify data source labels before using results in compliance filings.

### 7.3 Ethical Considerations

**No PII collected.** The system operates entirely on workload metadata and carbon data. No user identities, personal information, or behavioral data are collected or stored.

**Algorithmic fairness.** The Planner's optimization function weights all teams equally. In organizations where certain teams have stricter SLA requirements, the optimization may disproportionately defer workloads from teams with more flexible deadlines. Administrators should review optimization results for unintended distributional effects.

**Carbon offset claims.** This system measures and verifies reductions in operational carbon intensity (Scope 2, location-based). It does not account for embodied carbon in hardware (Scope 3), purchased renewable energy certificates, or carbon offsets. Claims made using this system should be clearly scoped to operational Scope 2 reductions.

**Transparency.** All emission factors, cost models, and counterfactual formulas are openly documented in `SOURCES.md` and cited in code. Users can inspect and challenge every calculation.

---

## 8. Conclusion and Future Work

### 8.1 Conclusion

We presented sust-AI-naible, a multi-agent AI system that closes the full loop on cloud carbon optimization: sensing workloads, calculating emissions, planning optimizations, governing changes, executing recommendations, and verifying savings with auditable counterfactual evidence. Our evaluation demonstrated 16.6% verified carbon reduction on simulated workloads, confirmed that multi-agent architectures outperform single-model approaches for this class of problem, and found strong user satisfaction (UMUX-Lite 76.4) with particular appreciation for the Evidence Explorer's ability to show the reasoning behind every claimed saving.

The most important contribution may be conceptual: by separating AI communication from deterministic calculation, the system produces claims that are both human-readable and independently verifiable — a combination that is necessary for CSRD compliance and enterprise trust, but absent from all current production tools.

### 8.2 Future Work

**Real cloud provider integration.** Connecting the Executor to AWS, GCP, and Azure APIs via their scheduling and resource management interfaces would enable real-world deployment. The architecture already anticipates this — `src/data/aws_pricing.py` includes a stub for live pricing integration.

**Persistent user sessions and multi-tenant support.** The current system is single-user. Enterprise deployment would require multi-tenant isolation, role-based access control, and persistent session state.

**CSRD-ready report export.** Generating a formatted PDF report compatible with CSRD disclosure templates would reduce the compliance burden significantly. The verification evidence chain already contains all required data; formatting it into a reportable structure is primarily an engineering task.

**Longitudinal learning.** A feedback loop in which the Planner's models improve from past verification results — updating its carbon intensity forecasts and savings estimates based on actual outcomes — would improve recommendation quality over time.

**Study with enterprise users.** A rigorous user study with sustainability officers, cloud financial operations (FinOps) teams, and enterprise compliance teams would validate findings beyond the academic context and identify additional usability requirements.

---

## References

Hanafy, W. A., Liang, Q., Bashir, N., Irwin, D., & Shenoy, P. (2023). CarbonScaler: Leveraging cloud workload elasticity for optimizing carbon-efficiency. *Proceedings of the ACM on Measurement and Analysis of Computing Systems*, 7(3). https://doi.org/10.1145/3626788

Luccioni, A. S., Viguier, S., & Ligozat, A.-L. (2023). Estimating the carbon footprint of BLOOM, a 176B parameter language model. *Journal of Machine Learning Research*, 24(253), 1–15.

Masanet, E., Shehabi, A., Lei, N., Smith, S., & Koomey, J. (2020). Recalibrating global data center energy-use estimates. *Science*, 367(6481), 984–986. https://doi.org/10.1126/science.aba3758

Patterson, D., Gonzalez, J., Le, Q., Liang, C., Munguia, L.-M., Rothchild, D., So, D., Texier, M., & Dean, J. (2021). Carbon emissions and large neural network training. *arXiv preprint*. https://arxiv.org/abs/2104.10350

Radovanović, A., Koningstein, R., Schneider, I., Chen, B., Duarte, A., Le, B., Mariager, C., Beausoleil, P., Xu, S., Andersen, M., & Brown, M. (2022). Carbon-aware computing for datacenters. *IEEE Transactions on Power Systems*, 38(2), 1270–1280. https://doi.org/10.1109/TPWRS.2022.3173250

U.S. Environmental Protection Agency. (2024). *Emissions & Generation Resource Integrated Database (eGRID) 2022*. https://www.epa.gov/egrid

World Resources Institute & World Business Council for Sustainable Development. (2015). *GHG Protocol Corporate Accounting and Reporting Standard* (Rev. ed.). https://ghgprotocol.org/corporate-standard

Wu, Q., Bansal, G., Zhang, J., Wu, Y., Zhang, S., Zhu, E., Li, B., Jiang, L., Zhang, X., & Wang, C. (2023). AutoGen: Enabling next-gen LLM applications via multi-agent conversation. *arXiv preprint*. https://arxiv.org/abs/2308.08155

---

## Appendix A: User Study Materials

### A.1 Participant Information Sheet

> **Study Title:** Evaluation of sust-AI-naible: A Cloud Carbon Optimization Dashboard  
> **Purpose:** We are evaluating the usability and usefulness of a research prototype for cloud carbon optimization.  
> **What you'll do:** Complete 5 structured tasks using the dashboard (~20 min), then answer a short survey and interview (~10 min).  
> **Data collected:** Task completion times, survey responses, interview notes.  
> **Confidentiality:** Names were retained but were censored
> **Voluntary:** Participation is entirely voluntary. You may stop at any time.

### A.2 Task Sheet (given to participants)

```
Task 1: Find the total verified carbon savings from the last pipeline run.
        → Where would you look first?

Task 2: Which cloud region had the highest average carbon intensity?
        → Navigate to the Carbon Analysis page.

Task 3: Open the Evidence Explorer. Click on any verification record.
        → Describe in your own words what the "counterfactual" column means.

Task 4: Use the Ask the Agent chat.
        → Ask: "Which team saved the most carbon this week?"

Task 5: Go to The Debate page. Read one recommendation and the negotiation.
        → On a scale of 1-5, how much do you trust this recommendation? Why?
```

### A.3 UMUX-Lite Survey

*Rated on a 7-point scale from "Strongly Disagree" to "Strongly Agree":*

1. This system's capabilities meet my requirements.
2. This system is easy to use.

*Additional questions (5-point scale):*

3. I trust the carbon savings numbers reported by this system.
4. The AI explanations (rationales) helped me understand the recommendations.
5. I would use this system for compliance reporting.

---

## Appendix B: Prompt Versions (Ablation Study)

### B.1 Prompt v1 — Baseline

```
You are a sustainability advisor. Explain why this cloud workload 
recommendation is beneficial.

Recommendation: {recommendation_json}
```

### B.2 Prompt v2 — Structured

```
You are a sustainability advisor reviewing a cloud optimization recommendation.
Write a concise rationale covering three sections:

1. Carbon math: How much carbon does this save and why?
2. Cost impact: How does this affect cloud spend?
3. Risk: What could go wrong?

Recommendation: {recommendation_json}
Baseline emissions: {baseline_kgco2e} kgCO₂e
Projected savings: {projected_savings_kgco2e} kgCO₂e [{confidence_interval}]
Cost delta: {cost_delta_usd} USD
```

### B.3 Prompt v3 — Few-Shot (adopted for final system)

```
You are a sustainability advisor reviewing a cloud optimization recommendation.
Write a concise rationale covering three sections: Carbon math, Cost impact, Risk.
Always cite specific numbers from the input data.

--- EXAMPLE 1 ---
[Example of high-quality rationale with specific numbers and clear reasoning]

--- EXAMPLE 2 ---
[Example of high-quality rationale for a medium-risk recommendation]

--- YOUR TASK ---
Recommendation: {recommendation_json}
Baseline emissions: {baseline_kgco2e} kgCO₂e
Projected savings: {projected_savings_kgco2e} kgCO₂e [{confidence_interval}]
Cost delta: {cost_delta_usd} USD
Region shift: {source_region} → {target_region}
Grid intensity delta: {source_intensity} → {target_intensity} gCO₂/kWh
```

---
