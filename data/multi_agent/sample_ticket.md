### SUST-9317: Carbon Optimization Change
#### Summary
This ticket proposes a time shift optimization change in the eu-west-1 region to reduce carbon footprint.

#### Change Details
* **Action Type:** Time shift
* **Current Region:** eu-west-1
* **Proposed Region:** eu-west-1 (no region change)
* **Carbon Impact:** Expected reduction of 0.6 gCO₂e
* **Cost Impact:** No expected cost change ($0.0000 delta)

#### Risk Assessment
* **Risk Level:** Low
* **Confidence:** 80%
* **Rollback Plan:** In case of issues, we will revert to the original configuration. The rollback process will be triggered if key performance indicators (KPIs) deviate from expected thresholds.

#### Verification
Verification of the change's effectiveness will be done by:
* Monitoring carbon footprint metrics
* Reviewing cost reports
* Validating that KPIs remain within expected ranges
The change is proposed in this [PR](https://github.com/org/infra/pull/317).