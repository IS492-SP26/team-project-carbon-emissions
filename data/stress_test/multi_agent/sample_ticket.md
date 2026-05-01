### SUST-7787: Carbon Optimization Change - Time Shift in eu-west-1
#### Change Description
We are proposing a time shift change in the eu-west-1 region to optimize carbon emissions. The change involves adjusting the timing of our workload to align with periods of lower carbon intensity in the region.

#### Expected Impact
* **Carbon Impact:** The expected carbon delta is `-0.0 gCO₂e`, indicating a neutral impact on our carbon footprint.
* **Cost Impact:** The expected cost delta is `$+0.0000`, resulting in no additional costs.

#### Risk Assessment and Rollback Plan
* **Risk Level:** Low
* **Rollback Plan:** In the event of any issues, we can quickly revert to the previous configuration. The rollback plan is outlined in the [PR](https://github.com/org/infra/pull/787).

#### Verification
Verification of the change will be performed by monitoring our carbon emissions and costs in the eu-west-1 region after the change is implemented. The results will be compared to our baseline measurements to ensure the expected outcomes are achieved. Confidence in the change is currently at `72%`.