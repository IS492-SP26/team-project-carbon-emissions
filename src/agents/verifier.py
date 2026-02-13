"""
Verifier Agent
==============
THE CORE DIFFERENTIATOR: Measures actual outcomes of executed recommendations
against counterfactual baselines with confidence intervals and evidence chains.

Approach (MVP): Difference-based counterfactual
  For a job moved from region A → region B:
    counterfactual = actual_energy_kwh × emission_factor(region_A, actual_time)
    actual         = actual_energy_kwh × emission_factor(region_B, actual_time)
    savings        = counterfactual - actual

  This answers: "What WOULD the emissions have been if we hadn't moved the job?"

Uncertainty propagation:
  - Grid intensity has ±20% bounds → propagate through the formula
  - 90% confidence intervals computed via interval arithmetic

Evidence chain:
  Every verification record includes a machine-readable trace of:
    - What data was used (activity record, emission factors, timestamps)
    - What formula was applied
    - What the intermediate values were
    - Hash of inputs for tamper detection

What this agent is NOT allowed to do:
  - Verify its own recommendations (separation of concerns)
  - Cherry-pick time windows to inflate savings
  - Report savings without confidence intervals
  - Retroactively modify past verification records

This is STRICTLY DETERMINISTIC for all calculations.
"""

import hashlib
import json
import pandas as pd
from datetime import datetime
from typing import Optional

from src.shared.models import Job, Recommendation, VerificationRecord
from src.agents.carbon_accountant import (
    compute_emissions_for_config,
    TDP_PER_VCPU_KW, GPU_TDP_KW, PUE,
)
from src.simulator.carbon_intensity import get_intensity_at


def _hash_inputs(**kwargs) -> str:
    """Create a SHA-256 hash of inputs for tamper detection."""
    serializable = {}
    for k, v in kwargs.items():
        if isinstance(v, datetime):
            serializable[k] = v.isoformat()
        else:
            serializable[k] = v
    payload = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def verify_single(
    rec: Recommendation,
    original_job: Job,
    executed_job: Job,
    intensity_df: pd.DataFrame,
) -> VerificationRecord:
    """
    Verify a single executed recommendation using counterfactual analysis.
    
    The key question: "What would emissions have been WITHOUT our intervention?"
    
    Counterfactual: same energy consumption, but at the ORIGINAL region/time
    Actual: same energy consumption, at the NEW region/time
    """
    # ── Step 1: Compute actual emissions (what DID happen) ────────────
    actual_intensity = get_intensity_at(
        intensity_df, executed_job.region, executed_job.started_at
    )
    
    # Use the EXECUTED job's actual resource usage
    cpu_power = executed_job.vcpus * TDP_PER_VCPU_KW
    gpu_power = executed_job.gpu_count * GPU_TDP_KW
    total_power_kw = (cpu_power + gpu_power) * PUE
    energy_kwh = total_power_kw * executed_job.duration_hours

    actual_kgco2e = energy_kwh * (actual_intensity["intensity"] / 1000)
    actual_lower = energy_kwh * (actual_intensity["lower"] / 1000)
    actual_upper = energy_kwh * (actual_intensity["upper"] / 1000)

    # ── Step 2: Compute counterfactual (what WOULD have happened) ─────
    # Key: use the SAME energy consumption, but the ORIGINAL region/time
    counterfactual_intensity = get_intensity_at(
        intensity_df, original_job.region, original_job.started_at
    )

    counterfactual_kgco2e = energy_kwh * (counterfactual_intensity["intensity"] / 1000)
    counterfactual_lower = energy_kwh * (counterfactual_intensity["lower"] / 1000)
    counterfactual_upper = energy_kwh * (counterfactual_intensity["upper"] / 1000)

    # ── Step 3: Compute verified savings ──────────────────────────────
    verified_savings = counterfactual_kgco2e - actual_kgco2e

    # 90% CI: worst case for counterfactual (lower) - best case for actual (upper)
    # through: best case for counterfactual (upper) - worst case for actual (lower)
    ci_lower = counterfactual_lower - actual_upper   # Conservative estimate
    ci_upper = counterfactual_upper - actual_lower    # Optimistic estimate

    # ── Step 4: Determine verification status ─────────────────────────
    if ci_lower > 0:
        # Even the conservative estimate shows savings
        verification_status = "confirmed"
    elif verified_savings > 0 and ci_lower <= 0:
        # Point estimate positive but CI includes zero
        verification_status = "partial"
    elif verified_savings <= 0:
        # No savings detected
        verification_status = "refuted"
    else:
        verification_status = "inconclusive"

    # ── Step 5: Check SLA compliance ──────────────────────────────────
    # In simulation: check if the job still completed within category constraints
    sla_compliant = True
    if rec.proposed_time and original_job.started_at:
        delay_hours = (rec.proposed_time - original_job.started_at).total_seconds() / 3600
        from src.shared.models import WorkloadCategory
        max_delay = {
            WorkloadCategory.URGENT: 0,
            WorkloadCategory.BALANCED: 4,
            WorkloadCategory.SUSTAINABLE: 24,
        }.get(original_job.category, 0)
        if delay_hours > max_delay:
            sla_compliant = False

    # ── Step 6: Build evidence chain ──────────────────────────────────
    input_hash = _hash_inputs(
        job_id=original_job.job_id,
        original_region=original_job.region,
        original_time=original_job.started_at,
        executed_region=executed_job.region,
        executed_time=executed_job.started_at,
        vcpus=executed_job.vcpus,
        gpu_count=executed_job.gpu_count,
        duration_hours=executed_job.duration_hours,
    )

    evidence_chain = [
        {
            "step": "input_activity",
            "description": "Original job configuration",
            "data": {
                "job_id": original_job.job_id,
                "region": original_job.region,
                "started_at": str(original_job.started_at),
                "vcpus": original_job.vcpus,
                "gpu_count": original_job.gpu_count,
                "duration_hours": original_job.duration_hours,
            },
        },
        {
            "step": "executed_config",
            "description": "Executed (optimized) job configuration",
            "data": {
                "region": executed_job.region,
                "started_at": str(executed_job.started_at),
            },
        },
        {
            "step": "energy_computation",
            "description": f"Energy = ({executed_job.vcpus} × {TDP_PER_VCPU_KW} + "
                          f"{executed_job.gpu_count} × {GPU_TDP_KW}) × {PUE} × "
                          f"{executed_job.duration_hours}h = {energy_kwh:.6f} kWh",
            "data": {
                "total_power_kw": round(total_power_kw, 6),
                "energy_kwh": round(energy_kwh, 6),
            },
        },
        {
            "step": "actual_emissions",
            "description": f"Actual: {energy_kwh:.6f} kWh × "
                          f"{actual_intensity['intensity']:.1f} gCO₂/kWh = "
                          f"{actual_kgco2e*1000:.2f} gCO₂e",
            "data": {
                "region": executed_job.region,
                "timestamp": str(executed_job.started_at),
                "grid_intensity_gco2_kwh": actual_intensity["intensity"],
                "source": actual_intensity["source"],
                "kgco2e": round(actual_kgco2e, 6),
            },
        },
        {
            "step": "counterfactual_emissions",
            "description": f"Counterfactual: {energy_kwh:.6f} kWh × "
                          f"{counterfactual_intensity['intensity']:.1f} gCO₂/kWh = "
                          f"{counterfactual_kgco2e*1000:.2f} gCO₂e",
            "data": {
                "region": original_job.region,
                "timestamp": str(original_job.started_at),
                "grid_intensity_gco2_kwh": counterfactual_intensity["intensity"],
                "source": counterfactual_intensity["source"],
                "kgco2e": round(counterfactual_kgco2e, 6),
            },
        },
        {
            "step": "savings_computation",
            "description": f"Savings: {counterfactual_kgco2e*1000:.2f} - "
                          f"{actual_kgco2e*1000:.2f} = {verified_savings*1000:.2f} gCO₂e",
            "data": {
                "verified_savings_kgco2e": round(verified_savings, 6),
                "ci_90_lower": round(ci_lower, 6),
                "ci_90_upper": round(ci_upper, 6),
                "verification_status": verification_status,
            },
        },
        {
            "step": "integrity",
            "description": "Input hash for tamper detection",
            "data": {
                "input_hash_sha256": input_hash,
                "methodology": "counterfactual_diff_v1",
                "uncertainty_method": "interval_arithmetic",
            },
        },
    ]

    return VerificationRecord(
        recommendation_id=rec.recommendation_id,
        counterfactual_kgco2e=round(counterfactual_kgco2e, 6),
        actual_kgco2e=round(actual_kgco2e, 6),
        verified_savings_kgco2e=round(verified_savings, 6),
        ci_lower=round(ci_lower, 6),
        ci_upper=round(ci_upper, 6),
        sla_compliant=sla_compliant,
        verification_status=verification_status,
        evidence_chain=evidence_chain,
        verified_at=datetime.now(),
    )


def verify_batch(
    recommendations: list[Recommendation],
    original_jobs: list[Job],
    executed_jobs: list[Job],
    intensity_df: pd.DataFrame,
    verbose: bool = False,
) -> list[VerificationRecord]:
    """
    Verify a batch of executed recommendations.
    
    Returns: list of VerificationRecord
    """
    original_map = {j.job_id: j for j in original_jobs}
    executed_map = {j.job_id: j for j in executed_jobs}

    records = []
    for i, rec in enumerate(recommendations):
        if rec.status != "executed":
            continue

        original = original_map.get(rec.job_id)
        executed = executed_map.get(rec.job_id)
        if original is None or executed is None:
            continue

        verification = verify_single(rec, original, executed, intensity_df)
        rec.status = "verified"
        records.append(verification)

        if verbose and (i + 1) % 500 == 0:
            print(f"  Verified {i + 1:,} / {len(recommendations):,} recommendations...")

    if verbose:
        confirmed = sum(1 for v in records if v.verification_status == "confirmed")
        partial = sum(1 for v in records if v.verification_status == "partial")
        refuted = sum(1 for v in records if v.verification_status == "refuted")
        print(f"  Verification complete: {len(records)} verified "
              f"({confirmed} confirmed, {partial} partial, {refuted} refuted)")

    return records


def verifications_to_dataframe(records: list[VerificationRecord]) -> pd.DataFrame:
    """Convert VerificationRecord list to DataFrame."""
    rows = []
    for v in records:
        rows.append({
            "verification_id": v.verification_id,
            "recommendation_id": v.recommendation_id,
            "counterfactual_kgco2e": v.counterfactual_kgco2e,
            "actual_kgco2e": v.actual_kgco2e,
            "verified_savings_kgco2e": v.verified_savings_kgco2e,
            "ci_lower": v.ci_lower,
            "ci_upper": v.ci_upper,
            "sla_compliant": v.sla_compliant,
            "verification_status": v.verification_status,
            "evidence_chain_steps": len(v.evidence_chain),
            "verified_at": v.verified_at,
        })
    return pd.DataFrame(rows)


def summarize_verification(records: list[VerificationRecord]) -> dict:
    """Produce a summary of verification results."""
    if not records:
        return {"count": 0}

    total_savings = sum(v.verified_savings_kgco2e for v in records)
    total_counterfactual = sum(v.counterfactual_kgco2e for v in records)
    total_actual = sum(v.actual_kgco2e for v in records)

    # Aggregate confidence intervals (sum of intervals)
    total_ci_lower = sum(v.ci_lower for v in records)
    total_ci_upper = sum(v.ci_upper for v in records)

    status_counts = {}
    for v in records:
        status_counts[v.verification_status] = status_counts.get(v.verification_status, 0) + 1

    sla_violations = sum(1 for v in records if not v.sla_compliant)

    # Calibration check: what % of individual CIs contain their point estimate?
    # (Self-consistency check — in real system, compare against future actuals)
    well_calibrated = sum(
        1 for v in records
        if v.ci_lower <= v.verified_savings_kgco2e <= v.ci_upper
    )

    return {
        "count": len(records),
        "total_verified_savings_kgco2e": round(total_savings, 4),
        "total_counterfactual_kgco2e": round(total_counterfactual, 4),
        "total_actual_kgco2e": round(total_actual, 4),
        "aggregate_ci_90": {
            "lower": round(total_ci_lower, 4),
            "upper": round(total_ci_upper, 4),
        },
        "by_status": status_counts,
        "sla_violations": sla_violations,
        "calibration_self_consistency": round(well_calibrated / len(records) * 100, 1),
    }


def format_evidence_chain(verification: VerificationRecord) -> str:
    """Format an evidence chain as human-readable text (for demo/dashboard)."""
    lines = [
        f"=== Evidence Chain for Recommendation {verification.recommendation_id} ===",
        f"Verification Status: {verification.verification_status.upper()}",
        f"Verified Savings: {verification.verified_savings_kgco2e * 1000:.2f} gCO₂e "
        f"[90% CI: {verification.ci_lower * 1000:.2f} – {verification.ci_upper * 1000:.2f}]",
        f"SLA Compliant: {'Yes' if verification.sla_compliant else 'NO — VIOLATION'}",
        "",
    ]

    for i, step in enumerate(verification.evidence_chain, 1):
        lines.append(f"Step {i}: {step['step']}")
        lines.append(f"  {step['description']}")
        if 'data' in step:
            for k, v in step['data'].items():
                lines.append(f"    {k}: {v}")
        lines.append("")

    return "\n".join(lines)
