"""
Carbon Accountant Agent
=======================
Computes location-based carbon emissions for every job, with uncertainty bounds.

This is STRICTLY DETERMINISTIC — no LLM in the calculation path.
Every output is reproducible given the same inputs.

Formula:
  kgCO₂e = vCPUs × duration_hrs × TDP_per_vCPU × PUE × grid_intensity_kgCO₂_per_kWh
         + gpu_count × duration_hrs × GPU_TDP × PUE × grid_intensity_kgCO₂_per_kWh

Assumptions:
  - TDP per vCPU: 0.005 kW (Assumption — ~200W server / 40 vCPUs)
  - GPU TDP: 0.300 kW (Assumption — NVIDIA A100 = 300W)
  - PUE: 1.1 (Assumption — hyperscaler average per Google/AWS sustainability reports)
  - Uncertainty: propagated from grid intensity bounds (±20%)

What this agent is NOT allowed to do:
  - Modify any activity records
  - Use unversioned emission factors
  - Suppress uncertainty bounds
  - Round results to make them look cleaner
"""

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Optional

from src.shared.models import Job, EmissionsRecord
from src.simulator.carbon_intensity import get_intensity_at


# ── Physical constants (Assumptions) ──────────────────────────────────
TDP_PER_VCPU_KW = 0.005    # kW per vCPU (Assumption)
GPU_TDP_KW = 0.300          # kW per GPU (Assumption — A100 class)
PUE = 1.1                   # Power Usage Effectiveness (Assumption — hyperscaler avg)
METHODOLOGY = "location_based_v1"


def compute_emissions_batch(
    jobs: list[Job],
    intensity_df: pd.DataFrame,
    verbose: bool = False,
) -> list[EmissionsRecord]:
    """
    Compute emissions for a batch of jobs using vectorized operations.
    Much faster than per-job loop for large batches.
    
    Returns: list of EmissionsRecord
    """
    if verbose:
        print(f"  Computing emissions for {len(jobs):,} jobs (vectorized)...")

    # Build a DataFrame from jobs for vectorized operations
    job_data = pd.DataFrame([{
        "job_id": j.job_id,
        "region": j.region,
        "vcpus": j.vcpus,
        "gpu_count": j.gpu_count,
        "duration_hours": j.duration_hours,
        "started_at": j.started_at,
    } for j in jobs])

    # Round timestamps to nearest hour for intensity lookup
    job_data["ts_hour"] = pd.to_datetime(job_data["started_at"]).dt.floor("h")

    # Merge with intensity data
    intensity_df_copy = intensity_df.copy()
    intensity_df_copy["timestamp"] = pd.to_datetime(intensity_df_copy["timestamp"])

    merged = job_data.merge(
        intensity_df_copy[["timestamp", "region", "intensity_gco2_kwh", "intensity_lower", "intensity_upper"]],
        left_on=["ts_hour", "region"],
        right_on=["timestamp", "region"],
        how="left",
    )

    # Fill NaN with region averages (fallback for missing timestamps)
    from src.simulator.carbon_intensity import REGION_PROFILES
    for region, profile in REGION_PROFILES.items():
        mask = (merged["region"] == region) & merged["intensity_gco2_kwh"].isna()
        merged.loc[mask, "intensity_gco2_kwh"] = profile["base_intensity"]
        merged.loc[mask, "intensity_lower"] = profile["base_intensity"] * 0.8
        merged.loc[mask, "intensity_upper"] = profile["base_intensity"] * 1.2

    # Vectorized emissions calculation
    merged["power_kw"] = (merged["vcpus"] * TDP_PER_VCPU_KW + merged["gpu_count"] * GPU_TDP_KW) * PUE
    merged["energy_kwh"] = merged["power_kw"] * merged["duration_hours"]
    merged["kgco2e"] = merged["energy_kwh"] * (merged["intensity_gco2_kwh"] / 1000)
    merged["kgco2e_lower"] = merged["energy_kwh"] * (merged["intensity_lower"] / 1000)
    merged["kgco2e_upper"] = merged["energy_kwh"] * (merged["intensity_upper"] / 1000)

    # Build EmissionsRecord list
    now = datetime.now()
    records = []
    for _, row in merged.iterrows():
        records.append(EmissionsRecord(
            job_id=row["job_id"],
            kgco2e=round(row["kgco2e"], 6),
            kgco2e_lower=round(row["kgco2e_lower"], 6),
            kgco2e_upper=round(row["kgco2e_upper"], 6),
            grid_intensity_used=row["intensity_gco2_kwh"] if pd.notna(row["intensity_gco2_kwh"]) else 0,
            methodology=METHODOLOGY,
            computed_at=now,
        ))

    if verbose:
        total = sum(r.kgco2e for r in records)
        print(f"  ✓ Total emissions: {total:.2f} kgCO₂e")

    return records


def compute_emissions_single(
    job: Job,
    intensity_df: pd.DataFrame,
) -> EmissionsRecord:
    """Compute carbon emissions for a single job."""
    intensity_data = get_intensity_at(intensity_df, job.region, job.started_at)

    intensity_kg = intensity_data["intensity"] / 1000
    intensity_lower_kg = intensity_data["lower"] / 1000
    intensity_upper_kg = intensity_data["upper"] / 1000

    cpu_power = job.vcpus * TDP_PER_VCPU_KW
    gpu_power = job.gpu_count * GPU_TDP_KW
    total_power = (cpu_power + gpu_power) * PUE
    energy_kwh = total_power * job.duration_hours

    kgco2e = energy_kwh * intensity_kg
    kgco2e_lower = energy_kwh * intensity_lower_kg
    kgco2e_upper = energy_kwh * intensity_upper_kg

    return EmissionsRecord(
        job_id=job.job_id,
        kgco2e=round(kgco2e, 6),
        kgco2e_lower=round(kgco2e_lower, 6),
        kgco2e_upper=round(kgco2e_upper, 6),
        grid_intensity_used=intensity_data["intensity"],
        methodology=METHODOLOGY,
        computed_at=datetime.now(),
    )


def emissions_to_dataframe(records: list[EmissionsRecord]) -> pd.DataFrame:
    """Convert EmissionsRecord list to DataFrame."""
    rows = []
    for r in records:
        rows.append({
            "activity_id": r.activity_id,
            "job_id": r.job_id,
            "kgco2e": r.kgco2e,
            "kgco2e_lower": r.kgco2e_lower,
            "kgco2e_upper": r.kgco2e_upper,
            "grid_intensity_gco2_kwh": r.grid_intensity_used,
            "methodology": r.methodology,
            "computed_at": r.computed_at,
        })
    return pd.DataFrame(rows)


def compute_emissions_for_config(
    vcpus: int,
    gpu_count: int,
    duration_hours: float,
    region: str,
    timestamp: datetime,
    intensity_df: pd.DataFrame,
) -> dict:
    """
    Compute emissions for a hypothetical config (used by Planner for what-if analysis).
    
    Returns:
        {"kgco2e": float, "kgco2e_lower": float, "kgco2e_upper": float}
    """
    intensity_data = get_intensity_at(intensity_df, region, timestamp)
    intensity_kg = intensity_data["intensity"] / 1000
    intensity_lower_kg = intensity_data["lower"] / 1000
    intensity_upper_kg = intensity_data["upper"] / 1000

    total_power = (vcpus * TDP_PER_VCPU_KW + gpu_count * GPU_TDP_KW) * PUE
    energy_kwh = total_power * duration_hours

    return {
        "kgco2e": round(energy_kwh * intensity_kg, 6),
        "kgco2e_lower": round(energy_kwh * intensity_lower_kg, 6),
        "kgco2e_upper": round(energy_kwh * intensity_upper_kg, 6),
        "grid_intensity": intensity_data["intensity"],
    }
