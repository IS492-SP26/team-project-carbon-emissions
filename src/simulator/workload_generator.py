"""
Workload Generator
==================
Generates synthetic cloud workloads that mimic a ~100 developer organization.

Workload mix (Assumption):
  - CI/CD builds:       ~800/day  (60% balanced, 30% urgent, 10% sustainable)
  - Batch analytics:     ~50/day  (20% balanced, 80% sustainable)
  - Model training:      ~10/day  (100% sustainable)
  - Dev/test envs:      ~100/day  (100% sustainable)
  - Production services: ~30/day  (100% urgent — represented as daily slices)

Arrival patterns:
  - CI/CD peaks during work hours (9am-6pm in job's home timezone)
  - Batch analytics: mostly overnight
  - Model training: submitted anytime, runs long
  - Dev/test: work hours only
  - Production: 24/7

All distributions are Assumptions — designed to be plausible, not precise.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional

from src.shared.models import Job, WorkloadCategory, REGIONS


# ── Team definitions ──────────────────────────────────────────────────
# 100 developers across 10 teams
TEAMS = [
    {"team_id": "platform",     "size": 15, "home_region": "us-east-1"},
    {"team_id": "frontend",     "size": 12, "home_region": "us-east-1"},
    {"team_id": "backend-api",  "size": 12, "home_region": "us-east-1"},
    {"team_id": "data-eng",     "size": 10, "home_region": "us-west-2"},
    {"team_id": "ml-team",      "size": 8,  "home_region": "us-west-2"},
    {"team_id": "mobile",       "size": 10, "home_region": "us-east-1"},
    {"team_id": "devops",       "size": 8,  "home_region": "us-east-1"},
    {"team_id": "eu-backend",   "size": 10, "home_region": "eu-west-1"},
    {"team_id": "eu-frontend",  "size": 8,  "home_region": "eu-west-1"},
    {"team_id": "analytics",    "size": 7,  "home_region": "us-west-2"},
]

# ── Workload templates ────────────────────────────────────────────────
WORKLOAD_TEMPLATES = {
    "ci_cd": {
        "daily_count_mean": 800,
        "vcpu_choices": [2, 4, 8],
        "vcpu_weights": [0.3, 0.5, 0.2],
        "gpu_count": 0,
        "duration_mean_hrs": 0.2,        # 12 minutes
        "duration_std_hrs": 0.15,
        "duration_min_hrs": 0.03,         # 2 minutes minimum
        "duration_max_hrs": 1.5,          # 90 minutes max
        "category_split": {
            WorkloadCategory.URGENT: 0.30,
            WorkloadCategory.BALANCED: 0.60,
            WorkloadCategory.SUSTAINABLE: 0.10,
        },
        "peak_hours_utc": list(range(13, 23)),  # 9am-6pm ET roughly
        "peak_multiplier": 3.0,
    },
    "batch_analytics": {
        "daily_count_mean": 50,
        "vcpu_choices": [8, 16, 32],
        "vcpu_weights": [0.3, 0.5, 0.2],
        "gpu_count": 0,
        "duration_mean_hrs": 2.0,
        "duration_std_hrs": 1.0,
        "duration_min_hrs": 0.5,
        "duration_max_hrs": 8.0,
        "category_split": {
            WorkloadCategory.BALANCED: 0.20,
            WorkloadCategory.SUSTAINABLE: 0.80,
        },
        "peak_hours_utc": list(range(2, 8)),  # Overnight US
        "peak_multiplier": 2.0,
    },
    "model_training": {
        "daily_count_mean": 10,
        "vcpu_choices": [8, 16],
        "vcpu_weights": [0.6, 0.4],
        "gpu_count": 1,
        "duration_mean_hrs": 6.0,
        "duration_std_hrs": 3.0,
        "duration_min_hrs": 1.0,
        "duration_max_hrs": 24.0,
        "category_split": {
            WorkloadCategory.SUSTAINABLE: 1.0,
        },
        "peak_hours_utc": list(range(14, 22)),  # Submitted during work hours
        "peak_multiplier": 2.0,
    },
    "dev_test": {
        "daily_count_mean": 100,
        "vcpu_choices": [1, 2, 4],
        "vcpu_weights": [0.3, 0.5, 0.2],
        "gpu_count": 0,
        "duration_mean_hrs": 4.0,
        "duration_std_hrs": 3.0,
        "duration_min_hrs": 0.5,
        "duration_max_hrs": 10.0,
        "category_split": {
            WorkloadCategory.SUSTAINABLE: 1.0,
        },
        "peak_hours_utc": list(range(13, 23)),
        "peak_multiplier": 4.0,
    },
    "production": {
        "daily_count_mean": 30,
        "vcpu_choices": [2, 4, 8],
        "vcpu_weights": [0.3, 0.5, 0.2],
        "gpu_count": 0,
        "duration_mean_hrs": 24.0,  # Always-on (represented as daily slices)
        "duration_std_hrs": 0.0,
        "duration_min_hrs": 24.0,
        "duration_max_hrs": 24.0,
        "category_split": {
            WorkloadCategory.URGENT: 1.0,
        },
        "peak_hours_utc": list(range(0, 24)),  # 24/7
        "peak_multiplier": 1.0,  # Flat
    },
}


def _hour_weight(hour: int, peak_hours: list, peak_multiplier: float) -> float:
    """Weight for job arrival at a given hour (higher during peak)."""
    if hour in peak_hours:
        return peak_multiplier
    return 1.0


def generate_workloads(
    start_date: datetime,
    num_days: int = 30,
    seed: Optional[int] = 42,
) -> list[Job]:
    """
    Generate a synthetic workload set for the organization.
    
    Returns: list of Job objects
    """
    rng = np.random.default_rng(seed)
    jobs = []

    for day_offset in range(num_days):
        current_date = start_date + timedelta(days=day_offset)
        is_weekend = current_date.weekday() >= 5  # Sat=5, Sun=6

        for wtype, template in WORKLOAD_TEMPLATES.items():
            # Weekend: 30% of weekday volume (except production)
            base_count = template["daily_count_mean"]
            if is_weekend and wtype != "production":
                base_count = int(base_count * 0.3)

            # Add some daily variance (±20%)
            daily_count = max(1, int(rng.normal(base_count, base_count * 0.1)))

            # Compute hourly weights for arrival distribution
            hour_weights = np.array([
                _hour_weight(h, template["peak_hours_utc"], template["peak_multiplier"])
                for h in range(24)
            ])
            hour_probs = hour_weights / hour_weights.sum()

            for _ in range(daily_count):
                # Pick arrival hour
                hour = rng.choice(24, p=hour_probs)
                minute = rng.integers(0, 60)
                arrival = current_date.replace(hour=int(hour), minute=int(minute), second=0)

                # Pick team (weighted by team size)
                team_weights = np.array([t["size"] for t in TEAMS], dtype=float)
                # ML team gets most training jobs, data-eng gets most analytics
                if wtype == "model_training":
                    team_weights[4] *= 5  # ml-team
                elif wtype == "batch_analytics":
                    team_weights[3] *= 3  # data-eng
                    team_weights[9] *= 3  # analytics
                team_probs = team_weights / team_weights.sum()
                team = TEAMS[rng.choice(len(TEAMS), p=team_probs)]

                # Pick resources
                vcpus = rng.choice(template["vcpu_choices"], p=template["vcpu_weights"])
                gpu_count = template["gpu_count"]

                # Pick duration (log-normal-ish, clamped)
                if template["duration_std_hrs"] > 0:
                    duration = rng.normal(template["duration_mean_hrs"], template["duration_std_hrs"])
                    duration = np.clip(duration, template["duration_min_hrs"], template["duration_max_hrs"])
                else:
                    duration = template["duration_mean_hrs"]

                # Pick category
                cat_choices = list(template["category_split"].keys())
                cat_probs = list(template["category_split"].values())
                category = rng.choice(cat_choices, p=cat_probs)

                # Region: typically team's home region
                region = team["home_region"]

                job = Job(
                    name=f"{wtype}-{day_offset:03d}-{rng.integers(10000):04d}",
                    team_id=team["team_id"],
                    service_name=wtype,
                    region=region,
                    vcpus=int(vcpus),
                    gpu_count=gpu_count,
                    duration_hours=round(float(duration), 3),
                    category=category,
                    started_at=arrival,
                    ended_at=arrival + timedelta(hours=float(duration)),
                    workload_type=wtype,
                )

                jobs.append(job)

    return jobs


def jobs_to_dataframe(jobs: list[Job]) -> pd.DataFrame:
    """Convert a list of Job objects to a pandas DataFrame."""
    rows = []
    for j in jobs:
        rows.append({
            "job_id": j.job_id,
            "name": j.name,
            "team_id": j.team_id,
            "service_name": j.service_name,
            "region": j.region,
            "vcpus": j.vcpus,
            "gpu_count": j.gpu_count,
            "duration_hours": j.duration_hours,
            "category": j.category.value,
            "started_at": j.started_at,
            "ended_at": j.ended_at,
            "workload_type": j.workload_type,
        })
    return pd.DataFrame(rows)


# ── Quick self-test ───────────────────────────────────────────────────
if __name__ == "__main__":
    print("Generating 30 days of workloads for a 100-dev org...")
    jobs = generate_workloads(datetime(2025, 1, 1), num_days=30)
    df = jobs_to_dataframe(jobs)
    
    print(f"\nTotal jobs generated: {len(df):,}")
    print(f"\nBreakdown by workload type:")
    print(df["workload_type"].value_counts().to_string())
    print(f"\nBreakdown by category:")
    print(df["category"].value_counts().to_string())
    print(f"\nBreakdown by region:")
    print(df["region"].value_counts().to_string())
    print(f"\nBreakdown by team:")
    print(df["team_id"].value_counts().to_string())
    print(f"\nAvg duration by type (hours):")
    print(df.groupby("workload_type")["duration_hours"].mean().round(2).to_string())
