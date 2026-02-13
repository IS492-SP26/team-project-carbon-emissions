"""
Run Baseline Analysis
=====================
This is your "hello world" — generates workloads, computes emissions, shows results.
Run this to verify everything works and see your first numbers.

Usage: python run_baseline.py
"""

from datetime import datetime
import pandas as pd

from src.simulator.workload_generator import generate_workloads, jobs_to_dataframe
from src.simulator.carbon_intensity import generate_intensity_timeseries
from src.simulator.cost_model import compute_job_cost
from src.agents.carbon_accountant import compute_emissions_batch, emissions_to_dataframe


def main():
    print("=" * 70)
    print("  sust-AI-naible — Baseline Analysis")
    print("=" * 70)

    # ── Step 1: Generate synthetic workloads ──────────────────────────
    print("\n[1/4] Generating 30 days of synthetic workloads...")
    start_date = datetime(2025, 1, 1)
    jobs = generate_workloads(start_date, num_days=30, seed=42)
    jobs_df = jobs_to_dataframe(jobs)
    print(f"  Generated {len(jobs):,} jobs")

    # ── Step 2: Generate carbon intensity time series ─────────────────
    print("\n[2/4] Generating carbon intensity time series...")
    intensity_df = generate_intensity_timeseries(start_date, num_days=30, seed=42)
    print(f"  Generated {len(intensity_df):,} intensity data points")

    # ── Step 3: Compute costs ─────────────────────────────────────────
    print("\n[3/4] Computing cloud costs...")
    jobs_df["cost_usd"] = jobs_df.apply(
        lambda row: compute_job_cost(row["region"], row["vcpus"], row["gpu_count"], row["duration_hours"]),
        axis=1,
    )

    # ── Step 4: Compute emissions ─────────────────────────────────────
    print("\n[4/4] Computing carbon emissions (this may take a moment)...")
    emissions = compute_emissions_batch(jobs, intensity_df, verbose=True)
    emissions_df = emissions_to_dataframe(emissions)

    # Merge for analysis
    merged = jobs_df.copy()
    merged["kgco2e"] = emissions_df["kgco2e"].values
    merged["kgco2e_lower"] = emissions_df["kgco2e_lower"].values
    merged["kgco2e_upper"] = emissions_df["kgco2e_upper"].values

    # ── Results ───────────────────────────────────────────────────────
    carbon_price_per_ton = 75  # $/ton (Given)
    total_cost = merged["cost_usd"].sum()
    total_kgco2e = merged["kgco2e"].sum()
    total_tco2e = total_kgco2e / 1000
    carbon_cost = total_tco2e * carbon_price_per_ton
    effective_cost = total_cost + carbon_cost

    print("\n" + "=" * 70)
    print("  BASELINE RESULTS (30-day simulation)")
    print("=" * 70)

    print(f"\n  Total jobs:           {len(merged):,}")
    print(f"  Total cloud spend:    ${total_cost:,.2f}")
    print(f"  Total emissions:      {total_kgco2e:,.2f} kgCO₂e ({total_tco2e:,.3f} tCO₂e)")
    print(f"  Carbon cost (@$75/t): ${carbon_cost:,.2f}")
    print(f"  Effective cost:       ${effective_cost:,.2f}")
    print(f"  Carbon % of eff.cost: {carbon_cost / effective_cost * 100:.2f}%")

    # ── By workload type ──────────────────────────────────────────────
    print(f"\n{'─' * 70}")
    print("  BY WORKLOAD TYPE")
    print(f"{'─' * 70}")
    by_type = merged.groupby("workload_type").agg(
        jobs=("job_id", "count"),
        cost_usd=("cost_usd", "sum"),
        kgco2e=("kgco2e", "sum"),
        avg_duration_hrs=("duration_hours", "mean"),
    ).round(2)
    by_type["$/job"] = (by_type["cost_usd"] / by_type["jobs"]).round(4)
    by_type["gCO2e/job"] = (by_type["kgco2e"] * 1000 / by_type["jobs"]).round(2)
    print(by_type.to_string())

    # ── By region ─────────────────────────────────────────────────────
    print(f"\n{'─' * 70}")
    print("  BY REGION")
    print(f"{'─' * 70}")
    by_region = merged.groupby("region").agg(
        jobs=("job_id", "count"),
        cost_usd=("cost_usd", "sum"),
        kgco2e=("kgco2e", "sum"),
    ).round(2)
    by_region["avg_gCO2e/job"] = (by_region["kgco2e"] * 1000 / by_region["jobs"]).round(2)
    print(by_region.to_string())

    # ── By category ───────────────────────────────────────────────────
    print(f"\n{'─' * 70}")
    print("  BY WORKLOAD CATEGORY (flexibility for optimization)")
    print(f"{'─' * 70}")
    by_cat = merged.groupby("category").agg(
        jobs=("job_id", "count"),
        cost_usd=("cost_usd", "sum"),
        kgco2e=("kgco2e", "sum"),
    ).round(2)
    by_cat["% of emissions"] = (by_cat["kgco2e"] / total_kgco2e * 100).round(1)
    print(by_cat.to_string())

    # ── Optimization opportunity ──────────────────────────────────────
    flexible = merged[merged["category"].isin(["balanced", "sustainable"])]
    print(f"\n{'─' * 70}")
    print("  OPTIMIZATION OPPORTUNITY")
    print(f"{'─' * 70}")
    print(f"  Flexible workloads (balanced + sustainable): {len(flexible):,} jobs "
          f"({len(flexible)/len(merged)*100:.0f}% of total)")
    print(f"  Their emissions: {flexible['kgco2e'].sum():,.2f} kgCO₂e "
          f"({flexible['kgco2e'].sum()/total_kgco2e*100:.0f}% of total)")
    print(f"  → These are the workloads the Planner can optimize")

    # ── Save to CSV ───────────────────────────────────────────────────
    output_path = "data/baseline_results.csv"
    merged.to_csv(output_path, index=False)
    print(f"\n  Results saved to: {output_path}")

    intensity_path = "data/carbon_intensity.csv"
    intensity_df.to_csv(intensity_path, index=False)
    print(f"  Intensity data saved to: {intensity_path}")

    print(f"\n{'=' * 70}")
    print("  NEXT STEP: Build the Planner agent to optimize flexible workloads")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
