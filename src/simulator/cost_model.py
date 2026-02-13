"""
Cloud Cost Model
================
Simple cost model: maps (region, resource_type) → $/hour.

In a real system, this would query AWS/GCP pricing APIs.
Here we use a lookup table with realistic-ish rates.

All prices are Assumptions based on approximate AWS on-demand pricing (2024).
"""

# ── Pricing tables ────────────────────────────────────────────────────
# $/hour per vCPU (approximate on-demand for general-purpose instances)
# Known range: $0.02–0.06/vCPU-hr depending on region and instance family

VCPU_COST_PER_HOUR = {
    "us-east-1":  0.040,   # Virginia — cheapest US region typically
    "us-west-2":  0.040,   # Oregon — same as us-east-1 usually
    "eu-west-1":  0.046,   # Ireland — ~15% premium over US
    "eu-north-1": 0.048,   # Stockholm — slightly more than Ireland
    "ap-south-1": 0.038,   # Mumbai — often cheapest
}

# $/hour per GPU (approximate for NVIDIA T4 / A10G equivalent)
GPU_COST_PER_HOUR = {
    "us-east-1":  0.75,
    "us-west-2":  0.75,
    "eu-west-1":  0.85,
    "eu-north-1": 0.90,
    "ap-south-1": 0.70,
}

# Cross-region data transfer ($/GB egress)
# Known: AWS charges $0.01-0.09/GB for cross-region transfer
EGRESS_COST_PER_GB = {
    ("same", "same"):       0.00,    # Same region — free
    ("NA", "NA"):           0.02,    # Within North America
    ("EU", "EU"):           0.02,    # Within Europe
    ("NA", "EU"):           0.05,    # Cross-Atlantic
    ("EU", "NA"):           0.05,
    ("NA", "AS"):           0.08,    # US to Asia
    ("AS", "NA"):           0.08,
    ("EU", "AS"):           0.07,    # Europe to Asia
    ("AS", "EU"):           0.07,
}

# Average data per job by workload type (GB) — Assumption
DATA_PER_JOB_GB = {
    "ci_cd":            2,
    "batch_analytics": 50,
    "model_training":  20,
    "dev_test":         1,
    "production":       5,
}


def compute_job_cost(
    region: str,
    vcpus: int,
    gpu_count: int,
    duration_hours: float,
) -> float:
    """
    Compute the cloud cost for a job (compute only, no egress).
    
    Returns: cost in USD
    """
    vcpu_rate = VCPU_COST_PER_HOUR.get(region, 0.045)
    gpu_rate = GPU_COST_PER_HOUR.get(region, 0.80)

    cost = (vcpus * vcpu_rate + gpu_count * gpu_rate) * duration_hours
    return round(cost, 4)


def compute_egress_cost(
    from_region: str,
    to_region: str,
    data_gb: float,
) -> float:
    """
    Compute the data transfer cost for moving a workload between regions.
    
    Returns: egress cost in USD
    """
    from src.shared.models import REGIONS

    if from_region == to_region:
        return 0.0

    from_continent = REGIONS.get(from_region, {}).get("continent", "NA")
    to_continent = REGIONS.get(to_region, {}).get("continent", "NA")

    rate = EGRESS_COST_PER_GB.get(
        (from_continent, to_continent),
        0.05  # Default fallback
    )

    return round(data_gb * rate, 4)


def compute_total_cost(
    region: str,
    vcpus: int,
    gpu_count: int,
    duration_hours: float,
    original_region: str = "",
    workload_type: str = "ci_cd",
) -> dict:
    """
    Compute total cost including compute + egress (if region changed).
    
    Returns:
        {"compute_cost": float, "egress_cost": float, "total_cost": float}
    """
    compute = compute_job_cost(region, vcpus, gpu_count, duration_hours)

    egress = 0.0
    if original_region and original_region != region:
        data_gb = DATA_PER_JOB_GB.get(workload_type, 5)
        egress = compute_egress_cost(original_region, region, data_gb)

    return {
        "compute_cost": compute,
        "egress_cost": egress,
        "total_cost": round(compute + egress, 4),
    }


# ── Quick self-test ───────────────────────────────────────────────────
if __name__ == "__main__":
    print("Cost model examples:")
    print()

    # CI/CD job: 4 vCPUs, 12 min, us-east-1
    c1 = compute_total_cost("us-east-1", vcpus=4, gpu_count=0, duration_hours=0.2)
    print(f"CI/CD (us-east-1, 4 vCPU, 12min): ${c1['total_cost']:.4f}")

    # Same job in eu-north-1 (moved)
    c2 = compute_total_cost("eu-north-1", vcpus=4, gpu_count=0, duration_hours=0.2,
                            original_region="us-east-1", workload_type="ci_cd")
    print(f"CI/CD (→eu-north-1, moved):        ${c2['total_cost']:.4f} "
          f"(compute: ${c2['compute_cost']:.4f}, egress: ${c2['egress_cost']:.4f})")

    # Model training: 8 vCPUs + 1 GPU, 6 hours
    c3 = compute_total_cost("us-east-1", vcpus=8, gpu_count=1, duration_hours=6.0)
    print(f"Training (us-east-1, 8vCPU+1GPU, 6hr): ${c3['total_cost']:.2f}")

    c4 = compute_total_cost("us-west-2", vcpus=8, gpu_count=1, duration_hours=6.0,
                            original_region="us-east-1", workload_type="model_training")
    print(f"Training (→us-west-2, moved):           ${c4['total_cost']:.2f} "
          f"(compute: ${c4['compute_cost']:.2f}, egress: ${c4['egress_cost']:.2f})")
