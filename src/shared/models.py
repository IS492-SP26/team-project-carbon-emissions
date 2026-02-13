"""
Shared data models for the sust-AI-naible system.
These are the "nouns" — the data structures every agent speaks in.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional
import uuid


class WorkloadCategory(Enum):
    """How flexible a workload is for scheduling."""
    URGENT = "urgent"          # No deferral, same continent only
    BALANCED = "balanced"      # ≤4hr deferral, any region same provider
    SUSTAINABLE = "sustainable" # ≤24hr deferral, any region any provider


class ResourceType(Enum):
    COMPUTE = "compute"
    GPU = "gpu"
    STORAGE = "storage"
    NETWORK = "network"


class CloudProvider(Enum):
    AWS = "aws"
    GCP = "gcp"
    AZURE = "azure"


# ── Region definitions with metadata ──────────────────────────────────
# These are the 5 regions we simulate. Real system would pull from a config.

REGIONS = {
    "us-east-1":  {"name": "Virginia",   "provider": "aws",   "continent": "NA", "lat": 39.0, "lon": -77.5},
    "us-west-2":  {"name": "Oregon",     "provider": "aws",   "continent": "NA", "lat": 45.6, "lon": -120.5},
    "eu-west-1":  {"name": "Ireland",    "provider": "aws",   "continent": "EU", "lat": 53.3, "lon": -6.3},
    "eu-north-1": {"name": "Stockholm",  "provider": "aws",   "continent": "EU", "lat": 59.3, "lon": 18.1},
    "ap-south-1": {"name": "Mumbai",     "provider": "aws",   "continent": "AS", "lat": 19.1, "lon": 72.9},
}


@dataclass
class Job:
    """A single cloud workload (the atomic unit we measure and optimize)."""
    job_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str = ""
    team_id: str = ""
    service_name: str = ""
    region: str = "us-east-1"
    vcpus: int = 4
    gpu_count: int = 0
    duration_hours: float = 0.5
    cost_usd: float = 0.0
    category: WorkloadCategory = WorkloadCategory.BALANCED
    started_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None
    workload_type: str = "ci_cd"  # ci_cd | batch_analytics | model_training | dev_test | production


@dataclass
class EmissionsRecord:
    """Carbon emissions attributed to a single job."""
    activity_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    job_id: str = ""
    kgco2e: float = 0.0
    kgco2e_lower: float = 0.0       # Uncertainty lower bound
    kgco2e_upper: float = 0.0       # Uncertainty upper bound
    grid_intensity_used: float = 0.0  # gCO2/kWh at time of computation
    emission_factor_id: str = ""
    methodology: str = "location_based_v1"
    computed_at: Optional[datetime] = None


@dataclass
class Recommendation:
    """A Planner recommendation to change a workload's config."""
    recommendation_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    job_id: str = ""
    action_type: str = ""           # region_shift | time_shift | right_size
    current_region: str = ""
    proposed_region: str = ""
    current_time: Optional[datetime] = None
    proposed_time: Optional[datetime] = None
    est_carbon_delta_kg: float = 0.0  # negative = reduction
    est_cost_delta_usd: float = 0.0
    confidence: float = 0.0
    rationale: str = ""
    status: str = "proposed"        # proposed | approved | executed | verified
    risk_level: str = "low"         # low | medium | high


@dataclass
class VerificationRecord:
    """Proof that a recommendation's claimed savings are real (or not)."""
    verification_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    recommendation_id: str = ""
    counterfactual_kgco2e: float = 0.0   # What WOULD have happened
    actual_kgco2e: float = 0.0           # What DID happen
    verified_savings_kgco2e: float = 0.0  # counterfactual - actual
    ci_lower: float = 0.0                # 90% confidence interval
    ci_upper: float = 0.0
    sla_compliant: bool = True
    verification_status: str = "confirmed"  # confirmed | partial | refuted | inconclusive
    evidence_chain: list = field(default_factory=list)
    verified_at: Optional[datetime] = None
