"""
Tests for src/agents/carbon_accountant.py

Validates that the emissions formula correctly applies the CPU utilization
factor to avoid over-estimating power consumption from TDP alone.
"""

import pytest
import pandas as pd
from datetime import datetime

from src.agents.carbon_accountant import (
    TDP_PER_VCPU_KW,
    AVERAGE_CPU_UTILIZATION,
    GPU_TDP_KW,
    PUE,
    compute_emissions_single,
    compute_emissions_for_config,
    compute_emissions_batch,
)
from src.shared.models import Job
from src.simulator.carbon_intensity import generate_intensity_timeseries


@pytest.fixture(scope="module")
def intensity_df():
    return generate_intensity_timeseries(datetime(2025, 1, 1), num_days=2, seed=42)


class TestConstants:
    def test_cpu_utilization_factor_in_range(self):
        """CPU utilization must be between 0 and 1."""
        assert 0 < AVERAGE_CPU_UTILIZATION <= 1.0

    def test_cpu_utilization_reduces_overestimate(self):
        """Utilization factor should be strictly less than 1 to correct TDP over-estimate."""
        assert AVERAGE_CPU_UTILIZATION < 1.0

    def test_tdp_per_vcpu_positive(self):
        assert TDP_PER_VCPU_KW > 0

    def test_gpu_tdp_positive(self):
        assert GPU_TDP_KW > 0

    def test_pue_above_one(self):
        assert PUE >= 1.0


class TestComputeEmissionsSingle:
    def test_cpu_only_job_applies_utilization_factor(self, intensity_df):
        """
        A CPU-only job's power should be vCPUs × TDP × utilization × PUE.
        Without the utilization factor the result would be higher.
        """
        job = Job(
            region="us-east-1",
            vcpus=4,
            gpu_count=0,
            duration_hours=1.0,
            started_at=datetime(2025, 1, 1, 12, 0),
        )
        record = compute_emissions_single(job, intensity_df)
        intensity_kg = record.grid_intensity_used / 1000

        expected_power = 4 * TDP_PER_VCPU_KW * AVERAGE_CPU_UTILIZATION * PUE
        expected_energy = expected_power * 1.0
        expected_kgco2e = expected_energy * intensity_kg

        assert abs(record.kgco2e - expected_kgco2e) < 1e-5

    def test_gpu_job_uses_full_gpu_tdp(self, intensity_df):
        """
        GPU power should use full GPU_TDP_KW (no utilization discount for GPUs).
        """
        job = Job(
            region="eu-north-1",
            vcpus=8,
            gpu_count=1,
            duration_hours=2.0,
            started_at=datetime(2025, 1, 1, 6, 0),
        )
        record = compute_emissions_single(job, intensity_df)
        intensity_kg = record.grid_intensity_used / 1000

        expected_power = (8 * TDP_PER_VCPU_KW * AVERAGE_CPU_UTILIZATION + 1 * GPU_TDP_KW) * PUE
        expected_energy = expected_power * 2.0
        expected_kgco2e = expected_energy * intensity_kg

        assert abs(record.kgco2e - expected_kgco2e) < 1e-5

    def test_utilization_factor_lowers_emissions_vs_full_tdp(self, intensity_df):
        """
        Emissions with AVERAGE_CPU_UTILIZATION < 1.0 must be lower than
        they would be with 100% utilization (the former over-estimate).
        """
        job = Job(
            region="us-east-1",
            vcpus=4,
            gpu_count=0,
            duration_hours=1.0,
            started_at=datetime(2025, 1, 1, 12, 0),
        )
        record = compute_emissions_single(job, intensity_df)
        intensity_kg = record.grid_intensity_used / 1000

        # What the old formula (100% utilization) would have produced
        full_tdp_power = 4 * TDP_PER_VCPU_KW * PUE  # no utilization factor
        full_tdp_kgco2e = full_tdp_power * 1.0 * intensity_kg

        assert record.kgco2e < full_tdp_kgco2e

    def test_zero_vcpus_zero_emissions(self, intensity_df):
        job = Job(
            region="us-east-1",
            vcpus=0,
            gpu_count=0,
            duration_hours=1.0,
            started_at=datetime(2025, 1, 1, 12, 0),
        )
        record = compute_emissions_single(job, intensity_df)
        assert record.kgco2e == 0.0

    def test_uncertainty_bounds_order(self, intensity_df):
        job = Job(
            region="ap-south-1",
            vcpus=4,
            gpu_count=0,
            duration_hours=0.5,
            started_at=datetime(2025, 1, 1, 14, 0),
        )
        record = compute_emissions_single(job, intensity_df)
        assert record.kgco2e_lower <= record.kgco2e <= record.kgco2e_upper


class TestComputeEmissionsForConfig:
    def test_matches_single_computation(self, intensity_df):
        """compute_emissions_for_config and compute_emissions_single agree."""
        job = Job(
            region="us-west-2",
            vcpus=2,
            gpu_count=0,
            duration_hours=0.25,
            started_at=datetime(2025, 1, 1, 10, 0),
        )
        single = compute_emissions_single(job, intensity_df)
        config = compute_emissions_for_config(
            vcpus=2,
            gpu_count=0,
            duration_hours=0.25,
            region="us-west-2",
            timestamp=datetime(2025, 1, 1, 10, 0),
            intensity_df=intensity_df,
        )
        assert abs(single.kgco2e - config["kgco2e"]) < 1e-5


class TestComputeEmissionsBatch:
    def test_batch_total_positive(self, intensity_df):
        jobs = [
            Job(region="us-east-1", vcpus=4, gpu_count=0,
                duration_hours=0.2, started_at=datetime(2025, 1, 1, 9, 0)),
            Job(region="eu-north-1", vcpus=4, gpu_count=0,
                duration_hours=0.2, started_at=datetime(2025, 1, 1, 9, 0)),
        ]
        records = compute_emissions_batch(jobs, intensity_df)
        assert all(r.kgco2e >= 0 for r in records)
        assert sum(r.kgco2e for r in records) > 0

    def test_cleaner_region_lower_emissions(self, intensity_df):
        """eu-north-1 (hydro + nuclear, ~30 gCO₂/kWh) must emit less than us-east-1 (~350)."""
        dirty_job = Job(
            region="us-east-1", vcpus=4, gpu_count=0,
            duration_hours=1.0, started_at=datetime(2025, 1, 1, 12, 0),
        )
        clean_job = Job(
            region="eu-north-1", vcpus=4, gpu_count=0,
            duration_hours=1.0, started_at=datetime(2025, 1, 1, 12, 0),
        )
        records = compute_emissions_batch([dirty_job, clean_job], intensity_df)
        assert records[0].kgco2e > records[1].kgco2e
