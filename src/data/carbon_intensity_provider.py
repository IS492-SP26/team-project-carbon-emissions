"""
Real carbon intensity data provider.
Primary: UK Carbon Intensity API (free, no key)
Secondary: Electricity Maps API (requires token)
Fallback: synthetic generator with loud warning
"""
import requests
import pandas as pd
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

UK_API = "https://api.carbonintensity.org.uk"
CACHE_PATH = Path("data/real_carbon_intensity.csv")

def fetch_uk_carbon_intensity(hours: int = 48) -> pd.DataFrame:
    """Fetch real GB grid carbon intensity. Free, no API key needed."""
    end = datetime.now(timezone.utc)
    start = end - timedelta(hours=hours)
    url = f"{UK_API}/intensity/{start.strftime('%Y-%m-%dT%H:%MZ')}/{end.strftime('%Y-%m-%dT%H:%MZ')}"
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()["data"]
        records = [
            {
                "timestamp": d["from"],
                "carbon_intensity_gco2_kwh": d["intensity"]["actual"] or d["intensity"]["forecast"],
                "source": "carbonintensity.org.uk",
                "is_real": True,
                "region": "GB"
            }
            for d in data if d["intensity"]["actual"] or d["intensity"]["forecast"]
        ]
        df = pd.DataFrame(records)
        CACHE_PATH.parent.mkdir(exist_ok=True)
        df.to_csv(CACHE_PATH, index=False)
        logger.info(f"Fetched {len(df)} real carbon intensity readings from carbonintensity.org.uk")
        return df
    except Exception as e:
        logger.warning(f"UK Carbon API failed: {e}. Falling back to synthetic data.")
        return _synthetic_fallback(hours)

def _synthetic_fallback(hours: int) -> pd.DataFrame:
    import numpy as np
    logger.warning("WARNING: Using SYNTHETIC carbon intensity data. Results are NOT real-world validated.")
    timestamps = [datetime.now(timezone.utc) - timedelta(minutes=30*i) for i in range(hours*2)]
    return pd.DataFrame({
        "timestamp": timestamps,
        "carbon_intensity_gco2_kwh": np.random.normal(250, 80, len(timestamps)).clip(50, 600),
        "source": "synthetic",
        "is_real": False,
        "region": "synthetic"
    })

def get_carbon_intensity(hours: int = 48) -> pd.DataFrame:
    """Main entry point. Returns real data if available, synthetic if not."""
    if CACHE_PATH.exists():
        df = pd.read_csv(CACHE_PATH)
        latest = pd.to_datetime(df["timestamp"], utc=True).max()
        now = datetime.now(timezone.utc)
        if now - latest < timedelta(hours=6):
            logger.info("Using cached real carbon intensity data.")
            return df
    return fetch_uk_carbon_intensity(hours)
