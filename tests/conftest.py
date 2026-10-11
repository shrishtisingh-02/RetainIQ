import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from retainiq.cleaning import clean_transactions  # noqa: E402
from retainiq.synthetic import generate_synthetic_transactions  # noqa: E402


@pytest.fixture(scope="session")
def raw_df():
    return generate_synthetic_transactions(n_customers=400, seed=7)


@pytest.fixture(scope="session")
def clean_df(raw_df):
    clean, _ = clean_transactions(raw_df)
    return clean
