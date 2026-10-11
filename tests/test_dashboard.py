import json

import pytest
from streamlit.testing.v1 import AppTest

from retainiq.pipeline import run_pipeline

ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def outputs(tmp_path_factory, raw_df):
    out = tmp_path_factory.mktemp("dash_outputs")
    run_pipeline(raw_df, out, n_iter=3, cv=3, source_label="synthetic")
    return out


def test_dashboard_renders_without_exception(outputs, monkeypatch):
    monkeypatch.setenv("RETAINIQ_OUTPUT_DIR", str(outputs))
    at = AppTest.from_file(str(ROOT / "app" / "dashboard.py"), default_timeout=60).run()
    assert not at.exception
    assert len(at.metric) == 4
    assert any("SYNTHETIC" in w.value for w in at.warning)


def test_dashboard_filter_reduces_customers(outputs, monkeypatch):
    monkeypatch.setenv("RETAINIQ_OUTPUT_DIR", str(outputs))
    at = AppTest.from_file(str(ROOT / "app" / "dashboard.py"), default_timeout=60).run()
    total = int(at.metric[0].value.replace(",", ""))
    at.sidebar.multiselect[2].set_value(["High"]).run()       # risk bucket filter
    assert not at.exception
    assert int(at.metric[0].value.replace(",", "")) < total


def test_dashboard_without_outputs_shows_instructions(tmp_path, monkeypatch):
    monkeypatch.setenv("RETAINIQ_OUTPUT_DIR", str(tmp_path / "missing"))
    at = AppTest.from_file(str(ROOT / "app" / "dashboard.py"), default_timeout=60).run()
    assert not at.exception                       # graceful: no traceback
    assert any("No pipeline outputs" in e.value for e in at.error)
