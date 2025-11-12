import json
import os

import numpy as np

from detector import compute_baseline, anomaly_score, extract_sample, identify_fault


def test_compute_baseline_simple():
    samples = [
        {"temperature": 10.0, "ph": 7.0, "rpm": 100.0},
        {"temperature": 12.0, "ph": 7.0, "rpm": 100.0},
    ]
    baseline = compute_baseline(samples)
    assert abs(baseline["temperature"]["mu"] - 11.0) < 1e-6
    assert baseline["ph"]["sigma"] == 0 or baseline["ph"]["sigma"] >= 0


def test_anomaly_score_and_identify():
    samples = [
        {"temperature": 37.0, "ph": 7.0, "rpm": 500.0},
        {"temperature": 37.2, "ph": 7.0, "rpm": 500.0},
    ]
    baseline = compute_baseline(samples)
    sample = {"temperature": 40.0, "ph": 7.0, "rpm": 500.0}
    score, z = anomaly_score(sample, baseline)
    assert score > 0
    fault, zscores = identify_fault(sample, baseline)
    assert fault == "therm_voltage_bias"


def test_extract_sample_variants():
    msg1 = {"temperature": 37.0, "ph": 7.0, "rpm": 500}
    s1 = extract_sample(msg1)
    assert "temperature" in s1 and "ph" in s1 and "rpm" in s1

    msg2 = {"summary": {"temp": 36.9, "ph": 6.9, "stir": 490}}
    s2 = extract_sample(msg2)
    assert s2["temperature"] == 36.9
