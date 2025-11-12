"""Run the detector logic on synthetic data (no MQTT required).

Generates a fault-free training stream then a test stream with
stochastically-occurring faults (using exponential MTBF/MTTR) and
evaluates the simple z-score detector implemented in `detector.py`.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from typing import Dict, List

import numpy as np

from detector import compute_baseline, anomaly_score, anomaly_flag, update_confusion_matrix, compute_covariance_baseline, mahalanobis_score


def generate_segmented_fault_sequence(total_seconds: int, mtbf: float = 300.0, mttr: float = 60.0):
    """Yield a boolean list of length total_seconds where True indicates a fault present.

    mtbf and mttr are in seconds and used as the mean of exponential distributions.
    """
    seq = [False] * total_seconds
    t = 0

    while t < total_seconds:
        # time to next fault
        time_to_fault = int(np.random.exponential(mtbf))
        t += time_to_fault
        if t >= total_seconds:
            break

        # fault duration
        dur = int(np.random.exponential(mttr))
        for i in range(t, min(total_seconds, t + max(1, dur))):
            seq[i] = True
        t += max(1, dur)
    return seq


def make_sample(base_means: Dict[str, float], base_stds: Dict[str, float], fault: bool, fault_type: str = None) -> Dict[str, float]:
    t = np.random.normal(base_means['temperature'], base_stds['temperature'])
    ph = np.random.normal(base_means['ph'], base_stds['ph'])
    rpm = np.random.normal(base_means['rpm'], base_stds['rpm'])

    faults = []
    if fault:
        faults.append(fault_type or 'therm_voltage_bias')
        if fault_type == 'therm_voltage_bias':
            # add a bias to temperature reading
            t += 2.5  # +2.5 deg bias
        elif fault_type == 'ph_offset_bias':
            ph += -0.5
        elif fault_type == 'heater_power_loss':
            # simulate heater loss as lower temperature
            t -= 1.5

    return {'temperature': float(t), 'ph': float(ph), 'rpm': float(rpm), 'faults': faults}


def run_synthetic_test(duration_sec: int = 600, mtbf: float = 300.0, mttr: float = 60.0, seed: int | None = None,
                       use_mahalanobis: bool = True, threshold_high: float = 4.0, threshold_low: float | None = None,
                       window_size: int = 120):
    # base normal params
    base_means = {'temperature': 37.0, 'ph': 7.0, 'rpm': 500.0}
    base_stds = {'temperature': 0.2, 'ph': 0.05, 'rpm': 5.0}

    # Training: collect fault-free samples
    train_n = 120
    print(f"Generating {train_n} fault-free training samples...")
    train_samples = []
    for _ in range(train_n):
        s = make_sample(base_means, base_stds, fault=False)
        # only keep monitored fields
        train_samples.append({'temperature': s['temperature'], 'ph': s['ph'], 'rpm': s['rpm']})

    # seed RNGs for reproducibility if requested
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    baseline = compute_baseline(train_samples)
    print("Trained baseline:")
    for k, v in baseline.items():
        print(f"  {k}: mu={v['mu']:.3f}, sigma={v['sigma']:.3f}")

    # Create fault schedule
    fault_seq = generate_segmented_fault_sequence(duration_sec, mtbf=mtbf, mttr=mttr)

    cm = {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0}
    threshold_h = threshold_high
    threshold_l = threshold_low if threshold_low is not None else threshold_h * 0.5

    header = f"Running {duration_sec}s test, window={window_size}, tau_high={threshold_h}, tau_low={threshold_l}, mahalanobis={use_mahalanobis}, seed={seed}\n"
    print(header)

    # initialize sliding window with training data
    window: List[Dict[str, float]] = train_samples.copy()
    # if using Mahalanobis compute initial mu/invcov
    if use_mahalanobis:
        vars_, mu, invcov = compute_covariance_baseline(window)
    else:
        baseline = compute_baseline(window)

    last_flag = 0
    # collect lines for a human-readable log
    lines = [header]
    for i in range(duration_sec):
        fault_active = fault_seq[i]
        # randomly choose fault type when active
        fault_type = None
        if fault_active:
            fault_type = random.choice(['therm_voltage_bias', 'ph_offset_bias', 'heater_power_loss'])
        s = make_sample(base_means, base_stds, fault_active, fault_type)
        sample = {'temperature': s['temperature'], 'ph': s['ph'], 'rpm': s['rpm']}
        if use_mahalanobis:
            # recompute baseline periodically (here we update from window)
            vars_, mu, invcov = compute_covariance_baseline(window)
            score = mahalanobis_score(sample, vars_, mu, invcov)
        else:
            baseline = compute_baseline(window)
            score, _ = anomaly_score(sample, baseline)

        # hysteresis
        if score >= threshold_h:
            flag = 1
        elif score <= threshold_l:
            flag = 0
        else:
            flag = last_flag
        last_flag = flag
        update_confusion_matrix(cm, flag, 1 if fault_active else 0)
        if i < 10 or (i < 60 and i % 10 == 0) or (i % 60 == 0):
            line = f"t={i}s fault={fault_active} type={fault_type} score={score:.2f} flag={flag}"
            print(line)
            lines.append(line)
        # sleep removed to run quickly

    print("\n=== Synthetic test summary ===")
    for k in ('TP', 'TN', 'FP', 'FN'):
        print(f"{k}: {cm[k]}")

    # prepare artifacts directory
    os.makedirs('artifacts', exist_ok=True)
    os.makedirs('examples', exist_ok=True)
    # save confusion matrix
    with open(os.path.join('artifacts', 'confusion.json'), 'w') as f:
        json.dump(cm, f, indent=2)
    # save baseline (from final window)
    try:
        final_baseline = compute_baseline(window)
        with open(os.path.join('artifacts', 'baseline.json'), 'w') as f:
            json.dump(final_baseline, f, indent=2)
    except Exception:
        pass

    # write human-readable run log
    with open(os.path.join('examples', 'synthetic_run.txt'), 'w') as f:
        for l in lines:
            f.write(l + "\n")
        f.write("\n=== Summary ===\n")
        for k in ('TP', 'TN', 'FP', 'FN'):
            f.write(f"{k}: {cm[k]}\n")

    return cm


def parse_args_and_run():
    p = argparse.ArgumentParser(description='Run synthetic test and save artifacts')
    p.add_argument('--duration', type=int, default=600)
    p.add_argument('--mtbf', type=float, default=300.0)
    p.add_argument('--mttr', type=float, default=60.0)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--mahalanobis', action='store_true')
    p.add_argument('--threshold', type=float, default=4.0)
    p.add_argument('--threshold-low', type=float, default=None)
    p.add_argument('--window', type=int, default=120)
    args = p.parse_args()
    run_synthetic_test(duration_sec=args.duration, mtbf=args.mtbf, mttr=args.mttr, seed=args.seed,
                       use_mahalanobis=args.mahalanobis, threshold_high=args.threshold,
                       threshold_low=args.threshold_low, window_size=args.window)


if __name__ == '__main__':
    parse_args_and_run()
