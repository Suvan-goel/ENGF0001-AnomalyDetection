# ENGF0001 Anomaly Detector — quick guide

This repository contains a small, easy-to-run anomaly detector for the ENGF0001
simulator. It can run either live (subscribe to the simulator's MQTT topics) or
offline using a built-in synthetic-data test harness.

What the detector does (plain language)
- Collect ~N seconds of "normal" data from the simulator (the `nofaults` stream)
  and compute a simple baseline (average and spread) for temperature, pH and RPM.
- For each new reading, compute how far it is from the baseline. If it is far
  enough, raise an anomaly flag.
- The detector uses two thresholds (high and low) to avoid rapid on/off
  flickering (hysteresis). A multivariate option (Mahalanobis distance) is
  available for correlated signals.

Quick setup (one-time)
1. Install Python 3.8+ if you don't have it.
2. Open a terminal and change to this project folder.
3. Install required packages:

```bash
python3 -m pip install -r requirements.txt
```

If you cannot install packages (or just want to try offline), skip step 3 and
use the synthetic test described below.

Run the detector live (connects to the UCL simulator)
1. If you're off-campus, connect to the UCL VPN.
2. Run the detector and collect 120s of training data (this trains the baseline):

```bash
python3 detector.py --topic bioreactor_sim/nofaults/telemetry/summary --train-samples 120 --threshold 4.0
```

- `--threshold` sets the high threshold (tau_high). The low threshold (tau_low)
  defaults to half of this value (tau_low = tau_high / 2). Use `--threshold-low`
  to override. Example with explicit low threshold and Mahalanobis mode:

```bash
python3 detector.py --topic bioreactor_sim/nofaults/telemetry/summary --train-samples 120 --threshold 4.0 --threshold-low 2.0 --mahalanobis
```

3. After training the detector will print per-sample scores and a running
   confusion matrix when you stop it (Ctrl-C). To evaluate on a faulted
   stream, change the topic to `bioreactor_sim/single_fault/telemetry/summary` or
   `bioreactor_sim/three_faults/telemetry/summary`.

Offline synthetic test (no broker required)
- Run the synthetic test harness to generate training and test sequences locally
  and see TP/TN/FP/FN counts quickly. This is ideal for tuning thresholds and
  trying the Mahalanobis option without network access or VPN.

```bash
python3 synthetic_test.py
```

Advanced options
- `synthetic_test.py` and `detector.py` accept parameters to control:
  - duration, MTBF/MTTR, and random seed for reproducible experiments
  - choice of Mahalanobis (multivariate) vs simple max z-score
  - high and low thresholds for hysteresis

A few notes
- The detector expects JSON messages containing temperature, pH and RPM.
  The code will try common key names and nested message layouts. If the
  simulator uses different names, edit `extract_sample()` in `detector.py`.
- The Mahalanobis option uses a covariance estimate and therefore needs a
  moderate number of samples to compute a stable inverse covariance matrix.
- `__pycache__` and compiled Python files are ignored in `.gitignore` and are
  safe to delete if you want to clear cached bytecode.

If you want, I can add one-line scripts that run common experiments (e.g.
threshold sweep, ROC/PR plots) or a small GUI to control parameters interactively.
Tell me which you'd like next.
# ENGF0001 Anomaly Detector (simple)

This repository contains a simple MQTT-based anomaly detector for the ENGF0001
simulator telemetry streams used in the ENGF0001 course.

Features
- Train a baseline (mean/std) on the `nofaults` stream and run online detection
- Per-variable z-score; sample score = max z across monitored vars
- Simple confusion matrix bookkeeping (TP/TN/FP/FN) when fault labels are present in the message

Requirements
- Python 3.8+
- See `requirements.txt` (install with `pip install -r requirements.txt`).

Usage
1. If off-campus, connect to the UCL VPN.
2. Run the script:

```bash
python detector.py --topic bioreactor_sim/nofaults/telemetry/summary --train-samples 120 --threshold 4.0
```

- To evaluate on a faulted stream, change `--topic` to `bioreactor_sim/single_fault/telemetry/summary` or
  `bioreactor_sim/three_faults/telemetry/summary`.

Notes
- The detector expects JSON messages containing temperature, pH and RPM fields (it will try several common key names and nested payloads).
- The script prints per-sample scores and a final confusion matrix when stopped (Ctrl-C).
