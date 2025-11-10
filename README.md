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
