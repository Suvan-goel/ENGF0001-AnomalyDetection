# ENGF0001 — Anomaly Detection (short report)

This short report summarises the detector implemented in this folder and provides reproducible results from a synthetic run.

1. Approach
- Detector: per-variable z-score (|x - mu|/sigma) with sample score = max z across temperature, pH, and RPM.
- Hysteresis: two thresholds (tau_high and tau_low) are used. The detector raises an anomaly when score >= tau_high and clears it when score <= tau_low. This reduces flicker.
- Multivariate option: an optional Mahalanobis-distance mode is available to capture correlated deviations between variables.
- Baseline: computed from an initial training window (default 120s), and updated online with a sliding window of recent samples.

2. Files of interest
- `detector.py`: main detector code (MQTT-compatible). Use `--mahalanobis` to enable multivariate mode and `--threshold-low` to set the low hysteresis threshold. Use `--train-samples` to set baseline training size. Add `--save-artifacts` to persist `artifacts/baseline.json` and `artifacts/confusion.json` after the run.
- `synthetic_test.py`: offline test harness. Produces `examples/synthetic_run.txt` and writes artifacts to `artifacts/` for reproducibility.
- `requirements.txt`: dependencies (numpy, paho-mqtt).
- `README.md`: step-by-step instructions for users unfamiliar with Python.

3. Reproducible synthetic run (used for evaluation)
- Command used:
```
python3 synthetic_test.py --duration 600 --seed 42
```
- Parameters: window=120 (training), tau_high=4.0, tau_low=2.0 (default), MTBF=300s, MTTR=60s.
- Result (from `examples/synthetic_run.txt` / `artifacts/confusion.json`):

- True Positives (TP): 180
- True Negatives (TN): 419
- False Positives (FP): 1
- False Negatives (FN): 0

4. How to run live demo (short)
- If off-campus, connect to UCL VPN.
- Install dependencies:
```
python3 -m pip install -r requirements.txt
```
- Run detector (train 120s then run):
```
python3 detector.py --topic bioreactor_sim/nofaults/telemetry/summary --train-samples 120 --threshold 4.0 --save-artifacts
```
- After stopping with Ctrl-C, artifacts will be in `artifacts/` (baseline.json and confusion.json).

5. Notes and limitations
- The synthetic test injects simple deterministic biases for faults (e.g. +2.5°C temp bias), so the detector performs very well under those conditions. Real simulator streams include control-loop reactions and more complex transients; consider using Mahalanobis mode and rolling baselines for robustness.
- For submission, include `examples/synthetic_run.txt` and `artifacts/confusion.json` to demonstrate reproducible output.

6. Next improvements (optional)
- Add small unit tests (pytest) for core functions.
- Implement simple fault labelling (map highest z variable to likely fault name).
- Add a threshold-sweep script to produce ROC/PR curves for systematic tuning.

End of report.
