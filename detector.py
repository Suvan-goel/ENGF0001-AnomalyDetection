"""ENGF0001 - Simple MQTT anomaly detector

This script trains a lightweight anomaly detector on a fault-free MQTT
telemetry stream and runs online detection on subsequent messages.

Algorithm: per-variable z-score (baseline mean and stddev). The sample
score is the maximum absolute z-score over the monitored variables. If
that score exceeds the user-specified threshold the sample is flagged as
anomalous.

Usage: see README.md or run `python detector.py -h`.
"""

from __future__ import annotations

import argparse
import json
import queue
import signal
import sys
import threading
import time
from collections import defaultdict
from typing import Dict, List, Tuple

try:
    import numpy as np
except Exception:
    print("Missing dependency: numpy. Install dependencies with:\n  python3 -m pip install -r requirements.txt")
    sys.exit(1)

try:
    import paho.mqtt.client as mqtt
except Exception:
    mqtt = None
    print("Optional dependency paho-mqtt not available; MQTT functionality will be disabled.\nInstall with: python3 -m pip install -r requirements.txt")


def compute_baseline(samples: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    """Compute mean and std (population) for each variable.

    Returns a dict mapping variable -> {"mu": float, "sigma": float}
    """
    if not samples:
        raise ValueError("No samples provided to compute baseline")
    vars_ = list(samples[0].keys())
    baseline: Dict[str, Dict[str, float]] = {}
    arr = np.array([[s[v] for v in vars_] for s in samples], dtype=float)
    mu = np.mean(arr, axis=0)
    sigma = np.std(arr, axis=0)
    # avoid zero sigma
    sigma[sigma == 0] = 1e-6
    for i, v in enumerate(vars_):
        baseline[v] = {"mu": float(mu[i]), "sigma": float(sigma[i])}
    return baseline


def anomaly_score(sample: Dict[str, float], baseline: Dict[str, Dict[str, float]]) -> Tuple[float, Dict[str, float]]:
    z_scores: Dict[str, float] = {}
    for v, stats in baseline.items():
        x = float(sample[v])
        mu = stats["mu"]
        sigma = stats["sigma"]
        z = abs(x - mu) / sigma
        z_scores[v] = z
    score = max(z_scores.values())
    return score, z_scores


def identify_fault_from_zscores(z_scores: Dict[str, float]) -> str:
    """Return a simple fault name based on the variable with the highest z-score.

    Mapping is heuristic:
      - 'temperature' -> 'therm_voltage_bias'
      - 'ph' -> 'ph_offset_bias'
      - 'rpm' -> 'heater_power_loss'

    Returns empty string if no mapping.
    """
    if not z_scores:
        return ""
    var = max(z_scores.items(), key=lambda kv: kv[1])[0]
    mapping = {
        "temperature": "therm_voltage_bias",
        "temp": "therm_voltage_bias",
        "ph": "ph_offset_bias",
        "rpm": "heater_power_loss",
    }
    return mapping.get(var, "")


def identify_fault(sample: Dict[str, float], baseline: Dict[str, Dict[str, float]]) -> Tuple[str, Dict[str, float]]:
    """Compute z-scores and return predicted fault name and z-scores.

    This is a lightweight heuristic useful for optional fault labelling.
    """
    score, z_scores = anomaly_score(sample, baseline)
    fault = identify_fault_from_zscores(z_scores)
    return fault, z_scores


def compute_covariance_baseline(samples: List[Dict[str, float]]):
    """Compute multivariate mean and inverse covariance for Mahalanobis distance.

    Returns mu (1D array) and invcov (2D array)
    """
    vars_ = list(samples[0].keys())
    arr = np.array([[s[v] for v in vars_] for s in samples], dtype=float)
    mu = np.mean(arr, axis=0)
    cov = np.cov(arr, rowvar=False)
    # regularize if singular
    cov += np.eye(cov.shape[0]) * 1e-6
    invcov = np.linalg.inv(cov)
    return vars_, mu, invcov


def mahalanobis_score(sample: Dict[str, float], vars_, mu, invcov) -> float:
    x = np.array([sample[v] for v in vars_], dtype=float)
    d2 = float((x - mu).T.dot(invcov).dot(x - mu))
    return float(np.sqrt(d2))


def anomaly_flag(score: float, threshold: float) -> int:
    return 1 if score > threshold else 0


def update_confusion_matrix(cm: Dict[str, int], anomaly: int, fault_present: int) -> None:
    if anomaly == 1 and fault_present == 1:
        cm["TP"] += 1
    elif anomaly == 0 and fault_present == 0:
        cm["TN"] += 1
    elif anomaly == 1 and fault_present == 0:
        cm["FP"] += 1
    elif anomaly == 0 and fault_present == 1:
        cm["FN"] += 1


def find_value(d: dict, candidates: List[str]):
    """Helper: return first matching numeric value for any key in candidates (case-insensitive)."""
    lk = {k.lower(): v for k, v in d.items()}
    for c in candidates:
        for k, v in lk.items():
            if c in k:
                try:
                    return float(v)
                except Exception:
                    pass
    return None


def extract_sample(msg: dict) -> Dict[str, float]:
    """Extract temperature, ph and rpm from a telemetry message dict.

    This function is conservative and tries several likely key names.
    It will raise KeyError if mandatory fields cannot be found.
    """
    # temperature candidates: keys containing 'temp' or 'temperature'
    t = find_value(msg, ["temperature", "temp"])
    ph = find_value(msg, ["ph"])
    rpm = find_value(msg, ["rpm", "stir", "stirring"])

    # Some simulator streams send summary objects where the key maps to a dict
    # with statistics (e.g. 'temperature_C': {'mean': ..., 'min': ..., ...}).
    # Try to extract numeric means from such structures if direct numeric
    # values aren't available.
    if (t is None or ph is None or rpm is None):
        for k, v in msg.items():
            if isinstance(v, dict) and 'mean' in v:
                lk = k.lower()
                try:
                    mean_val = float(v['mean'])
                except Exception:
                    # sometimes mean itself is nested; skip if not numeric
                    continue
                if t is None and ('temp' in lk or 'temperature' in lk):
                    t = mean_val
                if ph is None and ('ph' in lk):
                    ph = mean_val
                if rpm is None and ('rpm' in lk or 'stir' in lk or 'stirring' in lk):
                    rpm = mean_val

    # also try nested dicts (some messages place telemetry under 'telemetry' or 'summary')
    if (t is None or ph is None or rpm is None):
        for k in ("telemetry", "summary", "data"):
            if k in msg and isinstance(msg[k], dict):
                sub = msg[k]
                t = t or find_value(sub, ["temperature", "temp"])
                ph = ph or find_value(sub, ["ph"])
                rpm = rpm or find_value(sub, ["rpm", "stir", "stirring"])
                # also attempt summary-mean extraction in nested dict
                for kk, vv in sub.items():
                    if isinstance(vv, dict) and 'mean' in vv:
                        lk = kk.lower()
                        try:
                            mean_val = float(vv['mean'])
                        except Exception:
                            continue
                        if t is None and ('temp' in lk or 'temperature' in lk):
                            t = mean_val
                        if ph is None and ('ph' in lk):
                            ph = mean_val
                        if rpm is None and ('rpm' in lk or 'stir' in lk or 'stirring' in lk):
                            rpm = mean_val

    if t is None or ph is None or rpm is None:
        raise KeyError("Could not extract temperature/ph/rpm from message")
    return {"temperature": float(t), "ph": float(ph), "rpm": float(rpm)}


class MQTTDetector:
    def __init__(self, broker: str, port: int, topic: str, threshold: float, train_samples: int):
        # Sliding window config (used to update baseline online)
        self.window: List[Dict[str, float]] = []
        self.window_size = train_samples
        # Hysteresis thresholds (high, low) - high is provided, low defaults to half
        self.threshold_high = threshold
        self.threshold_low = threshold * 0.5
        # Mahalanobis option (can be enabled via CLI)
        self.use_mahalanobis = False
        self._maha_vars = None
        self._maha_mu = None
        self._maha_invcov = None

        self.broker = broker
        self.port = port
        self.topic = topic
        self.threshold = threshold
        self.train_samples = train_samples

        # Message queue (works even if MQTT is not available; allows offline testing)
        self._queue: "queue.Queue[str]" = queue.Queue()
        self._stop = threading.Event()
        self.baseline = None
        self.cm = defaultdict(int)
        # whether to save artifacts (baseline/confusion) at the end of the run
        self.save_artifacts = False

        # MQTT client is optional; only create if paho-mqtt is installed
        self.mqtt_enabled = mqtt is not None
        if self.mqtt_enabled:
            self._client = mqtt.Client()
            # wire callbacks
            self._client.on_connect = self._on_connect
            self._client.on_message = self._on_message
        else:
            self._client = None

    def _on_connect(self, client, userdata, flags, rc):
        print(f"Connected to {self.broker}:{self.port} (rc={rc}), subscribing to {self.topic}")
        client.subscribe(self.topic)

    def _on_message(self, client, userdata, msg):
        try:
            payload = msg.payload.decode("utf-8")
            # quick server-side debug: log small preview so we know messages arrive
            try:
                preview = payload if len(payload) < 200 else payload[:200] + '...'
                print(f"[MQTT] received on {msg.topic}: {preview}")
            except Exception:
                pass
            self._queue.put(payload)
        except Exception:
            pass

    def connect(self):
        if not self.mqtt_enabled:
            print("MQTT client not available; run in offline/test mode or install paho-mqtt to enable live mode")
            return
        self._client.connect(self.broker, self.port, keepalive=60)
        # run network loop in background thread
        t = threading.Thread(target=self._client.loop_forever, daemon=True)
        t.start()

    def train(self, progress_callback=None):
        """Collect `train_samples` training samples from the configured topic.

        If `progress_callback` is provided it will be called with the integer
        number of samples collected so far so callers (e.g. a UI) can report
        progress back to users.
        """
        print(f"Collecting {self.train_samples} training samples from {self.topic}...")
        samples: List[Dict[str, float]] = []
        self.window = []
        while len(samples) < self.train_samples:
            try:
                payload = self._queue.get(timeout=5.0)
            except queue.Empty:
                print("Waiting for training data...")
                continue
            try:
                msg = json.loads(payload)
                sample = extract_sample(msg)
                samples.append(sample)
                # initialize sliding window with training data
                self.window.append(sample)
                # report progress if requested
                if progress_callback:
                    try:
                        progress_callback(len(samples))
                    except Exception:
                        pass
                if len(samples) % 20 == 0:
                    print(f"  collected {len(samples)}")
            except Exception:
                # skip malformed
                continue
        self.baseline = compute_baseline(samples)
        # also compute Mahalanobis baseline
        try:
            self._maha_vars, self._maha_mu, self._maha_invcov = compute_covariance_baseline(samples)
        except Exception:
            self._maha_vars = None
            self._maha_mu = None
            self._maha_invcov = None
        print("Baseline trained:")
        for k, v in self.baseline.items():
            print(f"  {k}: mu={v['mu']:.3f}, sigma={v['sigma']:.3f}")

    def run(self):
        print("Starting online detection. Press Ctrl-C to stop.")
        try:
            while not self._stop.is_set():
                try:
                    payload = self._queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                try:
                    msg = json.loads(payload)
                except Exception:
                    continue
                # extract the sample and optional fault labels
                try:
                    sample = extract_sample(msg)
                except KeyError:
                    # can't evaluate this message
                    continue
                # update sliding window
                self.window.append(sample)
                if len(self.window) > self.window_size:
                    self.window.pop(0)

                # compute score using current window baseline (Mahalanobis or univariate)
                if self.use_mahalanobis and len(self.window) >= 2:
                    try:
                        vars_, mu, invcov = compute_covariance_baseline(self.window)
                        score = mahalanobis_score(sample, vars_, mu, invcov)
                    except Exception:
                        score, z = anomaly_score(sample, self.baseline)
                else:
                    try:
                        baseline = compute_baseline(self.window)
                        score, z = anomaly_score(sample, baseline)
                    except Exception:
                        score, z = anomaly_score(sample, self.baseline)

                # dual-threshold hysteresis
                if score >= self.threshold_high:
                    flag = 1
                elif score <= self.threshold_low:
                    flag = 0
                else:
                    flag = getattr(self, '_last_flag', 0)
                self._last_flag = flag
                faults = msg.get("faults") or msg.get("active_faults") or []
                fault_present = 1 if faults else 0
                update_confusion_matrix(self.cm, flag, fault_present)
                print(f"[{time.strftime('%H:%M:%S')}] score={score:.2f} flag={flag} faults={faults} sample={sample}")
        except KeyboardInterrupt:
            pass
        finally:
            self.summary()

    def summary(self):
        print("\n=== Detection summary ===")
        for k in ("TP", "TN", "FP", "FN"):
            print(f"{k}: {self.cm[k]}")
        # optionally persist artifacts
        if self.save_artifacts:
            try:
                import os, json
                os.makedirs('artifacts', exist_ok=True)
                # save confusion
                with open(os.path.join('artifacts', 'confusion.json'), 'w') as f:
                    json.dump(self.cm, f, indent=2)
                # save baseline (try window then baseline)
                baseline_to_save = None
                try:
                    if hasattr(self, 'window') and self.window:
                        baseline_to_save = compute_baseline(self.window)
                except Exception:
                    baseline_to_save = None
                if baseline_to_save is None and self.baseline is not None:
                    baseline_to_save = self.baseline
                if baseline_to_save is not None:
                    with open(os.path.join('artifacts', 'baseline.json'), 'w') as f:
                        json.dump(baseline_to_save, f, indent=2)
                print('Artifacts saved to artifacts/')
            except Exception as e:
                print('Failed to save artifacts:', e)


def main():
    p = argparse.ArgumentParser(description="Simple MQTT anomaly detector for ENGF0001 streams")
    p.add_argument("--broker", default="engf0001.cs.ucl.ac.uk", help="MQTT broker host")
    p.add_argument("--port", type=int, default=1883, help="MQTT broker port")
    p.add_argument("--topic", default="bioreactor_sim/nofaults/telemetry/summary", help="MQTT topic to subscribe to")
    p.add_argument("--train-samples", type=int, default=120, help="Number of samples to collect for baseline training (seconds)")
    p.add_argument("--threshold", type=float, default=4.0, help="z-score threshold for anomaly flag (high threshold). Low threshold will be half of this by default.")
    p.add_argument("--threshold-low", type=float, default=None, help="Optional explicit low threshold for hysteresis (if omitted, set to threshold/2)")
    p.add_argument("--mahalanobis", action="store_true", help="Use Mahalanobis distance (multivariate) instead of max z-score")
    args = p.parse_args()

    detector = MQTTDetector(args.broker, args.port, args.topic, args.threshold, args.train_samples)
    if args.threshold_low is not None:
        detector.threshold_low = args.threshold_low
    detector.threshold_high = args.threshold
    detector.threshold = args.threshold
    detector.use_mahalanobis = bool(args.mahalanobis)
    detector.connect()

    # allow clean shutdown on SIGINT
    def _sigint(signum, frame):
        detector._stop.set()

    signal.signal(signal.SIGINT, _sigint)

    # train then run
    detector.train()
    detector.run()


if __name__ == "__main__":
    main()
