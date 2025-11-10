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

import numpy as np
import paho.mqtt.client as mqtt


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
    if t is None or ph is None or rpm is None:
        # try nested dicts (some streams nest telemetry under 'telemetry' or 'summary')
        for k in ("telemetry", "summary", "data"):
            if k in msg and isinstance(msg[k], dict):
                t = t or find_value(msg[k], ["temperature", "temp"])
                ph = ph or find_value(msg[k], ["ph"])
                rpm = rpm or find_value(msg[k], ["rpm", "stir", "stirring"])
    if t is None or ph is None or rpm is None:
        raise KeyError("Could not extract temperature/ph/rpm from message")
    return {"temperature": float(t), "ph": float(ph), "rpm": float(rpm)}


class MQTTDetector:
    def __init__(self, broker: str, port: int, topic: str, threshold: float, train_samples: int):
        self.broker = broker
        self.port = port
        self.topic = topic
        self.threshold = threshold
        self.train_samples = train_samples

        self._client = mqtt.Client()
        self._queue: "queue.Queue[str]" = queue.Queue()
        self._stop = threading.Event()
        self.baseline = None
        self.cm = defaultdict(int)

        # wire callbacks
        self._client.on_connect = self._on_connect
        self._client.on_message = self._on_message

    def _on_connect(self, client, userdata, flags, rc):
        print(f"Connected to {self.broker}:{self.port} (rc={rc}), subscribing to {self.topic}")
        client.subscribe(self.topic)

    def _on_message(self, client, userdata, msg):
        try:
            payload = msg.payload.decode("utf-8")
            self._queue.put(payload)
        except Exception:
            pass

    def connect(self):
        self._client.connect(self.broker, self.port, keepalive=60)
        # run network loop in background thread
        t = threading.Thread(target=self._client.loop_forever, daemon=True)
        t.start()

    def train(self):
        print(f"Collecting {self.train_samples} training samples from {self.topic}...")
        samples: List[Dict[str, float]] = []
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
                if len(samples) % 20 == 0:
                    print(f"  collected {len(samples)}")
            except Exception:
                # skip malformed
                continue
        self.baseline = compute_baseline(samples)
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
                score, z = anomaly_score(sample, self.baseline)
                flag = anomaly_flag(score, self.threshold)
                faults = msg.get("faults") or msg.get("active_faults") or []
                fault_present = 1 if faults else 0
                update_confusion_matrix(self.cm, flag, fault_present)
                ts = msg.get("timestamp") or time.time()
                print(f"[{time.strftime('%H:%M:%S')}] score={score:.2f} flag={flag} faults={faults} sample={sample}")
        except KeyboardInterrupt:
            pass
        finally:
            self.summary()

    def summary(self):
        print("\n=== Detection summary ===")
        for k in ("TP", "TN", "FP", "FN"):
            print(f"{k}: {self.cm[k]}")


def main():
    p = argparse.ArgumentParser(description="Simple MQTT anomaly detector for ENGF0001 streams")
    p.add_argument("--broker", default="engf0001.cs.ucl.ac.uk", help="MQTT broker host")
    p.add_argument("--port", type=int, default=1883, help="MQTT broker port")
    p.add_argument("--topic", default="bioreactor_sim/nofaults/telemetry/summary", help="MQTT topic to subscribe to")
    p.add_argument("--train-samples", type=int, default=120, help="Number of samples to collect for baseline training (seconds)")
    p.add_argument("--threshold", type=float, default=4.0, help="z-score threshold for anomaly flag")
    args = p.parse_args()

    detector = MQTTDetector(args.broker, args.port, args.topic, args.threshold, args.train_samples)
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
