# detector.py

import math

# 1) Baseline computation
def compute_baseline(samples):
    """
    samples: list of dicts, e.g.
      [{"temperature": 37.1, "ph": 7.0, "rpm": 500}, ...]
    returns: {var: {"mu": ..., "sigma": ...}, ...}
    """
    vars_ = samples[0].keys()
    baseline = {}

    for v in vars_:
        values = [s[v] for s in samples]
        mu = sum(values) / len(values)
        var = sum((x - mu) ** 2 for x in values) / len(values)
        sigma = math.sqrt(var) if var > 0 else 1e-6
        baseline[v] = {"mu": mu, "sigma": sigma}

    return baseline

# 2) Score for one sample
def anomaly_score(sample, baseline):
    z_scores = {}
    for v, stats in baseline.items():
        x = sample[v]
        mu = stats["mu"]
        sigma = stats["sigma"]
        z = abs(x - mu) / sigma
        z_scores[v] = z
    score = max(z_scores.values())
    return score, z_scores

# 3) Turn score into flag
def anomaly_flag(score, threshold):
    return 1 if score > threshold else 0

# 4) Update confusion matrix
def update_confusion_matrix(cm, anomaly, fault_present):
    if anomaly == 1 and fault_present == 1:
        cm["TP"] += 1
    elif anomaly == 0 and fault_present == 0:
        cm["TN"] += 1
    elif anomaly == 1 and fault_present == 0:
        cm["FP"] += 1
    elif anomaly == 0 and fault_present == 1:
        cm["FN"] += 1
