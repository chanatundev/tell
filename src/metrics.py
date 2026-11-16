"""Small metric helpers shared by train.py, baselines.py and eval_heldout.py."""
import numpy as np
from sklearn.metrics import roc_auc_score


def wilson(k, n, z=1.96):
    """Wilson score 95% interval for a proportion k/n -> (lo, hi). n=0 -> (nan, nan)."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, (c - m) / d), min(1.0, (c + m) / d))


def fmt_prop(k, n):
    lo, hi = wilson(k, n)
    return f"{k}/{n} = {k / max(n, 1):.3f} (95% CI {lo:.3f}-{hi:.3f})"


def threshold_at_fpr(scores_real, max_fpr=0.05):
    """Smallest threshold t such that the share of REAL scores > t is <= max_fpr (flag if score > t)."""
    s = np.sort(np.asarray(scores_real))
    n = len(s)
    k = int(np.floor(max_fpr * n))  # allowed false alarms
    if k >= n:
        return float(s[0]) - 1e-9
    return float(s[n - k - 1])  # at most k real scores are strictly greater


def summarize(scores, labels, thr):
    """AUROC, recall and false-alarm rate with Wilson CIs at threshold thr (flag if score > thr)."""
    scores, labels = np.asarray(scores, float), np.asarray(labels)
    flagged = scores > thr
    ai, re = labels == 1, labels == 0
    out = dict(n_ai=int(ai.sum()), n_real=int(re.sum()),
               auroc=float(roc_auc_score(labels, scores)) if ai.any() and re.any() else float("nan"))
    out["tp"], out["fp"] = int((flagged & ai).sum()), int((flagged & re).sum())
    out["recall"] = out["tp"] / max(out["n_ai"], 1)
    out["far"] = out["fp"] / max(out["n_real"], 1)
    out["recall_ci"] = wilson(out["tp"], out["n_ai"])
    out["far_ci"] = wilson(out["fp"], out["n_real"])
    return out


def bootstrap_ci(values, n_boot=10000, seed=0, alpha=0.05):
    """Percentile bootstrap CI for the mean of a 0/1 (or real) vector."""
    v = np.asarray(values, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), size=(n_boot, len(v)))
    m = v[idx].mean(1)
    return (float(np.quantile(m, alpha / 2)), float(np.quantile(m, 1 - alpha / 2)))
