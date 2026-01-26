import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import logging

import numpy as np
from scipy import stats

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def calculate_psi(
    expected: np.ndarray,
    actual: np.ndarray,
    buckets: int = 10,
) -> float:
    breakpoints = np.linspace(0, 100, buckets + 1)

    expected_percents = np.percentile(expected, breakpoints)
    actual_percents = np.percentile(actual, breakpoints)

    def get_bucket_counts(data, percentiles):
        counts = np.zeros(len(percentiles) - 1)
        for i in range(len(percentiles) - 1):
            if i == len(percentiles) - 2:
                counts[i] = np.sum((data >= percentiles[i]) & (data <= percentiles[i + 1]))
            else:
                counts[i] = np.sum((data >= percentiles[i]) & (data < percentiles[i + 1]))
        return counts / len(data)

    expected_counts = get_bucket_counts(expected, expected_percents)
    actual_counts = get_bucket_counts(actual, actual_percents)

    expected_counts = np.clip(expected_counts, 0.0001, 1)
    actual_counts = np.clip(actual_counts, 0.0001, 1)

    psi = np.sum((actual_counts - expected_counts) * np.log(actual_counts / expected_counts))

    return float(psi)


def calculate_ks_statistic(
    expected: np.ndarray,
    actual: np.ndarray,
) -> Tuple[float, float]:
    statistic, p_value = stats.ks_2samp(expected, actual)
    return float(statistic), float(p_value)


def calculate_js_divergence(
    expected: np.ndarray,
    actual: np.ndarray,
    bins: int = 50,
) -> float:
    min_val = min(expected.min(), actual.min())
    max_val = max(expected.max(), actual.max())

    hist_exp, _ = np.histogram(expected, bins=bins, range=(min_val, max_val), density=True)
    hist_act, _ = np.histogram(actual, bins=bins, range=(min_val, max_val), density=True)

    hist_exp = hist_exp + 1e-10
    hist_act = hist_act + 1e-10

    hist_exp = hist_exp / hist_exp.sum()
    hist_act = hist_act / hist_act.sum()

    m = 0.5 * (hist_exp + hist_act)

    js_div = 0.5 * stats.entropy(hist_exp, m) + 0.5 * stats.entropy(hist_act, m)

    return float(js_div)


class DriftDetector:
    def __init__(
        self,
        reference_data: Optional[Dict[str, np.ndarray]] = None,
        psi_threshold: float = 0.2,
        ks_threshold: float = 0.05,
        js_threshold: float = 0.1,
    ):
        self.reference_data = reference_data or {}
        self.psi_threshold = psi_threshold
        self.ks_threshold = ks_threshold
        self.js_threshold = js_threshold
        self.history: List[Dict] = []

    def set_reference(self, feature_name: str, data: np.ndarray) -> None:
        self.reference_data[feature_name] = data.flatten()

    def detect_drift(
        self,
        feature_name: str,
        current_data: np.ndarray,
    ) -> Dict:
        current_data = current_data.flatten()

        if feature_name not in self.reference_data:
            return {
                "feature": feature_name,
                "error": "No reference data available",
                "drift_detected": False,
            }

        reference = self.reference_data[feature_name]

        psi = calculate_psi(reference, current_data)
        ks_stat, ks_pvalue = calculate_ks_statistic(reference, current_data)
        js_div = calculate_js_divergence(reference, current_data)

        drift_detected = (
            psi > self.psi_threshold or
            ks_pvalue < self.ks_threshold or
            js_div > self.js_threshold
        )

        result = {
            "feature": feature_name,
            "timestamp": datetime.now().isoformat(),
            "metrics": {
                "psi": psi,
                "ks_statistic": ks_stat,
                "ks_pvalue": ks_pvalue,
                "js_divergence": js_div,
            },
            "thresholds": {
                "psi": self.psi_threshold,
                "ks_pvalue": self.ks_threshold,
                "js_divergence": self.js_threshold,
            },
            "drift_detected": drift_detected,
            "reference_stats": {
                "mean": float(np.mean(reference)),
                "std": float(np.std(reference)),
                "min": float(np.min(reference)),
                "max": float(np.max(reference)),
            },
            "current_stats": {
                "mean": float(np.mean(current_data)),
                "std": float(np.std(current_data)),
                "min": float(np.min(current_data)),
                "max": float(np.max(current_data)),
            },
        }

        self.history.append(result)

        return result

    def detect_all_drift(
        self,
        current_data: Dict[str, np.ndarray],
    ) -> Dict:
        results = {}
        any_drift = False

        for feature_name, data in current_data.items():
            result = self.detect_drift(feature_name, data)
            results[feature_name] = result
            if result.get("drift_detected", False):
                any_drift = True

        return {
            "timestamp": datetime.now().isoformat(),
            "any_drift_detected": any_drift,
            "feature_results": results,
        }

    def save_reference(self, path: str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        save_data = {
            k: v.tolist() for k, v in self.reference_data.items()
        }

        with open(path, "w") as f:
            json.dump(save_data, f)

    def load_reference(self, path: str) -> None:
        with open(path, "r") as f:
            data = json.load(f)

        self.reference_data = {
            k: np.array(v) for k, v in data.items()
        }

    def get_history(self) -> List[Dict]:
        return self.history

    def clear_history(self) -> None:
        self.history = []


def compute_feature_distributions(
    samples: List[Dict],
    news_data: Dict,
) -> Dict[str, np.ndarray]:
    click_rates = []
    category_counts = {}
    history_lengths = []

    for sample in samples:
        if "label" in sample:
            click_rates.append(sample["label"])

        if "history" in sample:
            history_lengths.append(sum(1 for h in sample["history"] if h > 0))

        if "news_idx" in sample and sample["news_idx"] in news_data:
            cat = news_data[sample["news_idx"]].get("category", 0)
            category_counts[cat] = category_counts.get(cat, 0) + 1

    features = {}

    if click_rates:
        features["click_rate"] = np.array(click_rates)

    if history_lengths:
        features["history_length"] = np.array(history_lengths)

    if category_counts and len(category_counts) > 0:
        total = sum(category_counts.values())
        if total > 0:
            cat_dist = np.array([category_counts.get(i, 0) / total for i in range(max(category_counts.keys()) + 1)])
            features["category_distribution"] = cat_dist

    return features


if __name__ == "__main__":
    np.random.seed(42)

    reference_data = np.random.normal(0, 1, 10000)

    no_drift_data = np.random.normal(0, 1, 10000)
    drift_data = np.random.normal(0.5, 1.2, 10000)

    detector = DriftDetector()
    detector.set_reference("test_feature", reference_data)

    print("No drift case:")
    result = detector.detect_drift("test_feature", no_drift_data)
    print(f"  PSI: {result['metrics']['psi']:.4f}")
    print(f"  Drift detected: {result['drift_detected']}")

    print("\nDrift case:")
    result = detector.detect_drift("test_feature", drift_data)
    print(f"  PSI: {result['metrics']['psi']:.4f}")
    print(f"  Drift detected: {result['drift_detected']}")
