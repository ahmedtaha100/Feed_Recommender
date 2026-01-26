import time
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
from collections import defaultdict
import logging
import threading

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MetricsCollector:
    def __init__(self, flush_interval: int = 60):
        self.metrics = defaultdict(list)
        self.counters = defaultdict(int)
        self.flush_interval = flush_interval
        self.lock = threading.Lock()

    def record_latency(self, name: str, latency_ms: float) -> None:
        with self.lock:
            self.metrics[f"{name}_latency"].append({
                "value": latency_ms,
                "timestamp": datetime.now().isoformat(),
            })

    def increment_counter(self, name: str, value: int = 1) -> None:
        with self.lock:
            self.counters[name] += value

    def get_stats(self, name: str) -> Dict:
        with self.lock:
            values = [m["value"] for m in self.metrics.get(name, [])]
            if not values:
                return {}
            import numpy as np
            return {
                "count": len(values),
                "mean": float(np.mean(values)),
                "p50": float(np.percentile(values, 50)),
                "p95": float(np.percentile(values, 95)),
                "p99": float(np.percentile(values, 99)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            }

    def get_counters(self) -> Dict[str, int]:
        with self.lock:
            return dict(self.counters)

    def reset(self) -> None:
        with self.lock:
            self.metrics.clear()
            self.counters.clear()


class ModelMonitor:
    def __init__(self, log_dir: str = "logs/monitoring"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.collector = MetricsCollector()
        self.alerts = []
        self.thresholds = {
            "latency_p99_ms": 50.0,
            "error_rate": 0.01,
            "recall_degradation": 0.05,
        }

    def log_prediction(
        self,
        user_id: str,
        recommendations: List[str],
        latency_ms: float,
        cache_hit: bool,
    ) -> None:
        self.collector.record_latency("inference", latency_ms)
        self.collector.increment_counter("total_requests")
        if cache_hit:
            self.collector.increment_counter("cache_hits")

        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "user_id": user_id,
            "num_recommendations": len(recommendations),
            "latency_ms": latency_ms,
            "cache_hit": cache_hit,
        }

        log_file = self.log_dir / f"predictions_{datetime.now().strftime('%Y%m%d')}.jsonl"
        with open(log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

    def log_error(self, error_type: str, message: str) -> None:
        self.collector.increment_counter("errors")
        self.collector.increment_counter(f"error_{error_type}")

        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "error_type": error_type,
            "message": message,
        }

        log_file = self.log_dir / f"errors_{datetime.now().strftime('%Y%m%d')}.jsonl"
        with open(log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

    def check_alerts(self) -> List[Dict]:
        alerts = []
        stats = self.collector.get_stats("inference_latency")
        counters = self.collector.get_counters()

        if stats.get("p99", 0) > self.thresholds["latency_p99_ms"]:
            alerts.append({
                "type": "latency",
                "severity": "warning",
                "message": f"P99 latency {stats['p99']:.2f}ms exceeds threshold {self.thresholds['latency_p99_ms']}ms",
                "timestamp": datetime.now().isoformat(),
            })

        total = counters.get("total_requests", 0)
        errors = counters.get("errors", 0)
        if total > 0 and (errors / total) > self.thresholds["error_rate"]:
            alerts.append({
                "type": "error_rate",
                "severity": "critical",
                "message": f"Error rate {errors/total:.2%} exceeds threshold {self.thresholds['error_rate']:.2%}",
                "timestamp": datetime.now().isoformat(),
            })

        self.alerts.extend(alerts)
        return alerts

    def get_dashboard_metrics(self) -> Dict:
        counters = self.collector.get_counters()
        total = counters.get("total_requests", 0)
        cache_hits = counters.get("cache_hits", 0)
        errors = counters.get("errors", 0)

        return {
            "total_requests": total,
            "cache_hit_rate": cache_hits / total if total > 0 else 0,
            "error_rate": errors / total if total > 0 else 0,
            "latency_stats": self.collector.get_stats("inference_latency"),
            "recent_alerts": self.alerts[-10:],
        }

    def export_metrics(self, output_path: str) -> None:
        metrics = self.get_dashboard_metrics()
        metrics["exported_at"] = datetime.now().isoformat()

        with open(output_path, "w") as f:
            json.dump(metrics, f, indent=2)


class ABTestManager:
    def __init__(self, log_dir: str = "logs/ab_tests"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.experiments = {}

    def create_experiment(
        self,
        name: str,
        control_model: str,
        treatment_model: str,
        traffic_split: float = 0.5,
    ) -> None:
        self.experiments[name] = {
            "control": control_model,
            "treatment": treatment_model,
            "traffic_split": traffic_split,
            "control_metrics": [],
            "treatment_metrics": [],
            "created_at": datetime.now().isoformat(),
        }

    def assign_variant(self, experiment_name: str, user_id: str) -> str:
        if experiment_name not in self.experiments:
            return "control"

        experiment = self.experiments[experiment_name]
        user_hash = hash(user_id) % 100
        if user_hash < experiment["traffic_split"] * 100:
            return "treatment"
        return "control"

    def log_outcome(
        self,
        experiment_name: str,
        variant: str,
        user_id: str,
        clicked: bool,
    ) -> None:
        if experiment_name not in self.experiments:
            return

        experiment = self.experiments[experiment_name]
        metric_key = f"{variant}_metrics"
        experiment[metric_key].append({
            "user_id": user_id,
            "clicked": clicked,
            "timestamp": datetime.now().isoformat(),
        })

    def get_experiment_results(self, experiment_name: str) -> Dict:
        if experiment_name not in self.experiments:
            return {}

        experiment = self.experiments[experiment_name]
        control_clicks = sum(1 for m in experiment["control_metrics"] if m["clicked"])
        treatment_clicks = sum(1 for m in experiment["treatment_metrics"] if m["clicked"])
        control_total = len(experiment["control_metrics"])
        treatment_total = len(experiment["treatment_metrics"])

        return {
            "experiment": experiment_name,
            "control": {
                "model": experiment["control"],
                "impressions": control_total,
                "clicks": control_clicks,
                "ctr": control_clicks / control_total if control_total > 0 else 0,
            },
            "treatment": {
                "model": experiment["treatment"],
                "impressions": treatment_total,
                "clicks": treatment_clicks,
                "ctr": treatment_clicks / treatment_total if treatment_total > 0 else 0,
            },
        }
