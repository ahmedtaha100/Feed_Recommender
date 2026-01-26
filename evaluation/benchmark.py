import json
import time
import sys
from pathlib import Path
from typing import Dict, List, Optional
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import requests

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LatencyBenchmark:
    def __init__(self):
        self.latencies: List[float] = []
        self.errors: int = 0
        self.lock = threading.Lock()

    def add_latency(self, latency_ms: float) -> None:
        with self.lock:
            self.latencies.append(latency_ms)

    def add_error(self) -> None:
        with self.lock:
            self.errors += 1

    def compute_stats(self) -> Dict[str, float]:
        if not self.latencies:
            return {}

        latencies = np.array(self.latencies)
        return {
            "count": len(latencies),
            "errors": self.errors,
            "mean_ms": float(np.mean(latencies)),
            "std_ms": float(np.std(latencies)),
            "min_ms": float(np.min(latencies)),
            "max_ms": float(np.max(latencies)),
            "p50_ms": float(np.percentile(latencies, 50)),
            "p90_ms": float(np.percentile(latencies, 90)),
            "p95_ms": float(np.percentile(latencies, 95)),
            "p99_ms": float(np.percentile(latencies, 99)),
            "throughput_qps": len(latencies) / (sum(latencies) / 1000) if sum(latencies) > 0 else 0,
        }


def benchmark_api(
    base_url: str = "http://localhost:8000",
    num_requests: int = 1000,
    num_workers: int = 10,
    user_ids: Optional[List[str]] = None,
) -> Dict[str, any]:
    benchmark = LatencyBenchmark()

    if user_ids is None:
        user_ids = [f"user_{i}" for i in range(100)]

    def make_request(user_id: str) -> None:
        try:
            start_time = time.perf_counter()
            response = requests.post(
                f"{base_url}/recommend",
                json={
                    "user_id": user_id,
                    "history": [],
                    "num_recommendations": 20,
                },
                timeout=5.0,
            )
            latency_ms = (time.perf_counter() - start_time) * 1000

            if response.status_code == 200:
                benchmark.add_latency(latency_ms)
            else:
                benchmark.add_error()
        except Exception as e:
            benchmark.add_error()

    start_time = time.time()

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for i in range(num_requests):
            user_id = user_ids[i % len(user_ids)]
            futures.append(executor.submit(make_request, user_id))

        for future in as_completed(futures):
            pass

    total_time = time.time() - start_time

    stats = benchmark.compute_stats()
    stats["total_time_s"] = total_time
    stats["actual_qps"] = num_requests / total_time

    return stats


def benchmark_retrieval(
    index_path: str = "artifacts/faiss_index",
    embeddings_path: str = "artifacts/item_embeddings.npy",
    num_queries: int = 10000,
    top_k: int = 100,
    batch_sizes: List[int] = [1, 8, 32, 64],
) -> Dict[str, Dict[str, float]]:
    from retrieval.retriever import Retriever

    retriever = Retriever(
        index_path=index_path,
        item_embeddings_path=embeddings_path,
    )

    results = {}

    query_dim = retriever.index_builder.embedding_dim
    all_queries = np.random.randn(num_queries, query_dim).astype(np.float32)

    for batch_size in batch_sizes:
        latencies = []

        for i in range(0, num_queries, batch_size):
            batch_queries = all_queries[i:min(i + batch_size, num_queries)]

            start_time = time.perf_counter()
            retriever.retrieve(batch_queries, top_k=top_k)
            latency_ms = (time.perf_counter() - start_time) * 1000

            latencies.append(latency_ms)

        latencies = np.array(latencies)
        per_query_latencies = latencies / batch_size

        results[f"batch_{batch_size}"] = {
            "batch_mean_ms": float(np.mean(latencies)),
            "batch_p99_ms": float(np.percentile(latencies, 99)),
            "per_query_mean_ms": float(np.mean(per_query_latencies)),
            "per_query_p99_ms": float(np.percentile(per_query_latencies, 99)),
            "throughput_qps": 1000 / np.mean(per_query_latencies),
        }

    return results


def benchmark_model_inference(
    model_path: str = "artifacts/two_tower_model.pt",
    config_path: str = "artifacts/model_config.json",
    num_samples: int = 1000,
    batch_sizes: List[int] = [1, 8, 32, 64],
) -> Dict[str, Dict[str, float]]:
    import torch
    from models.two_tower import create_model_from_config
    from training.trainer import get_device

    device = get_device()

    with open(config_path, "r") as f:
        config = json.load(f)

    model = create_model_from_config(config)
    from training.trainer import load_checkpoint_safe
    checkpoint = load_checkpoint_safe(model_path, device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()

    results = {}

    max_history = config["data"]["max_history_len"]
    max_title = config["data"]["max_title_len"]

    for batch_size in batch_sizes:
        latencies = []

        for _ in range(num_samples // batch_size):
            history_title = torch.randint(0, 1000, (batch_size, max_history, max_title)).to(device)
            history_category = torch.randint(0, 20, (batch_size, max_history)).to(device)
            history_mask = torch.ones(batch_size, max_history).to(device)

            if device.type == "cuda":
                torch.cuda.synchronize()

            start_time = time.perf_counter()

            with torch.no_grad():
                _ = model.encode_user(
                    history_title=history_title,
                    history_category=history_category,
                    history_mask=history_mask,
                )

            if device.type == "cuda":
                torch.cuda.synchronize()

            latency_ms = (time.perf_counter() - start_time) * 1000
            latencies.append(latency_ms)

        latencies = np.array(latencies)
        per_sample_latencies = latencies / batch_size

        results[f"batch_{batch_size}"] = {
            "batch_mean_ms": float(np.mean(latencies)),
            "batch_p99_ms": float(np.percentile(latencies, 99)),
            "per_sample_mean_ms": float(np.mean(per_sample_latencies)),
            "per_sample_p99_ms": float(np.percentile(per_sample_latencies, 99)),
            "throughput_samples_per_sec": 1000 / np.mean(per_sample_latencies),
        }

    return results


def run_full_benchmark(
    output_path: str = "artifacts/benchmark_results.json",
) -> Dict:
    results = {}

    logger.info("Benchmarking retrieval...")
    try:
        results["retrieval"] = benchmark_retrieval()
        logger.info(f"Retrieval benchmark complete")
    except Exception as e:
        logger.error(f"Retrieval benchmark failed: {e}")
        results["retrieval"] = {"error": str(e)}

    logger.info("Benchmarking model inference...")
    try:
        results["model_inference"] = benchmark_model_inference()
        logger.info(f"Model inference benchmark complete")
    except Exception as e:
        logger.error(f"Model inference benchmark failed: {e}")
        results["model_inference"] = {"error": str(e)}

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Benchmark results saved to {output_path}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--api-url", type=str, default="http://localhost:8000")
    parser.add_argument("--num-requests", type=int, default=1000)
    parser.add_argument("--num-workers", type=int, default=10)
    parser.add_argument("--output", type=str, default="artifacts/benchmark_results.json")
    parser.add_argument("--api-only", action="store_true")
    args = parser.parse_args()

    if args.api_only:
        logger.info("Benchmarking API...")
        results = benchmark_api(
            base_url=args.api_url,
            num_requests=args.num_requests,
            num_workers=args.num_workers,
        )
        print("\nAPI Benchmark Results:")
        print(json.dumps(results, indent=2))
    else:
        results = run_full_benchmark(output_path=args.output)
        print("\nBenchmark Results:")
        print(json.dumps(results, indent=2))
