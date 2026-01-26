import json
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def recall_at_k(predictions: np.ndarray, ground_truth: np.ndarray, k: int) -> float:
    if len(predictions) == 0 or len(ground_truth) == 0:
        return 0.0

    top_k = set(predictions[:k])
    relevant = set(ground_truth)

    return len(top_k & relevant) / len(relevant)


def precision_at_k(predictions: np.ndarray, ground_truth: np.ndarray, k: int) -> float:
    if len(predictions) == 0:
        return 0.0

    top_k = set(predictions[:k])
    relevant = set(ground_truth)

    return len(top_k & relevant) / k


def ndcg_at_k(predictions: np.ndarray, ground_truth: np.ndarray, k: int) -> float:
    if len(predictions) == 0 or len(ground_truth) == 0:
        return 0.0

    relevant_set = set(ground_truth)
    relevance = np.array([1.0 if p in relevant_set else 0.0 for p in predictions[:k]])

    dcg = np.sum(relevance / np.log2(np.arange(2, len(relevance) + 2)))

    ideal_relevance = np.zeros(min(k, len(ground_truth)))
    ideal_relevance[:] = 1.0
    idcg = np.sum(ideal_relevance / np.log2(np.arange(2, len(ideal_relevance) + 2)))

    if idcg == 0:
        return 0.0

    return dcg / idcg


def mrr(predictions: np.ndarray, ground_truth: np.ndarray) -> float:
    if len(predictions) == 0 or len(ground_truth) == 0:
        return 0.0

    relevant_set = set(ground_truth)

    for i, p in enumerate(predictions):
        if p in relevant_set:
            return 1.0 / (i + 1)

    return 0.0


def hit_rate(predictions: np.ndarray, ground_truth: np.ndarray, k: int) -> float:
    if len(predictions) == 0 or len(ground_truth) == 0:
        return 0.0

    top_k = set(predictions[:k])
    relevant = set(ground_truth)

    return 1.0 if len(top_k & relevant) > 0 else 0.0


def map_at_k(predictions: np.ndarray, ground_truth: np.ndarray, k: int) -> float:
    if len(predictions) == 0 or len(ground_truth) == 0:
        return 0.0

    relevant_set = set(ground_truth)
    num_relevant = 0
    precision_sum = 0.0

    for i, p in enumerate(predictions[:k]):
        if p in relevant_set:
            num_relevant += 1
            precision_sum += num_relevant / (i + 1)

    if num_relevant == 0:
        return 0.0

    return precision_sum / min(len(ground_truth), k)


class MetricsCalculator:
    def __init__(self, ks: List[int] = [10, 50, 100]):
        self.ks = ks
        self.results = {
            f"recall@{k}": [] for k in ks
        }
        self.results.update({
            f"precision@{k}": [] for k in ks
        })
        self.results.update({
            f"ndcg@{k}": [] for k in ks
        })
        self.results.update({
            f"hit_rate@{k}": [] for k in ks
        })
        self.results["mrr"] = []
        self.results.update({
            f"map@{k}": [] for k in ks
        })

    def add_sample(self, predictions: np.ndarray, ground_truth: np.ndarray) -> None:
        for k in self.ks:
            self.results[f"recall@{k}"].append(recall_at_k(predictions, ground_truth, k))
            self.results[f"precision@{k}"].append(precision_at_k(predictions, ground_truth, k))
            self.results[f"ndcg@{k}"].append(ndcg_at_k(predictions, ground_truth, k))
            self.results[f"hit_rate@{k}"].append(hit_rate(predictions, ground_truth, k))
            self.results[f"map@{k}"].append(map_at_k(predictions, ground_truth, k))

        self.results["mrr"].append(mrr(predictions, ground_truth))

    def compute(self) -> Dict[str, float]:
        return {
            metric: float(np.mean(values)) if values else 0.0
            for metric, values in self.results.items()
        }

    def reset(self) -> None:
        for key in self.results:
            self.results[key] = []


def evaluate_retrieval(
    model_path: str,
    index_path: str,
    data_path: str,
    top_k: List[int] = [10, 50, 100],
    num_samples: int = 1000,
    batch_size: int = 64,
) -> Dict[str, float]:
    from models.two_tower import create_model_from_config
    from retrieval.retriever import Retriever
    from training.trainer import get_device

    device = get_device()

    artifacts_dir = Path(model_path).parent
    with open(artifacts_dir / "model_config.json", "r") as f:
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

    retriever = Retriever(
        index_path=index_path,
        item_embeddings_path=str(artifacts_dir / "item_embeddings.npy"),
    )

    data_dir = Path(data_path)
    with open(data_dir / "dev_samples.pkl", "rb") as f:
        samples = pickle.load(f)

    with open(data_dir / "news_data.pkl", "rb") as f:
        news_data = pickle.load(f)

    with open(data_dir / "mappings.json", "r") as f:
        mappings = json.load(f)

    idx2news = {v: k for k, v in mappings["news2idx"].items()}

    impressions = {}
    for sample in samples:
        imp_id = sample.get("impression_id", 0)
        if imp_id not in impressions:
            impressions[imp_id] = {
                "user_idx": sample["user_idx"],
                "history": sample["history"],
                "items": [],
                "labels": [],
            }
        impressions[imp_id]["items"].append(sample["news_idx"])
        impressions[imp_id]["labels"].append(sample.get("label", 0))

    impression_list = list(impressions.values())[:num_samples]

    calculator = MetricsCalculator(ks=top_k)

    max_history = config["data"]["max_history_len"]

    for imp in tqdm(impression_list, desc="Evaluating"):
        history = imp["history"]
        ground_truth = [idx for idx, label in zip(imp["items"], imp["labels"]) if label == 1]

        if not ground_truth:
            continue

        history_titles = []
        history_categories = []

        for idx in history:
            if idx > 0 and idx in idx2news:
                news_id = idx2news[idx]
                news = news_data.get(news_id, {})
                history_titles.append(news.get("title", [0] * 30))
                history_categories.append(news.get("category", 0))
            else:
                history_titles.append([0] * 30)
                history_categories.append(0)

        history_title = torch.tensor([history_titles], dtype=torch.long).to(device)
        history_category = torch.tensor([history_categories], dtype=torch.long).to(device)
        history_mask = torch.tensor(
            [[1.0 if idx > 0 else 0.0 for idx in history]], dtype=torch.float
        ).to(device)

        with torch.no_grad():
            user_emb = model.encode_user(
                history_title=history_title,
                history_category=history_category,
                history_mask=history_mask,
            ).cpu().numpy()

        _, predictions = retriever.retrieve(user_emb, top_k=max(top_k))
        predictions = predictions[0]

        calculator.add_sample(predictions, np.array(ground_truth))

    results = calculator.compute()

    logger.info("Evaluation Results:")
    for metric, value in results.items():
        logger.info(f"  {metric}: {value:.4f}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="artifacts/two_tower_model.pt")
    parser.add_argument("--index", type=str, default="artifacts/faiss_index")
    parser.add_argument("--data", type=str, default="data/processed")
    parser.add_argument("--num-samples", type=int, default=1000)
    args = parser.parse_args()

    results = evaluate_retrieval(
        model_path=args.model,
        index_path=args.index,
        data_path=args.data,
        num_samples=args.num_samples,
    )

    print("\nResults:")
    print(json.dumps(results, indent=2))
