import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

import numpy as np
import torch

from .index_builder import FAISSIndexBuilder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Retriever:
    def __init__(
        self,
        index_path: str,
        item_embeddings_path: Optional[str] = None,
    ):
        self.index_builder = FAISSIndexBuilder()
        self.index_builder.load(index_path)
        self.item_embeddings: Optional[np.ndarray] = None

        if item_embeddings_path:
            self.item_embeddings = np.load(item_embeddings_path)

    def retrieve(
        self,
        user_embeddings: np.ndarray,
        top_k: int = 100,
        exclude_ids: Optional[List[List[int]]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if user_embeddings.ndim == 1:
            user_embeddings = user_embeddings.reshape(1, -1)

        fetch_k = top_k if exclude_ids is None else top_k * 2

        scores, item_ids = self.index_builder.search(user_embeddings, fetch_k)

        if exclude_ids is not None:
            filtered_scores = []
            filtered_ids = []

            for i, (s_row, id_row) in enumerate(zip(scores, item_ids)):
                exclude_set = set(exclude_ids[i]) if i < len(exclude_ids) else set()
                mask = np.array([iid not in exclude_set for iid in id_row])
                filtered_s = s_row[mask][:top_k]
                filtered_id = id_row[mask][:top_k]

                pad_len = top_k - len(filtered_s)
                if pad_len > 0:
                    filtered_s = np.pad(filtered_s, (0, pad_len), constant_values=-1e9)
                    filtered_id = np.pad(filtered_id, (0, pad_len), constant_values=-1)

                filtered_scores.append(filtered_s)
                filtered_ids.append(filtered_id)

            scores = np.array(filtered_scores)
            item_ids = np.array(filtered_ids)

        return scores, item_ids

    def get_item_embeddings(self, item_ids: List[int]) -> np.ndarray:
        if self.item_embeddings is None:
            raise ValueError("Item embeddings not loaded")

        embeddings = []
        for item_id in item_ids:
            if 0 <= item_id < len(self.item_embeddings):
                embeddings.append(self.item_embeddings[item_id])
            else:
                embeddings.append(np.zeros(self.item_embeddings.shape[1]))

        return np.array(embeddings)

    def benchmark_latency(
        self,
        num_queries: int = 1000,
        top_k: int = 100,
        batch_sizes: List[int] = [1, 8, 32],
    ) -> Dict[str, Dict[str, float]]:
        results = {}

        query_dim = self.index_builder.embedding_dim
        all_queries = np.random.randn(num_queries, query_dim).astype(np.float32)

        for batch_size in batch_sizes:
            latencies = []

            for i in range(0, num_queries, batch_size):
                batch_queries = all_queries[i:i+batch_size]

                start_time = time.perf_counter()
                self.retrieve(batch_queries, top_k)
                end_time = time.perf_counter()

                latency_ms = (end_time - start_time) * 1000
                latencies.append(latency_ms)

            latencies = np.array(latencies)
            results[f"batch_{batch_size}"] = {
                "mean_ms": float(np.mean(latencies)),
                "p50_ms": float(np.percentile(latencies, 50)),
                "p95_ms": float(np.percentile(latencies, 95)),
                "p99_ms": float(np.percentile(latencies, 99)),
                "min_ms": float(np.min(latencies)),
                "max_ms": float(np.max(latencies)),
            }

        return results


class TwoStageRetriever:
    def __init__(
        self,
        retriever: Retriever,
        reranker,
        retrieval_top_k: int = 100,
        final_top_k: int = 20,
    ):
        self.retriever = retriever
        self.reranker = reranker
        self.retrieval_top_k = retrieval_top_k
        self.final_top_k = final_top_k

    def retrieve_and_rerank(
        self,
        user_embedding: np.ndarray,
        exclude_ids: Optional[List[int]] = None,
    ) -> List[Tuple[int, float]]:
        if user_embedding.ndim == 1:
            user_embedding = user_embedding.reshape(1, -1)

        exclude_list = [exclude_ids] if exclude_ids else None

        scores, candidate_ids = self.retriever.retrieve(
            user_embedding,
            top_k=self.retrieval_top_k,
            exclude_ids=exclude_list,
        )

        candidate_ids = candidate_ids[0].tolist()
        candidate_ids = [cid for cid in candidate_ids if cid >= 0]

        if not candidate_ids:
            return []

        candidate_embeddings = self.retriever.get_item_embeddings(candidate_ids)

        reranked = self.reranker.rerank(
            user_emb=user_embedding[0],
            candidate_emb=candidate_embeddings,
            candidate_ids=candidate_ids,
            top_k=self.final_top_k,
        )

        return reranked

    def batch_retrieve_and_rerank(
        self,
        user_embeddings: np.ndarray,
        exclude_ids: Optional[List[List[int]]] = None,
    ) -> List[List[Tuple[int, float]]]:
        results = []
        for i, user_emb in enumerate(user_embeddings):
            exclude = exclude_ids[i] if exclude_ids else None
            result = self.retrieve_and_rerank(user_emb, exclude)
            results.append(result)
        return results
