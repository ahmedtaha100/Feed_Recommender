import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import numpy as np

from serving.schemas import RecommendRequest, RecommendResponse, RecommendationItem
from serving.cache import RedisCache
from evaluation.metrics import recall_at_k, ndcg_at_k, mrr, precision_at_k


class TestSchemas:
    def test_recommend_request(self):
        request = RecommendRequest(
            user_id="user123",
            history=["news1", "news2", "news3"],
            num_recommendations=20,
        )

        assert request.user_id == "user123"
        assert len(request.history) == 3
        assert request.num_recommendations == 20

    def test_recommend_request_defaults(self):
        request = RecommendRequest(user_id="user456")

        assert request.user_id == "user456"
        assert request.history is None
        assert request.num_recommendations == 20

    def test_recommendation_item(self):
        item = RecommendationItem(
            item_id="news123",
            score=0.95,
            rank=1,
        )

        assert item.item_id == "news123"
        assert item.score == 0.95
        assert item.rank == 1

    def test_recommend_response(self):
        recommendations = [
            RecommendationItem(item_id="news1", score=0.9, rank=1),
            RecommendationItem(item_id="news2", score=0.8, rank=2),
        ]

        response = RecommendResponse(
            user_id="user123",
            recommendations=recommendations,
            latency_ms=5.5,
        )

        assert response.user_id == "user123"
        assert len(response.recommendations) == 2
        assert response.latency_ms == 5.5


class TestRedisCache:
    def test_memory_fallback(self):
        cache = RedisCache(host="nonexistent", port=9999)
        connected = cache.connect()

        assert not connected
        assert not cache.is_connected

    def test_set_get_embedding(self):
        cache = RedisCache()
        cache.connect()

        user_id = "test_user"
        embedding = np.random.randn(128).astype(np.float32)

        cache.set_user_embedding(user_id, embedding)
        retrieved = cache.get_user_embedding(user_id)

        assert retrieved is not None
        np.testing.assert_array_almost_equal(embedding, retrieved)

    def test_get_nonexistent(self):
        cache = RedisCache()
        cache.connect()

        result = cache.get_user_embedding("nonexistent_user")

        assert result is None

    def test_batch_operations(self):
        cache = RedisCache()
        cache.connect()

        embeddings = {
            f"user_{i}": np.random.randn(64).astype(np.float32)
            for i in range(5)
        }

        cache.set_batch_embeddings(embeddings)

        retrieved = cache.get_batch_embeddings(list(embeddings.keys()))

        for user_id, emb in embeddings.items():
            assert user_id in retrieved
            np.testing.assert_array_almost_equal(emb, retrieved[user_id])

    def test_history_operations(self):
        cache = RedisCache()
        cache.connect()

        user_id = "hist_user"
        history = ["news1", "news2", "news3"]

        cache.set_user_history(user_id, history)
        retrieved = cache.get_user_history(user_id)

        assert retrieved == history


class TestMetrics:
    def test_recall_at_k(self):
        predictions = np.array([1, 2, 3, 4, 5])
        ground_truth = np.array([1, 3, 6])

        result = recall_at_k(predictions, ground_truth, k=5)

        assert result == 2 / 3

    def test_recall_at_k_all_relevant(self):
        predictions = np.array([1, 2, 3, 4, 5])
        ground_truth = np.array([1, 2])

        result = recall_at_k(predictions, ground_truth, k=5)

        assert result == 1.0

    def test_precision_at_k(self):
        predictions = np.array([1, 2, 3, 4, 5])
        ground_truth = np.array([1, 3])

        result = precision_at_k(predictions, ground_truth, k=5)

        assert result == 2 / 5

    def test_ndcg_at_k_perfect(self):
        predictions = np.array([1, 2, 3, 4, 5])
        ground_truth = np.array([1, 2])

        result = ndcg_at_k(predictions, ground_truth, k=5)

        assert result == 1.0

    def test_ndcg_at_k_partial(self):
        predictions = np.array([5, 1, 3, 2, 4])
        ground_truth = np.array([1, 2])

        result = ndcg_at_k(predictions, ground_truth, k=5)

        assert 0 < result < 1

    def test_mrr_first(self):
        predictions = np.array([1, 2, 3, 4, 5])
        ground_truth = np.array([1])

        result = mrr(predictions, ground_truth)

        assert result == 1.0

    def test_mrr_second(self):
        predictions = np.array([5, 1, 3, 4, 2])
        ground_truth = np.array([1])

        result = mrr(predictions, ground_truth)

        assert result == 0.5

    def test_mrr_not_found(self):
        predictions = np.array([1, 2, 3, 4, 5])
        ground_truth = np.array([10])

        result = mrr(predictions, ground_truth)

        assert result == 0.0

    def test_empty_inputs(self):
        assert recall_at_k(np.array([]), np.array([1, 2]), k=5) == 0.0
        assert recall_at_k(np.array([1, 2]), np.array([]), k=5) == 0.0
        assert ndcg_at_k(np.array([]), np.array([1]), k=5) == 0.0
        assert mrr(np.array([]), np.array([1])) == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
