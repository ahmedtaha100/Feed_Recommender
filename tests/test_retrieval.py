import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import numpy as np
import tempfile
import os

from retrieval.index_builder import FAISSIndexBuilder


class TestFAISSIndexBuilder:
    def test_build_flat_index(self):
        n_items = 1000
        dim = 128

        builder = FAISSIndexBuilder(embedding_dim=dim, index_type="Flat")
        embeddings = np.random.randn(n_items, dim).astype(np.float32)

        builder.build_index(embeddings)

        assert builder.index is not None
        assert builder.index.ntotal == n_items

    def test_build_ivf_index(self):
        n_items = 1000
        dim = 128

        builder = FAISSIndexBuilder(
            embedding_dim=dim,
            index_type="IVF",
            nlist=10,
            nprobe=5,
        )
        embeddings = np.random.randn(n_items, dim).astype(np.float32)

        builder.build_index(embeddings)

        assert builder.index is not None
        assert builder.index.ntotal == n_items

    def test_search(self):
        n_items = 1000
        dim = 64
        top_k = 10

        builder = FAISSIndexBuilder(embedding_dim=dim, index_type="Flat")
        embeddings = np.random.randn(n_items, dim).astype(np.float32)
        builder.build_index(embeddings)

        query = np.random.randn(1, dim).astype(np.float32)
        scores, ids = builder.search(query, top_k=top_k)

        assert scores.shape == (1, top_k)
        assert ids.shape == (1, top_k)

    def test_batch_search(self):
        n_items = 500
        dim = 64
        n_queries = 10
        top_k = 20

        builder = FAISSIndexBuilder(embedding_dim=dim, index_type="Flat")
        embeddings = np.random.randn(n_items, dim).astype(np.float32)
        builder.build_index(embeddings)

        queries = np.random.randn(n_queries, dim).astype(np.float32)
        scores, ids = builder.search(queries, top_k=top_k)

        assert scores.shape == (n_queries, top_k)
        assert ids.shape == (n_queries, top_k)

    def test_id_mapping(self):
        n_items = 100
        dim = 32

        builder = FAISSIndexBuilder(embedding_dim=dim, index_type="Flat")
        embeddings = np.random.randn(n_items, dim).astype(np.float32)
        item_ids = list(range(1000, 1000 + n_items))

        builder.build_index(embeddings, item_ids=item_ids)

        query = np.random.randn(1, dim).astype(np.float32)
        _, ids = builder.search(query, top_k=5)

        for id in ids[0]:
            assert 1000 <= id < 1000 + n_items

    def test_save_load(self):
        n_items = 200
        dim = 64

        builder = FAISSIndexBuilder(embedding_dim=dim, index_type="Flat")
        embeddings = np.random.randn(n_items, dim).astype(np.float32)
        builder.build_index(embeddings)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test_index")
            builder.save(path)

            new_builder = FAISSIndexBuilder()
            new_builder.load(path)

            assert new_builder.index.ntotal == n_items

            query = np.random.randn(1, dim).astype(np.float32)
            scores1, ids1 = builder.search(query, top_k=5)
            scores2, ids2 = new_builder.search(query, top_k=5)

            np.testing.assert_array_equal(ids1, ids2)
            np.testing.assert_array_almost_equal(scores1, scores2, decimal=5)

    def test_search_returns_sorted(self):
        n_items = 500
        dim = 32
        top_k = 50

        builder = FAISSIndexBuilder(embedding_dim=dim, index_type="Flat")
        embeddings = np.random.randn(n_items, dim).astype(np.float32)
        builder.build_index(embeddings)

        query = np.random.randn(1, dim).astype(np.float32)
        scores, _ = builder.search(query, top_k=top_k)

        for i in range(top_k - 1):
            assert scores[0, i] >= scores[0, i + 1]

    def test_hnsw_index(self):
        n_items = 500
        dim = 64

        builder = FAISSIndexBuilder(embedding_dim=dim, index_type="HNSW")
        embeddings = np.random.randn(n_items, dim).astype(np.float32)

        builder.build_index(embeddings)

        query = np.random.randn(1, dim).astype(np.float32)
        scores, ids = builder.search(query, top_k=10)

        assert scores.shape == (1, 10)
        assert ids.shape == (1, 10)


class TestRetrieverIntegration:
    def test_latency(self):
        import time

        n_items = 10000
        dim = 128
        n_queries = 100

        builder = FAISSIndexBuilder(
            embedding_dim=dim,
            index_type="IVF",
            nlist=100,
            nprobe=10,
        )
        embeddings = np.random.randn(n_items, dim).astype(np.float32)
        builder.build_index(embeddings)

        queries = np.random.randn(n_queries, dim).astype(np.float32)

        start = time.perf_counter()
        for i in range(n_queries):
            builder.search(queries[i:i+1], top_k=100)
        elapsed_ms = (time.perf_counter() - start) * 1000

        avg_latency = elapsed_ms / n_queries
        assert avg_latency < 15, f"Average latency {avg_latency:.2f}ms exceeds 15ms target"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
