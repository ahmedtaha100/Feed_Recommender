import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

import numpy as np
import faiss

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FAISSIndexBuilder:
    def __init__(
        self,
        embedding_dim: int = 128,
        index_type: str = "IVF",
        nlist: int = 100,
        nprobe: int = 10,
        use_gpu: bool = False,
    ):
        self.embedding_dim = embedding_dim
        self.index_type = index_type
        self.nlist = nlist
        self.nprobe = nprobe
        self.use_gpu = use_gpu
        self.index: Optional[faiss.Index] = None
        self.id_mapping: Dict[int, int] = {}
        self.reverse_mapping: Dict[int, int] = {}

    def build_index(
        self,
        embeddings: np.ndarray,
        item_ids: Optional[List[int]] = None,
    ) -> faiss.Index:
        n_samples, dim = embeddings.shape
        assert dim == self.embedding_dim, f"Expected dim {self.embedding_dim}, got {dim}"

        embeddings = embeddings.astype(np.float32)
        faiss.normalize_L2(embeddings)

        if item_ids is None:
            item_ids = list(range(n_samples))

        for faiss_idx, item_id in enumerate(item_ids):
            self.id_mapping[faiss_idx] = item_id
            self.reverse_mapping[item_id] = faiss_idx

        logger.info(f"Building {self.index_type} index for {n_samples} items...")
        start_time = time.time()

        if self.index_type == "Flat":
            self.index = faiss.IndexFlatIP(self.embedding_dim)
        elif self.index_type == "IVF":
            quantizer = faiss.IndexFlatIP(self.embedding_dim)
            self.index = faiss.IndexIVFFlat(
                quantizer, self.embedding_dim, self.nlist, faiss.METRIC_INNER_PRODUCT
            )
            self.index.train(embeddings)
            self.index.nprobe = self.nprobe
        elif self.index_type == "IVFPQ":
            quantizer = faiss.IndexFlatIP(self.embedding_dim)
            self.index = faiss.IndexIVFPQ(
                quantizer, self.embedding_dim, self.nlist, 8, 8, faiss.METRIC_INNER_PRODUCT
            )
            self.index.train(embeddings)
            self.index.nprobe = self.nprobe
        elif self.index_type == "HNSW":
            self.index = faiss.IndexHNSWFlat(self.embedding_dim, 32, faiss.METRIC_INNER_PRODUCT)
            self.index.hnsw.efConstruction = 200
            self.index.hnsw.efSearch = 128
        else:
            raise ValueError(f"Unknown index type: {self.index_type}")

        self.index.add(embeddings)

        build_time = time.time() - start_time
        logger.info(f"Index built in {build_time:.2f}s. Total items: {self.index.ntotal}")

        return self.index

    def search(
        self,
        query_embeddings: np.ndarray,
        top_k: int = 100,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if self.index is None:
            raise ValueError("Index not built. Call build_index first.")

        query_embeddings = query_embeddings.astype(np.float32)
        faiss.normalize_L2(query_embeddings)

        scores, indices = self.index.search(query_embeddings, top_k)

        item_ids = np.array([
            [self.id_mapping.get(idx, -1) for idx in row]
            for row in indices
        ])

        return scores, item_ids

    def save(self, path: str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(path))

        meta_path = path.with_suffix(".meta.json")
        with open(meta_path, "w") as f:
            json.dump({
                "embedding_dim": self.embedding_dim,
                "index_type": self.index_type,
                "nlist": self.nlist,
                "nprobe": self.nprobe,
                "id_mapping": {str(k): v for k, v in self.id_mapping.items()},
                "reverse_mapping": {str(k): v for k, v in self.reverse_mapping.items()},
            }, f)

        logger.info(f"Index saved to {path}")

    def load(self, path: str) -> None:
        path = Path(path)

        self.index = faiss.read_index(str(path))

        meta_path = path.with_suffix(".meta.json")
        if meta_path.exists():
            with open(meta_path, "r") as f:
                meta = json.load(f)
                self.embedding_dim = meta["embedding_dim"]
                self.index_type = meta["index_type"]
                self.nlist = meta["nlist"]
                self.nprobe = meta["nprobe"]
                self.id_mapping = {int(k): v for k, v in meta["id_mapping"].items()}
                self.reverse_mapping = {int(k): v for k, v in meta["reverse_mapping"].items()}

        if hasattr(self.index, 'nprobe'):
            self.index.nprobe = self.nprobe

        logger.info(f"Index loaded from {path}. Total items: {self.index.ntotal}")


def build_index_from_embeddings(
    embeddings_path: str,
    idx_mapping_path: str,
    output_path: str,
    index_type: str = "IVF",
    nlist: int = 100,
    nprobe: int = 10,
) -> FAISSIndexBuilder:
    embeddings = np.load(embeddings_path)

    with open(idx_mapping_path, "r") as f:
        idx_mapping = json.load(f)

    item_ids = list(range(len(embeddings)))

    builder = FAISSIndexBuilder(
        embedding_dim=embeddings.shape[1],
        index_type=index_type,
        nlist=min(nlist, len(embeddings) // 10),
        nprobe=nprobe,
    )

    builder.build_index(embeddings, item_ids)
    builder.save(output_path)

    return builder


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings", type=str, default="artifacts/item_embeddings.npy")
    parser.add_argument("--idx-mapping", type=str, default="artifacts/item_idx_mapping.json")
    parser.add_argument("--output", type=str, default="artifacts/faiss_index")
    parser.add_argument("--index-type", type=str, default="IVF")
    parser.add_argument("--nlist", type=int, default=100)
    parser.add_argument("--nprobe", type=int, default=10)
    args = parser.parse_args()

    builder = build_index_from_embeddings(
        embeddings_path=args.embeddings,
        idx_mapping_path=args.idx_mapping,
        output_path=args.output,
        index_type=args.index_type,
        nlist=args.nlist,
        nprobe=args.nprobe,
    )

    test_query = np.random.randn(1, builder.embedding_dim).astype(np.float32)
    scores, ids = builder.search(test_query, top_k=10)
    print(f"Test search results: {ids[0]}")
