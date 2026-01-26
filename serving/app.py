import os
import sys
import time
import json
from pathlib import Path
from typing import Dict, List, Optional
from contextlib import asynccontextmanager
import logging

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from serving.schemas import (
    RecommendRequest,
    RecommendResponse,
    RecommendationItem,
    HealthResponse,
    BatchRecommendRequest,
    BatchRecommendResponse,
)
from serving.cache import RedisCache
from retrieval.index_builder import FAISSIndexBuilder
from retrieval.retriever import Retriever
from models.two_tower import create_model_from_config
from models.reranker import LightGBMReranker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RecommenderService:
    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.model = None
        self.retriever = None
        self.reranker = None
        self.cache = None
        self.news_data = None
        self.user_data = None
        self.news2idx = None
        self.user2idx = None
        self.idx2news = None
        self.device = None

    def load(self):
        logger.info("Loading recommender service...")

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        logger.info(f"Using device: {self.device}")

        artifacts_dir = Path("artifacts")
        data_dir = Path("data/processed")

        with open(artifacts_dir / "model_config.json", "r") as f:
            model_config = json.load(f)

        self.model = create_model_from_config(model_config)
        from training.trainer import load_checkpoint_safe
        checkpoint = load_checkpoint_safe(artifacts_dir / "two_tower_model.pt", self.device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        self.model.to(self.device)
        self.model.eval()

        logger.info("Model loaded")

        self.retriever = Retriever(
            index_path=str(artifacts_dir / "faiss_index"),
            item_embeddings_path=str(artifacts_dir / "item_embeddings.npy"),
        )
        logger.info("FAISS index loaded")

        reranker_path = artifacts_dir / "reranker.txt"
        if reranker_path.exists():
            self.reranker = LightGBMReranker()
            self.reranker.load(str(reranker_path))
            logger.info("Reranker loaded")
        else:
            self.reranker = None
            logger.info("No reranker found, using retrieval scores only")

        import pickle
        with open(data_dir / "news_data.pkl", "rb") as f:
            self.news_data = pickle.load(f)

        with open(data_dir / "mappings.json", "r") as f:
            mappings = json.load(f)
            self.news2idx = mappings["news2idx"]
            self.user2idx = mappings["user2idx"]
            self.idx2news = {v: k for k, v in self.news2idx.items()}

        logger.info("Data loaded")

        redis_config = self.config.get("redis", {})
        self.cache = RedisCache(
            host=redis_config.get("host", "localhost"),
            port=redis_config.get("port", 6379),
            db=redis_config.get("db", 0),
            password=redis_config.get("password"),
            embedding_ttl=redis_config.get("embedding_ttl", 3600),
        )
        self.cache.connect()

        logger.info("Service ready")

    @torch.no_grad()
    def compute_user_embedding(
        self,
        history: List[str],
    ) -> np.ndarray:
        history_indices = []
        for news_id in history:
            if news_id in self.news2idx:
                history_indices.append(self.news2idx[news_id])

        if not history_indices:
            return np.zeros(self.config["model"]["embedding_dim"], dtype=np.float32)

        max_history = self.config["data"]["max_history_len"]
        history_indices = history_indices[-max_history:]
        history_indices = history_indices + [0] * (max_history - len(history_indices))

        history_titles = []
        history_categories = []

        for idx in history_indices:
            if idx > 0 and idx in self.idx2news:
                news_id = self.idx2news[idx]
                news = self.news_data.get(news_id, {})
                history_titles.append(news.get("title", [0] * 30))
                history_categories.append(news.get("category", 0))
            else:
                history_titles.append([0] * 30)
                history_categories.append(0)

        history_title = torch.tensor([history_titles], dtype=torch.long).to(self.device)
        history_category = torch.tensor([history_categories], dtype=torch.long).to(self.device)
        history_mask = torch.tensor(
            [[1.0 if idx > 0 else 0.0 for idx in history_indices]], dtype=torch.float
        ).to(self.device)

        user_emb = self.model.encode_user(
            history_title=history_title,
            history_category=history_category,
            history_mask=history_mask,
        )

        return user_emb.cpu().numpy()[0]

    def recommend(
        self,
        user_id: str,
        history: Optional[List[str]] = None,
        num_recommendations: int = 20,
        exclude_ids: Optional[List[str]] = None,
    ) -> List[RecommendationItem]:
        user_emb = self.cache.get_user_embedding(user_id)

        if user_emb is None:
            if history is None:
                cached_history = self.cache.get_user_history(user_id)
                if cached_history:
                    history = cached_history
                else:
                    history = []

            user_emb = self.compute_user_embedding(history)
            self.cache.set_user_embedding(user_id, user_emb)

            if history:
                self.cache.set_user_history(user_id, history)

        exclude_indices = None
        if exclude_ids:
            exclude_indices = [
                self.news2idx.get(nid, -1) for nid in exclude_ids
            ]
            exclude_indices = [[idx for idx in exclude_indices if idx >= 0]]

        retrieval_k = self.config["serving"]["retrieval_top_k"]
        scores, item_indices = self.retriever.retrieve(
            user_emb.reshape(1, -1),
            top_k=retrieval_k,
            exclude_ids=exclude_indices,
        )

        item_indices = item_indices[0]
        scores = scores[0]

        if self.reranker is not None:
            valid_mask = item_indices >= 0
            valid_indices = item_indices[valid_mask].tolist()

            if valid_indices:
                candidate_emb = self.retriever.get_item_embeddings(valid_indices)
                reranked = self.reranker.rerank(
                    user_emb=user_emb,
                    candidate_emb=candidate_emb,
                    candidate_ids=valid_indices,
                    top_k=num_recommendations,
                )

                recommendations = []
                for rank, (item_idx, score) in enumerate(reranked):
                    if item_idx in self.idx2news:
                        news_id = self.idx2news[item_idx]
                        recommendations.append(
                            RecommendationItem(
                                item_id=news_id,
                                score=float(score),
                                rank=rank + 1,
                            )
                        )

                return recommendations

        recommendations = []
        for rank, (item_idx, score) in enumerate(zip(item_indices[:num_recommendations], scores[:num_recommendations])):
            if item_idx >= 0 and item_idx in self.idx2news:
                news_id = self.idx2news[item_idx]
                recommendations.append(
                    RecommendationItem(
                        item_id=news_id,
                        score=float(score),
                        rank=rank + 1,
                    )
                )

        return recommendations


service: Optional[RecommenderService] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global service
    service = RecommenderService()
    service.load()
    yield
    service = None


app = FastAPI(
    title="Two-Tower Feed Recommender",
    description="Production-grade feed recommendation API",
    version="1.0.0",
    lifespan=lifespan,
)

CORS_ORIGINS = os.environ.get("CORS_ORIGINS", "http://localhost:3000,http://localhost:8080").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    global service
    return HealthResponse(
        status="healthy",
        index_size=service.retriever.index_builder.index.ntotal if service else 0,
        cache_connected=service.cache.is_connected if service else False,
    )


@app.post("/recommend", response_model=RecommendResponse)
async def recommend(request: RecommendRequest):
    global service
    if service is None:
        raise HTTPException(status_code=503, detail="Service not ready")

    start_time = time.perf_counter()

    recommendations = service.recommend(
        user_id=request.user_id,
        history=request.history,
        num_recommendations=request.num_recommendations,
        exclude_ids=request.exclude_ids,
    )

    latency_ms = (time.perf_counter() - start_time) * 1000

    return RecommendResponse(
        user_id=request.user_id,
        recommendations=recommendations,
        latency_ms=latency_ms,
    )


@app.post("/recommend/batch", response_model=BatchRecommendResponse)
async def batch_recommend(request: BatchRecommendRequest):
    global service
    if service is None:
        raise HTTPException(status_code=503, detail="Service not ready")

    start_time = time.perf_counter()

    responses = []
    for req in request.requests:
        req_start = time.perf_counter()
        recommendations = service.recommend(
            user_id=req.user_id,
            history=req.history,
            num_recommendations=req.num_recommendations,
            exclude_ids=req.exclude_ids,
        )
        req_latency = (time.perf_counter() - req_start) * 1000

        responses.append(
            RecommendResponse(
                user_id=req.user_id,
                recommendations=recommendations,
                latency_ms=req_latency,
            )
        )

    total_latency = (time.perf_counter() - start_time) * 1000

    return BatchRecommendResponse(
        responses=responses,
        total_latency_ms=total_latency,
    )


@app.delete("/cache/{user_id}")
async def invalidate_cache(user_id: str):
    global service
    if service is None:
        raise HTTPException(status_code=503, detail="Service not ready")

    service.cache.delete_user_embedding(user_id)
    return {"status": "ok", "user_id": user_id}


@app.get("/cache/stats")
async def cache_stats():
    global service
    if service is None:
        raise HTTPException(status_code=503, detail="Service not ready")

    return service.cache.get_stats()


def run_server(host: str = "0.0.0.0", port: int = 8000, workers: int = 1):
    import uvicorn
    uvicorn.run(
        "serving.app:app",
        host=host,
        port=port,
        workers=workers,
        reload=False,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    run_server(args.host, args.port, args.workers)
