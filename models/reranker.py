import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LightGBMReranker:

    def __init__(
        self,
        num_leaves: int = 31,
        max_depth: int = -1,
        learning_rate: float = 0.05,
        n_estimators: int = 100,
        objective: str = "lambdarank",
        metric: str = "ndcg",
        ndcg_eval_at: List[int] = [10, 50, 100],
        early_stopping_rounds: int = 50,
        verbose: int = 10,
    ):
        self.params = {
            "objective": objective,
            "metric": metric,
            "num_leaves": num_leaves,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "n_estimators": n_estimators,
            "ndcg_eval_at": ndcg_eval_at,
            "verbose": verbose,
            "force_col_wise": True,
        }
        self.early_stopping_rounds = early_stopping_rounds
        self.model: Optional[lgb.Booster] = None
        self.feature_names: Optional[List[str]] = None

    def create_features(
        self,
        user_emb: np.ndarray,
        item_emb: np.ndarray,
        item_popularity: Optional[np.ndarray] = None,
        user_history_categories: Optional[np.ndarray] = None,
        item_categories: Optional[np.ndarray] = None,
        additional_features: Optional[Dict[str, np.ndarray]] = None,
    ) -> Tuple[np.ndarray, List[str]]:
        features_list = []
        feature_names = []

        dot_sim = np.sum(user_emb * item_emb, axis=1, keepdims=True)
        features_list.append(dot_sim)
        feature_names.append("dot_similarity")

        user_norm = np.linalg.norm(user_emb, axis=1, keepdims=True) + 1e-8
        item_norm = np.linalg.norm(item_emb, axis=1, keepdims=True) + 1e-8
        cos_sim = np.sum(user_emb * item_emb, axis=1, keepdims=True) / (user_norm * item_norm)
        features_list.append(cos_sim)
        feature_names.append("cosine_similarity")

        l2_dist = np.linalg.norm(user_emb - item_emb, axis=1, keepdims=True)
        features_list.append(l2_dist)
        feature_names.append("l2_distance")

        if item_popularity is not None:
            features_list.append(item_popularity.reshape(-1, 1))
            feature_names.append("item_popularity")

        if user_history_categories is not None and item_categories is not None:
            cat_match = np.sum(user_history_categories * item_categories, axis=1, keepdims=True)
            features_list.append(cat_match)
            feature_names.append("category_match")

        if additional_features:
            for name, values in additional_features.items():
                if values.ndim == 1:
                    values = values.reshape(-1, 1)
                features_list.append(values)
                feature_names.extend([f"{name}_{i}" for i in range(values.shape[1])])

        features = np.hstack(features_list)
        self.feature_names = feature_names

        return features, feature_names

    def train(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        groups: np.ndarray,
        val_features: Optional[np.ndarray] = None,
        val_labels: Optional[np.ndarray] = None,
        val_groups: Optional[np.ndarray] = None,
    ) -> Dict[str, List[float]]:
        train_data = lgb.Dataset(
            features,
            label=labels,
            group=groups,
            feature_name=self.feature_names,
        )

        valid_sets = [train_data]
        valid_names = ["train"]

        if val_features is not None:
            val_data = lgb.Dataset(
                val_features,
                label=val_labels,
                group=val_groups,
                reference=train_data,
            )
            valid_sets.append(val_data)
            valid_names.append("valid")

        callbacks = [
            lgb.log_evaluation(self.params["verbose"]),
        ]
        if val_features is not None:
            callbacks.append(lgb.early_stopping(self.early_stopping_rounds))

        self.model = lgb.train(
            self.params,
            train_data,
            num_boost_round=self.params["n_estimators"],
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks,
        )

        logger.info(f"Training complete. Best iteration: {self.model.best_iteration}")

        return {"best_iteration": self.model.best_iteration}

    def predict(self, features: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        return self.model.predict(features, num_iteration=self.model.best_iteration)

    def rerank(
        self,
        user_emb: np.ndarray,
        candidate_emb: np.ndarray,
        candidate_ids: List[int],
        item_popularity: Optional[np.ndarray] = None,
        top_k: int = 20,
    ) -> List[Tuple[int, float]]:
        num_candidates = len(candidate_ids)

        user_emb_expanded = np.tile(user_emb, (num_candidates, 1))

        features, _ = self.create_features(
            user_emb=user_emb_expanded,
            item_emb=candidate_emb,
            item_popularity=item_popularity,
        )

        scores = self.predict(features)

        ranked_indices = np.argsort(scores)[::-1][:top_k]
        return [(candidate_ids[i], float(scores[i])) for i in ranked_indices]

    def save(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if self.model is not None:
            self.model.save_model(str(path))

        meta_path = path.with_suffix(".meta.json")
        with open(meta_path, "w") as f:
            json.dump({
                "feature_names": self.feature_names,
                "params": self.params,
            }, f)

        logger.info(f"Model saved to {path}")

    def load(self, path: Union[str, Path]) -> None:
        path = Path(path)

        self.model = lgb.Booster(model_file=str(path))

        meta_path = path.with_suffix(".meta.json")
        if meta_path.exists():
            with open(meta_path, "r") as f:
                meta = json.load(f)
                self.feature_names = meta.get("feature_names")
                self.params = meta.get("params", self.params)

        logger.info(f"Model loaded from {path}")

    def get_feature_importance(self) -> Dict[str, float]:
        if self.model is None:
            raise ValueError("Model not trained.")

        importance = self.model.feature_importance(importance_type="gain")
        return dict(zip(self.feature_names, importance))


def prepare_reranker_training_data(
    user_embeddings: np.ndarray,
    item_embeddings: np.ndarray,
    samples: List[dict],
    item_idx_to_emb_idx: Dict[int, int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    impressions = {}
    for sample in samples:
        imp_id = sample.get("impression_id", 0)
        if imp_id not in impressions:
            impressions[imp_id] = {
                "user_idx": sample["user_idx"],
                "items": [],
                "labels": [],
            }
        impressions[imp_id]["items"].append(sample["news_idx"])
        impressions[imp_id]["labels"].append(sample.get("label", 0))

    all_features = []
    all_labels = []
    all_groups = []

    for imp_id, imp_data in impressions.items():
        user_idx = imp_data["user_idx"]
        items = imp_data["items"]
        labels = imp_data["labels"]

        if len(items) < 2:
            continue

        user_emb = user_embeddings[user_idx]
        user_emb_expanded = np.tile(user_emb, (len(items), 1))

        item_embs = []
        for item_idx in items:
            if item_idx in item_idx_to_emb_idx:
                emb_idx = item_idx_to_emb_idx[item_idx]
                item_embs.append(item_embeddings[emb_idx])
            else:
                item_embs.append(np.zeros(item_embeddings.shape[1]))
        item_embs = np.array(item_embs)

        features = np.concatenate([
            np.sum(user_emb_expanded * item_embs, axis=1, keepdims=True),
            np.linalg.norm(user_emb_expanded - item_embs, axis=1, keepdims=True),
        ], axis=1)

        all_features.append(features)
        all_labels.extend(labels)
        all_groups.append(len(items))

    features = np.vstack(all_features)
    labels = np.array(all_labels)
    groups = np.array(all_groups)

    n_groups = len(groups)
    split_idx = int(0.8 * n_groups)

    train_end = sum(groups[:split_idx])
    val_start = train_end

    train_features = features[:train_end]
    train_labels = labels[:train_end]
    train_groups = groups[:split_idx]

    val_features = features[val_start:]
    val_labels = labels[val_start:]
    val_groups = groups[split_idx:]

    return train_features, train_labels, train_groups, val_features, val_labels, val_groups
