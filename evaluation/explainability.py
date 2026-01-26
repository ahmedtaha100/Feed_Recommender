import json
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RecommendationExplainer:
    def __init__(
        self,
        model,
        news_data: Dict,
        idx2news: Dict,
        word2idx: Dict,
        device: torch.device,
    ):
        self.model = model
        self.news_data = news_data
        self.idx2news = idx2news
        self.word2idx = word2idx
        self.idx2word = {v: k for k, v in word2idx.items()}
        self.device = device
        self.model.eval()

    def _get_attention_weights(
        self,
        history_title: torch.Tensor,
        history_category: torch.Tensor,
        history_mask: torch.Tensor,
    ) -> Dict[str, np.ndarray]:
        attention_weights = {}

        with torch.no_grad():
            user_tower = self.model.user_tower

            batch_size, history_len, seq_len = history_title.shape
            flat_titles = history_title.view(-1, seq_len)

            title_embed = user_tower.title_encoder.word_embedding(flat_titles)

            attn_output = user_tower.title_encoder.self_attention.qkv(title_embed)
            q, k, v = attn_output.chunk(3, dim=-1)

            num_heads = user_tower.title_encoder.self_attention.num_heads
            head_dim = q.size(-1) // num_heads

            q = q.view(-1, seq_len, num_heads, head_dim).transpose(1, 2)
            k = k.view(-1, seq_len, num_heads, head_dim).transpose(1, 2)

            attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (head_dim ** 0.5)
            word_attention = torch.softmax(attn_scores, dim=-1)

            attention_weights["word_attention"] = word_attention.mean(dim=1).cpu().numpy()

            title_encoded = user_tower.title_encoder(flat_titles)
            title_encoded = title_encoded.view(batch_size, history_len, -1)

            news_emb = user_tower.news_proj(
                torch.cat([
                    title_encoded,
                    user_tower.category_embedding(history_category),
                ], dim=-1)
            )

            pool_attn = user_tower.history_attention.attention(news_emb)
            history_attention = torch.softmax(pool_attn.squeeze(-1) + (1 - history_mask) * -1e9, dim=-1)

            attention_weights["history_attention"] = history_attention.cpu().numpy()

        return attention_weights

    def explain_user_embedding(
        self,
        history_indices: List[int],
        top_k_history: int = 5,
    ) -> Dict:
        history_titles = []
        history_categories = []
        history_items = []

        max_title_len = 30

        for idx in history_indices:
            if idx > 0 and idx in self.idx2news:
                news_id = self.idx2news[idx]
                news = self.news_data.get(news_id, {})
                title = news.get("title", [0] * max_title_len)
                if len(title) < max_title_len:
                    title = title + [0] * (max_title_len - len(title))
                history_titles.append(title[:max_title_len])
                history_categories.append(news.get("category", 0))
                history_items.append({"idx": idx, "news_id": news_id})
            else:
                history_titles.append([0] * max_title_len)
                history_categories.append(0)
                history_items.append({"idx": idx, "news_id": None})

        history_title = torch.tensor([history_titles], dtype=torch.long).to(self.device)
        history_category = torch.tensor([history_categories], dtype=torch.long).to(self.device)
        history_mask = torch.tensor(
            [[1.0 if idx > 0 else 0.0 for idx in history_indices]], dtype=torch.float
        ).to(self.device)

        attention_weights = self._get_attention_weights(
            history_title, history_category, history_mask
        )

        history_importance = attention_weights["history_attention"][0]

        valid_indices = [(i, imp) for i, (imp, item) in enumerate(zip(history_importance, history_items)) if item["news_id"] is not None]
        valid_indices.sort(key=lambda x: x[1], reverse=True)

        top_influential = []
        for i, importance in valid_indices[:top_k_history]:
            item = history_items[i]
            news = self.news_data.get(item["news_id"], {})
            top_influential.append({
                "rank": len(top_influential) + 1,
                "news_id": item["news_id"],
                "importance_score": float(importance),
                "category": news.get("category", 0),
            })

        return {
            "total_history_items": sum(1 for item in history_items if item["news_id"] is not None),
            "top_influential_items": top_influential,
            "attention_distribution": {
                "mean": float(np.mean(history_importance[history_importance > 0])) if any(history_importance > 0) else 0,
                "std": float(np.std(history_importance[history_importance > 0])) if any(history_importance > 0) else 0,
                "max": float(np.max(history_importance)),
                "entropy": float(-np.sum(history_importance * np.log(history_importance + 1e-10))),
            },
        }

    def explain_recommendation(
        self,
        user_history: List[int],
        recommended_item_idx: int,
        candidate_pool: Optional[List[int]] = None,
    ) -> Dict:
        if recommended_item_idx not in self.idx2news:
            return {"error": "Item not found"}

        news_id = self.idx2news[recommended_item_idx]
        news = self.news_data.get(news_id, {})

        user_explanation = self.explain_user_embedding(user_history)

        with torch.no_grad():
            max_title_len = 30
            max_abstract_len = 100

            title = news.get("title", [0] * max_title_len)
            abstract = news.get("abstract", [0] * max_abstract_len)
            if len(title) < max_title_len:
                title = title + [0] * (max_title_len - len(title))
            if len(abstract) < max_abstract_len:
                abstract = abstract + [0] * (max_abstract_len - len(abstract))

            title_tensor = torch.tensor([title[:max_title_len]], dtype=torch.long).to(self.device)
            abstract_tensor = torch.tensor([abstract[:max_abstract_len]], dtype=torch.long).to(self.device)
            category_tensor = torch.tensor([news.get("category", 0)], dtype=torch.long).to(self.device)
            subcategory_tensor = torch.tensor([news.get("subcategory", 0)], dtype=torch.long).to(self.device)

            item_emb = self.model.encode_item(
                title=title_tensor,
                abstract=abstract_tensor,
                category=category_tensor,
                subcategory=subcategory_tensor,
            )

            history_titles = []
            history_categories = []
            for idx in user_history:
                if idx > 0 and idx in self.idx2news:
                    h_news_id = self.idx2news[idx]
                    h_news = self.news_data.get(h_news_id, {})
                    h_title = h_news.get("title", [0] * max_title_len)
                    if len(h_title) < max_title_len:
                        h_title = h_title + [0] * (max_title_len - len(h_title))
                    history_titles.append(h_title[:max_title_len])
                    history_categories.append(h_news.get("category", 0))
                else:
                    history_titles.append([0] * max_title_len)
                    history_categories.append(0)

            history_title = torch.tensor([history_titles], dtype=torch.long).to(self.device)
            history_category = torch.tensor([history_categories], dtype=torch.long).to(self.device)
            history_mask = torch.tensor(
                [[1.0 if idx > 0 else 0.0 for idx in user_history]], dtype=torch.float
            ).to(self.device)

            user_emb = self.model.encode_user(
                history_title=history_title,
                history_category=history_category,
                history_mask=history_mask,
            )

            similarity = torch.cosine_similarity(user_emb, item_emb).item()

        category_match = any(
            self.news_data.get(self.idx2news.get(idx, ""), {}).get("category") == news.get("category")
            for idx in user_history if idx > 0 and idx in self.idx2news
        )

        return {
            "recommended_item": {
                "news_id": news_id,
                "category": news.get("category", 0),
                "subcategory": news.get("subcategory", 0),
            },
            "similarity_score": float(similarity),
            "category_match_with_history": category_match,
            "user_profile_explanation": user_explanation,
            "recommendation_factors": [
                {"factor": "embedding_similarity", "contribution": float(similarity)},
                {"factor": "category_affinity", "contribution": 0.2 if category_match else 0.0},
            ],
        }

    def compute_feature_importance(
        self,
        num_samples: int = 100,
    ) -> Dict[str, float]:
        importance = {
            "title_words": 0.0,
            "category": 0.0,
            "history_length": 0.0,
            "history_recency": 0.0,
        }

        with torch.no_grad():
            max_title_len = 30

            base_title = torch.randint(1, 100, (num_samples, max_title_len)).to(self.device)
            base_category = torch.randint(0, 10, (num_samples,)).to(self.device)
            base_subcategory = torch.randint(0, 30, (num_samples,)).to(self.device)
            base_abstract = torch.randint(1, 100, (num_samples, 100)).to(self.device)

            base_emb = self.model.encode_item(
                title=base_title,
                abstract=base_abstract,
                category=base_category,
                subcategory=base_subcategory,
            )

            shuffled_title = base_title[torch.randperm(num_samples)]
            perm_emb = self.model.encode_item(
                title=shuffled_title,
                abstract=base_abstract,
                category=base_category,
                subcategory=base_subcategory,
            )
            importance["title_words"] = float(torch.mean(torch.abs(base_emb - perm_emb)).item())

            shuffled_category = base_category[torch.randperm(num_samples)]
            perm_emb = self.model.encode_item(
                title=base_title,
                abstract=base_abstract,
                category=shuffled_category,
                subcategory=base_subcategory,
            )
            importance["category"] = float(torch.mean(torch.abs(base_emb - perm_emb)).item())

        total = sum(importance.values())
        if total > 0:
            importance = {k: v / total for k, v in importance.items()}

        return importance


def generate_explanation_report(
    explainer: RecommendationExplainer,
    user_history: List[int],
    recommendations: List[Tuple[int, float]],
    output_path: str,
) -> None:
    report = {
        "user_profile": explainer.explain_user_embedding(user_history),
        "recommendations": [],
        "feature_importance": explainer.compute_feature_importance(),
    }

    for item_idx, score in recommendations[:5]:
        explanation = explainer.explain_recommendation(user_history, item_idx)
        explanation["retrieval_score"] = float(score)
        report["recommendations"].append(explanation)

    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    logger.info(f"Explanation report saved to {output_path}")
