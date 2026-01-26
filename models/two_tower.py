import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

from .towers import ItemTower, UserTower


class TwoTowerModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        word_embedding_dim: int = 256,
        num_categories: int = 20,
        num_subcategories: int = 300,
        category_embedding_dim: int = 64,
        hidden_dim: int = 256,
        output_dim: int = 128,
        num_heads: int = 8,
        dropout: float = 0.2,
        max_history_len: int = 50,
        temperature: float = 0.07,
        use_in_batch_negatives: bool = True,
    ):
        super().__init__()

        self.output_dim = output_dim
        self.temperature = temperature
        self.use_in_batch_negatives = use_in_batch_negatives

        self.user_tower = UserTower(
            vocab_size=vocab_size,
            word_embedding_dim=word_embedding_dim,
            num_categories=num_categories,
            category_embedding_dim=category_embedding_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_heads=num_heads,
            dropout=dropout,
            max_history_len=max_history_len,
        )

        self.item_tower = ItemTower(
            vocab_size=vocab_size,
            word_embedding_dim=word_embedding_dim,
            num_categories=num_categories,
            num_subcategories=num_subcategories,
            category_embedding_dim=category_embedding_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

    def encode_user(
        self,
        history_title: torch.Tensor,
        history_category: torch.Tensor,
        history_mask: torch.Tensor,
    ) -> torch.Tensor:
        return self.user_tower(
            history_title=history_title,
            history_category=history_category,
            history_mask=history_mask,
            normalize=True,
        )

    def encode_item(
        self,
        title: torch.Tensor,
        abstract: torch.Tensor,
        category: torch.Tensor,
        subcategory: torch.Tensor,
    ) -> torch.Tensor:
        return self.item_tower(
            title=title,
            abstract=abstract,
            category=category,
            subcategory=subcategory,
            normalize=True,
        )

    def forward(
        self,
        history_title: torch.Tensor,
        history_category: torch.Tensor,
        history_mask: torch.Tensor,
        pos_title: torch.Tensor,
        pos_abstract: torch.Tensor,
        pos_category: torch.Tensor,
        pos_subcategory: torch.Tensor,
        neg_title: Optional[torch.Tensor] = None,
        neg_abstract: Optional[torch.Tensor] = None,
        neg_category: Optional[torch.Tensor] = None,
        neg_subcategory: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        batch_size = history_title.size(0)

        user_emb = self.encode_user(
            history_title=history_title,
            history_category=history_category,
            history_mask=history_mask,
        )

        pos_emb = self.encode_item(
            title=pos_title,
            abstract=pos_abstract,
            category=pos_category,
            subcategory=pos_subcategory,
        )

        pos_scores = torch.sum(user_emb * pos_emb, dim=-1) / self.temperature

        neg_scores_list = []

        if self.use_in_batch_negatives:
            in_batch_scores = torch.mm(user_emb, pos_emb.t()) / self.temperature
            mask = torch.eye(batch_size, device=user_emb.device, dtype=torch.bool)
            in_batch_neg = in_batch_scores.masked_fill(mask, -1e9)
            neg_scores_list.append(in_batch_neg)

        if neg_title is not None:
            num_neg = neg_title.size(1)

            flat_neg_title = neg_title.view(-1, neg_title.size(-1))
            flat_neg_abstract = neg_abstract.view(-1, neg_abstract.size(-1))
            flat_neg_category = neg_category.view(-1)
            flat_neg_subcategory = neg_subcategory.view(-1)

            neg_emb = self.encode_item(
                title=flat_neg_title,
                abstract=flat_neg_abstract,
                category=flat_neg_category,
                subcategory=flat_neg_subcategory,
            )

            neg_emb = neg_emb.view(batch_size, num_neg, -1)

            sampled_neg_scores = torch.bmm(
                user_emb.unsqueeze(1), neg_emb.transpose(1, 2)
            ).squeeze(1) / self.temperature

            neg_scores_list.append(sampled_neg_scores)

        if neg_scores_list:
            all_neg_scores = torch.cat(neg_scores_list, dim=-1)
        else:
            all_neg_scores = torch.mm(user_emb, pos_emb.t()) / self.temperature
            mask = torch.eye(batch_size, device=user_emb.device, dtype=torch.bool)
            all_neg_scores = all_neg_scores.masked_fill(mask, -1e9)

        all_scores = torch.cat([pos_scores.unsqueeze(-1), all_neg_scores], dim=-1)
        log_probs = F.log_softmax(all_scores, dim=-1)
        loss = -log_probs[:, 0].mean()

        loss = torch.clamp(loss, min=0.0, max=100.0)

        return {
            "loss": loss,
            "user_emb": user_emb,
            "pos_emb": pos_emb,
            "pos_scores": pos_scores * self.temperature,
        }

    def compute_scores(
        self,
        user_emb: torch.Tensor,
        item_emb: torch.Tensor,
    ) -> torch.Tensor:
        return torch.mm(user_emb, item_emb.t())


class TwoTowerWithHardNegatives(TwoTowerModel):
    def __init__(self, *args, hard_neg_weight: float = 0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.hard_neg_weight = hard_neg_weight

    def forward(
        self,
        history_title: torch.Tensor,
        history_category: torch.Tensor,
        history_mask: torch.Tensor,
        pos_title: torch.Tensor,
        pos_abstract: torch.Tensor,
        pos_category: torch.Tensor,
        pos_subcategory: torch.Tensor,
        neg_title: Optional[torch.Tensor] = None,
        neg_abstract: Optional[torch.Tensor] = None,
        neg_category: Optional[torch.Tensor] = None,
        neg_subcategory: Optional[torch.Tensor] = None,
        hard_neg_title: Optional[torch.Tensor] = None,
        hard_neg_abstract: Optional[torch.Tensor] = None,
        hard_neg_category: Optional[torch.Tensor] = None,
        hard_neg_subcategory: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        output = super().forward(
            history_title=history_title,
            history_category=history_category,
            history_mask=history_mask,
            pos_title=pos_title,
            pos_abstract=pos_abstract,
            pos_category=pos_category,
            pos_subcategory=pos_subcategory,
            neg_title=neg_title,
            neg_abstract=neg_abstract,
            neg_category=neg_category,
            neg_subcategory=neg_subcategory,
        )

        if hard_neg_title is not None:
            user_emb = output["user_emb"]
            batch_size = user_emb.size(0)
            num_hard_neg = hard_neg_title.size(1)

            flat_hard_title = hard_neg_title.view(-1, hard_neg_title.size(-1))
            flat_hard_abstract = hard_neg_abstract.view(-1, hard_neg_abstract.size(-1))
            flat_hard_category = hard_neg_category.view(-1)
            flat_hard_subcategory = hard_neg_subcategory.view(-1)

            hard_neg_emb = self.encode_item(
                title=flat_hard_title,
                abstract=flat_hard_abstract,
                category=flat_hard_category,
                subcategory=flat_hard_subcategory,
            ).view(batch_size, num_hard_neg, -1)

            hard_neg_scores = torch.bmm(
                user_emb.unsqueeze(1), hard_neg_emb.transpose(1, 2)
            ).squeeze(1) / self.temperature

            pos_scores = output["pos_scores"] / self.temperature
            all_scores = torch.cat([pos_scores.unsqueeze(-1), hard_neg_scores], dim=-1)
            log_probs = F.log_softmax(all_scores, dim=-1)
            hard_neg_loss = -log_probs[:, 0].mean()

            output["loss"] = (1 - self.hard_neg_weight) * output["loss"] + self.hard_neg_weight * hard_neg_loss
            output["hard_neg_loss"] = hard_neg_loss

        return output


def create_model_from_config(config: dict) -> TwoTowerModel:
    model_config = config.get("model", {})
    two_tower_config = config.get("two_tower", {})

    return TwoTowerModel(
        vocab_size=model_config.get("vocab_size", 50000),
        word_embedding_dim=model_config.get("word_embedding_dim", 256),
        num_categories=model_config.get("num_categories", 20),
        num_subcategories=model_config.get("num_subcategories", 300),
        category_embedding_dim=model_config.get("category_embedding_dim", 64),
        hidden_dim=256,
        output_dim=model_config.get("embedding_dim", 128),
        num_heads=model_config.get("attention_heads", 8),
        dropout=model_config.get("dropout", 0.2),
        max_history_len=config.get("data", {}).get("max_history_len", 50),
        temperature=two_tower_config.get("temperature", 0.07),
        use_in_batch_negatives=two_tower_config.get("use_in_batch_negatives", True),
    )
