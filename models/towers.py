import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class AttentionPooling(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1, bias=False),
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        attn_scores = self.attention(x).squeeze(-1)

        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)

        output = torch.bmm(attn_weights.unsqueeze(1), x).squeeze(1)
        return output


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        assert dim % num_heads == 0

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale

        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = F.softmax(attn, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)
        attn = self.dropout(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)

        return x


class TextEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        word_embedding_dim: int = 256,
        output_dim: int = 256,
        num_heads: int = 8,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.word_embedding = nn.Embedding(vocab_size, word_embedding_dim, padding_idx=0)
        self.self_attention = MultiHeadSelfAttention(word_embedding_dim, num_heads, dropout)
        self.attention_pool = AttentionPooling(word_embedding_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if mask is None:
            mask = (input_ids != 0).float()

        x = self.word_embedding(input_ids)
        x = self.dropout(x)
        x = x + self.self_attention(x, mask)
        output = self.attention_pool(x, mask)

        return output


class ItemTower(nn.Module):
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
    ):
        super().__init__()

        self.output_dim = output_dim

        self.title_encoder = TextEncoder(vocab_size, word_embedding_dim, hidden_dim, num_heads, dropout)
        self.abstract_encoder = TextEncoder(vocab_size, word_embedding_dim, hidden_dim, num_heads, dropout)

        self.category_embedding = nn.Embedding(num_categories, category_embedding_dim, padding_idx=0)
        self.subcategory_embedding = nn.Embedding(num_subcategories, category_embedding_dim, padding_idx=0)

        fusion_input_dim = hidden_dim * 2 + category_embedding_dim * 2
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(
        self,
        title: torch.Tensor,
        abstract: torch.Tensor,
        category: torch.Tensor,
        subcategory: torch.Tensor,
        normalize: bool = True,
    ) -> torch.Tensor:
        title_emb = self.title_encoder(title)
        abstract_emb = self.abstract_encoder(abstract)
        cat_emb = self.category_embedding(category)
        subcat_emb = self.subcategory_embedding(subcategory)

        combined = torch.cat([title_emb, abstract_emb, cat_emb, subcat_emb], dim=-1)
        output = self.fusion(combined)

        if normalize:
            output = F.normalize(output, p=2, dim=-1, eps=1e-8)

        return output


class UserTower(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        word_embedding_dim: int = 256,
        num_categories: int = 20,
        category_embedding_dim: int = 64,
        hidden_dim: int = 256,
        output_dim: int = 128,
        num_heads: int = 8,
        dropout: float = 0.2,
        max_history_len: int = 50,
    ):
        super().__init__()

        self.output_dim = output_dim

        self.title_encoder = TextEncoder(vocab_size, word_embedding_dim, hidden_dim, num_heads, dropout)
        self.category_embedding = nn.Embedding(num_categories, category_embedding_dim, padding_idx=0)

        news_dim = hidden_dim + category_embedding_dim
        self.news_proj = nn.Linear(news_dim, hidden_dim)
        self.history_attention = AttentionPooling(hidden_dim, hidden_dim)

        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(
        self,
        history_title: torch.Tensor,
        history_category: torch.Tensor,
        history_mask: torch.Tensor,
        normalize: bool = True,
    ) -> torch.Tensor:
        batch_size, history_len, title_len = history_title.shape

        flat_title = history_title.view(-1, title_len)
        flat_category = history_category.view(-1)

        title_emb = self.title_encoder(flat_title)
        cat_emb = self.category_embedding(flat_category)

        news_emb = torch.cat([title_emb, cat_emb], dim=-1)
        news_emb = self.news_proj(news_emb)
        news_emb = news_emb.view(batch_size, history_len, -1)

        user_emb = self.history_attention(news_emb, history_mask)
        output = self.output_proj(user_emb)

        if normalize:
            output = F.normalize(output, p=2, dim=-1, eps=1e-8)

        return output


class SharedBottomTower(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        word_embedding_dim: int = 256,
        hidden_dim: int = 256,
        output_dim: int = 128,
        num_heads: int = 8,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.text_encoder = TextEncoder(vocab_size, word_embedding_dim, hidden_dim, num_heads, dropout)

        self.user_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

        self.item_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def encode_user(self, history_text: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        text_emb = self.text_encoder(history_text, mask)
        return F.normalize(self.user_head(text_emb), p=2, dim=-1, eps=1e-8)

    def encode_item(self, item_text: torch.Tensor) -> torch.Tensor:
        text_emb = self.text_encoder(item_text)
        return F.normalize(self.item_head(text_emb), p=2, dim=-1, eps=1e-8)
