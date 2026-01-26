import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import torch
import numpy as np

from models.towers import ItemTower, UserTower, AttentionPooling, TextEncoder
from models.two_tower import TwoTowerModel, create_model_from_config


class TestAttentionPooling:
    def test_forward(self):
        batch_size = 4
        seq_len = 10
        dim = 64

        layer = AttentionPooling(dim, hidden_dim=32)
        x = torch.randn(batch_size, seq_len, dim)
        mask = torch.ones(batch_size, seq_len)

        output = layer(x, mask)

        assert output.shape == (batch_size, dim)

    def test_masking(self):
        batch_size = 2
        seq_len = 5
        dim = 32

        layer = AttentionPooling(dim)
        x = torch.randn(batch_size, seq_len, dim)
        mask = torch.tensor([
            [1, 1, 1, 0, 0],
            [1, 1, 0, 0, 0],
        ], dtype=torch.float)

        output = layer(x, mask)

        assert output.shape == (batch_size, dim)
        assert not torch.isnan(output).any()


class TestTextEncoder:
    def test_forward(self):
        batch_size = 4
        seq_len = 30
        vocab_size = 1000
        output_dim = 128

        encoder = TextEncoder(vocab_size, word_embedding_dim=64, output_dim=output_dim)
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))

        output = encoder(input_ids)

        assert output.shape == (batch_size, 64)

    def test_with_padding(self):
        batch_size = 2
        seq_len = 20
        vocab_size = 500

        encoder = TextEncoder(vocab_size, word_embedding_dim=64)
        input_ids = torch.zeros(batch_size, seq_len, dtype=torch.long)
        input_ids[0, :10] = torch.randint(1, vocab_size, (10,))
        input_ids[1, :5] = torch.randint(1, vocab_size, (5,))

        output = encoder(input_ids)

        assert not torch.isnan(output).any()


class TestItemTower:
    def test_forward(self):
        batch_size = 8
        vocab_size = 1000
        output_dim = 128

        tower = ItemTower(
            vocab_size=vocab_size,
            word_embedding_dim=64,
            num_categories=20,
            num_subcategories=100,
            output_dim=output_dim,
        )

        title = torch.randint(0, vocab_size, (batch_size, 30))
        abstract = torch.randint(0, vocab_size, (batch_size, 100))
        category = torch.randint(0, 20, (batch_size,))
        subcategory = torch.randint(0, 100, (batch_size,))

        output = tower(title, abstract, category, subcategory)

        assert output.shape == (batch_size, output_dim)

    def test_l2_normalization(self):
        batch_size = 4
        tower = ItemTower(vocab_size=500, output_dim=64)

        title = torch.randint(0, 500, (batch_size, 30))
        abstract = torch.randint(0, 500, (batch_size, 100))
        category = torch.randint(0, 20, (batch_size,))
        subcategory = torch.randint(0, 300, (batch_size,))

        output = tower(title, abstract, category, subcategory, normalize=True)
        norms = torch.norm(output, dim=1)

        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


class TestUserTower:
    def test_forward(self):
        batch_size = 4
        history_len = 50
        title_len = 30
        vocab_size = 1000
        output_dim = 128

        tower = UserTower(
            vocab_size=vocab_size,
            output_dim=output_dim,
            max_history_len=history_len,
        )

        history_title = torch.randint(0, vocab_size, (batch_size, history_len, title_len))
        history_category = torch.randint(0, 20, (batch_size, history_len))
        history_mask = torch.ones(batch_size, history_len)

        output = tower(history_title, history_category, history_mask)

        assert output.shape == (batch_size, output_dim)

    def test_l2_normalization(self):
        batch_size = 2
        tower = UserTower(vocab_size=500, output_dim=64)

        history_title = torch.randint(0, 500, (batch_size, 50, 30))
        history_category = torch.randint(0, 20, (batch_size, 50))
        history_mask = torch.ones(batch_size, 50)

        output = tower(history_title, history_category, history_mask, normalize=True)
        norms = torch.norm(output, dim=1)

        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


class TestTwoTowerModel:
    def test_forward(self):
        batch_size = 4
        vocab_size = 1000
        output_dim = 128

        model = TwoTowerModel(
            vocab_size=vocab_size,
            output_dim=output_dim,
            temperature=0.07,
        )

        history_title = torch.randint(0, vocab_size, (batch_size, 50, 30))
        history_category = torch.randint(0, 20, (batch_size, 50))
        history_mask = torch.ones(batch_size, 50)
        pos_title = torch.randint(0, vocab_size, (batch_size, 30))
        pos_abstract = torch.randint(0, vocab_size, (batch_size, 100))
        pos_category = torch.randint(0, 20, (batch_size,))
        pos_subcategory = torch.randint(0, 300, (batch_size,))

        outputs = model(
            history_title=history_title,
            history_category=history_category,
            history_mask=history_mask,
            pos_title=pos_title,
            pos_abstract=pos_abstract,
            pos_category=pos_category,
            pos_subcategory=pos_subcategory,
        )

        assert "loss" in outputs
        assert "user_emb" in outputs
        assert "pos_emb" in outputs
        assert outputs["user_emb"].shape == (batch_size, output_dim)
        assert outputs["pos_emb"].shape == (batch_size, output_dim)

    def test_with_negatives(self):
        batch_size = 4
        num_neg = 4
        vocab_size = 500

        model = TwoTowerModel(vocab_size=vocab_size, output_dim=64)

        outputs = model(
            history_title=torch.randint(0, vocab_size, (batch_size, 50, 30)),
            history_category=torch.randint(0, 20, (batch_size, 50)),
            history_mask=torch.ones(batch_size, 50),
            pos_title=torch.randint(0, vocab_size, (batch_size, 30)),
            pos_abstract=torch.randint(0, vocab_size, (batch_size, 100)),
            pos_category=torch.randint(0, 20, (batch_size,)),
            pos_subcategory=torch.randint(0, 300, (batch_size,)),
            neg_title=torch.randint(0, vocab_size, (batch_size, num_neg, 30)),
            neg_abstract=torch.randint(0, vocab_size, (batch_size, num_neg, 100)),
            neg_category=torch.randint(0, 20, (batch_size, num_neg)),
            neg_subcategory=torch.randint(0, 300, (batch_size, num_neg)),
        )

        assert outputs["loss"].item() > 0

    def test_encode_methods(self):
        model = TwoTowerModel(vocab_size=500, output_dim=64)

        user_emb = model.encode_user(
            history_title=torch.randint(0, 500, (2, 50, 30)),
            history_category=torch.randint(0, 20, (2, 50)),
            history_mask=torch.ones(2, 50),
        )

        item_emb = model.encode_item(
            title=torch.randint(0, 500, (3, 30)),
            abstract=torch.randint(0, 500, (3, 100)),
            category=torch.randint(0, 20, (3,)),
            subcategory=torch.randint(0, 300, (3,)),
        )

        assert user_emb.shape == (2, 64)
        assert item_emb.shape == (3, 64)

    def test_create_from_config(self):
        config = {
            "model": {
                "vocab_size": 1000,
                "word_embedding_dim": 64,
                "num_categories": 20,
                "num_subcategories": 100,
                "embedding_dim": 128,
                "attention_heads": 4,
                "dropout": 0.1,
            },
            "two_tower": {
                "temperature": 0.07,
                "use_in_batch_negatives": True,
            },
            "data": {
                "max_history_len": 50,
            },
        }

        model = create_model_from_config(config)

        assert model.output_dim == 128
        assert model.temperature == 0.07


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
