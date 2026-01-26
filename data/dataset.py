import random
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class NewsDataset(Dataset):

    def __init__(self, news_data: Dict[str, dict]):
        self.news_ids = list(news_data.keys())
        self.news_data = news_data

    def __len__(self) -> int:
        return len(self.news_ids)

    def __getitem__(self, idx: int) -> dict:
        news_id = self.news_ids[idx]
        news = self.news_data[news_id]

        return {
            "news_idx": news["news_idx"],
            "category": news["category"],
            "subcategory": news["subcategory"],
            "title": torch.tensor(news["title"], dtype=torch.long),
            "abstract": torch.tensor(news["abstract"], dtype=torch.long),
        }


class TwoTowerDataset(Dataset):

    def __init__(
        self,
        samples: List[dict],
        news_data: Dict[str, dict],
        num_negatives: int = 4,
        is_train: bool = True,
    ):
        self.samples = samples
        self.news_data = news_data
        self.num_negatives = num_negatives
        self.is_train = is_train

        self.all_news_indices = [
            data["news_idx"] for data in news_data.values() if data["news_idx"] > 0
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]

        pos_news_idx = sample["news_idx"]
        pos_news = self._get_news_features(pos_news_idx)

        history = sample["history"]
        history_features = self._get_history_features(history)

        result = {
            "user_idx": sample["user_idx"],
            "history": torch.tensor(history, dtype=torch.long),
            "history_title": history_features["title"],
            "history_abstract": history_features["abstract"],
            "history_category": history_features["category"],
            "history_mask": history_features["mask"],
            "pos_news_idx": pos_news_idx,
            "pos_title": pos_news["title"],
            "pos_abstract": pos_news["abstract"],
            "pos_category": pos_news["category"],
            "pos_subcategory": pos_news["subcategory"],
            "label": sample.get("label", 1),
        }

        if self.is_train:
            neg_indices = self._sample_negatives(pos_news_idx)
            neg_features = self._get_batch_news_features(neg_indices)
            result.update({
                "neg_news_indices": torch.tensor(neg_indices, dtype=torch.long),
                "neg_title": neg_features["title"],
                "neg_abstract": neg_features["abstract"],
                "neg_category": neg_features["category"],
                "neg_subcategory": neg_features["subcategory"],
            })

        return result

    def _get_news_features(self, news_idx: int) -> dict:
        for news_id, data in self.news_data.items():
            if data["news_idx"] == news_idx:
                return {
                    "title": torch.tensor(data["title"], dtype=torch.long),
                    "abstract": torch.tensor(data["abstract"], dtype=torch.long),
                    "category": data["category"],
                    "subcategory": data["subcategory"],
                }

        return {
            "title": torch.zeros(30, dtype=torch.long),
            "abstract": torch.zeros(100, dtype=torch.long),
            "category": 0,
            "subcategory": 0,
        }

    def _get_batch_news_features(self, news_indices: List[int]) -> dict:
        titles = []
        abstracts = []
        categories = []
        subcategories = []

        for idx in news_indices:
            features = self._get_news_features(idx)
            titles.append(features["title"])
            abstracts.append(features["abstract"])
            categories.append(features["category"])
            subcategories.append(features["subcategory"])

        return {
            "title": torch.stack(titles),
            "abstract": torch.stack(abstracts),
            "category": torch.tensor(categories, dtype=torch.long),
            "subcategory": torch.tensor(subcategories, dtype=torch.long),
        }

    def _get_history_features(self, history: List[int]) -> dict:
        titles = []
        abstracts = []
        categories = []
        mask = []

        for news_idx in history:
            if news_idx > 0:
                features = self._get_news_features(news_idx)
                titles.append(features["title"])
                abstracts.append(features["abstract"])
                categories.append(features["category"])
                mask.append(1.0)
            else:
                titles.append(torch.zeros(30, dtype=torch.long))
                abstracts.append(torch.zeros(100, dtype=torch.long))
                categories.append(0)
                mask.append(0.0)

        return {
            "title": torch.stack(titles),
            "abstract": torch.stack(abstracts),
            "category": torch.tensor(categories, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.float),
        }

    def _sample_negatives(self, pos_idx: int) -> List[int]:
        available = [idx for idx in self.all_news_indices if idx != pos_idx]
        if len(available) <= self.num_negatives:
            return available
        return random.sample(available, self.num_negatives)


class EvaluationDataset(Dataset):

    def __init__(
        self,
        samples: List[dict],
        news_data: Dict[str, dict],
    ):
        self.samples = samples
        self.news_data = news_data

        self.impressions = {}
        for sample in samples:
            imp_id = sample.get("impression_id", 0)
            if imp_id not in self.impressions:
                self.impressions[imp_id] = []
            self.impressions[imp_id].append(sample)

        self.impression_ids = list(self.impressions.keys())

    def __len__(self) -> int:
        return len(self.impression_ids)

    def __getitem__(self, idx: int) -> dict:
        imp_id = self.impression_ids[idx]
        samples = self.impressions[imp_id]

        first_sample = samples[0]
        history = first_sample["history"]

        news_indices = []
        labels = []

        for sample in samples:
            news_indices.append(sample["news_idx"])
            labels.append(sample["label"])

        history_features = self._get_history_features(history)

        return {
            "impression_id": imp_id,
            "user_idx": first_sample["user_idx"],
            "history": torch.tensor(history, dtype=torch.long),
            "history_title": history_features["title"],
            "history_abstract": history_features["abstract"],
            "history_category": history_features["category"],
            "history_mask": history_features["mask"],
            "news_indices": news_indices,
            "labels": labels,
        }

    def _get_history_features(self, history: List[int]) -> dict:
        titles = []
        abstracts = []
        categories = []
        mask = []

        for news_idx in history:
            features = self._get_news_features(news_idx) if news_idx > 0 else None
            if features:
                titles.append(features["title"])
                abstracts.append(features["abstract"])
                categories.append(features["category"])
                mask.append(1.0)
            else:
                titles.append(torch.zeros(30, dtype=torch.long))
                abstracts.append(torch.zeros(100, dtype=torch.long))
                categories.append(0)
                mask.append(0.0)

        return {
            "title": torch.stack(titles),
            "abstract": torch.stack(abstracts),
            "category": torch.tensor(categories, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.float),
        }

    def _get_news_features(self, news_idx: int) -> Optional[dict]:
        for news_id, data in self.news_data.items():
            if data["news_idx"] == news_idx:
                return {
                    "title": torch.tensor(data["title"], dtype=torch.long),
                    "abstract": torch.tensor(data["abstract"], dtype=torch.long),
                    "category": data["category"],
                    "subcategory": data["subcategory"],
                }
        return None


def collate_fn(batch: List[dict]) -> dict:
    result = {}

    for key in batch[0].keys():
        values = [item[key] for item in batch]

        if isinstance(values[0], torch.Tensor):
            result[key] = torch.stack(values)
        elif isinstance(values[0], (int, float)):
            result[key] = torch.tensor(values)
        else:
            result[key] = values

    return result


def create_data_loaders(
    train_samples: List[dict],
    dev_samples: List[dict],
    news_data: Dict[str, dict],
    batch_size: int = 256,
    num_negatives: int = 4,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader]:

    train_dataset = TwoTowerDataset(
        samples=train_samples,
        news_data=news_data,
        num_negatives=num_negatives,
        is_train=True,
    )

    eval_dataset = TwoTowerDataset(
        samples=dev_samples,
        news_data=news_data,
        num_negatives=num_negatives,
        is_train=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    eval_loader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    return train_loader, eval_loader


def create_news_loader(
    news_data: Dict[str, dict],
    batch_size: int = 512,
    num_workers: int = 4,
) -> DataLoader:

    dataset = NewsDataset(news_data)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return loader
