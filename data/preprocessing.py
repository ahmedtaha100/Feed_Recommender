import os
import json
import pickle
from pathlib import Path
from collections import Counter
from typing import Dict, List, Tuple, Optional
import logging

import numpy as np
import pandas as pd
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MINDPreprocessor:

    def __init__(
        self,
        data_dir: str = "data/mind",
        max_title_len: int = 30,
        max_abstract_len: int = 100,
        max_history_len: int = 50,
        min_word_freq: int = 2,
    ):
        self.data_dir = Path(data_dir)
        self.max_title_len = max_title_len
        self.max_abstract_len = max_abstract_len
        self.max_history_len = max_history_len
        self.min_word_freq = min_word_freq

        self.word2idx: Dict[str, int] = {"<PAD>": 0, "<UNK>": 1}
        self.category2idx: Dict[str, int] = {"<PAD>": 0}
        self.subcategory2idx: Dict[str, int] = {"<PAD>": 0}
        self.news2idx: Dict[str, int] = {"<PAD>": 0}
        self.user2idx: Dict[str, int] = {}

        self.news_data: Dict[str, dict] = {}

    def tokenize(self, text: str) -> List[str]:
        if pd.isna(text) or text is None:
            return []
        return text.lower().split()

    def build_vocab(self, texts: List[str]) -> None:
        word_counter = Counter()
        for text in tqdm(texts, desc="Building vocab"):
            words = self.tokenize(text)
            word_counter.update(words)

        for word, freq in word_counter.items():
            if freq >= self.min_word_freq:
                self.word2idx[word] = len(self.word2idx)

        logger.info(f"Vocabulary size: {len(self.word2idx)}")

    def encode_text(self, text: str, max_len: int) -> List[int]:
        words = self.tokenize(text)[:max_len]
        indices = [self.word2idx.get(w, self.word2idx["<UNK>"]) for w in words]
        indices = indices + [0] * (max_len - len(indices))
        return indices

    def parse_news(self, news_path: Path) -> pd.DataFrame:
        columns = ["news_id", "category", "subcategory", "title", "abstract",
                   "url", "title_entities", "abstract_entities"]

        df = pd.read_csv(
            news_path,
            sep='\t',
            names=columns,
            quoting=3,
        )

        logger.info(f"Loaded {len(df)} news articles from {news_path}")
        return df

    def parse_behaviors(self, behaviors_path: Path) -> pd.DataFrame:
        columns = ["impression_id", "user_id", "time", "history", "impressions"]

        df = pd.read_csv(
            behaviors_path,
            sep='\t',
            names=columns,
            quoting=3,
        )

        logger.info(f"Loaded {len(df)} behavior records from {behaviors_path}")
        return df

    def process_news(self, train_path: Path, dev_path: Path) -> Dict[str, dict]:
        train_news = self.parse_news(train_path / "news.tsv")
        dev_news = self.parse_news(dev_path / "news.tsv")

        all_news = pd.concat([train_news, dev_news]).drop_duplicates(subset="news_id")
        logger.info(f"Total unique news articles: {len(all_news)}")

        all_titles = all_news["title"].fillna("").tolist()
        all_abstracts = all_news["abstract"].fillna("").tolist()
        self.build_vocab(all_titles + all_abstracts)

        for cat in all_news["category"].dropna().unique():
            if cat not in self.category2idx:
                self.category2idx[cat] = len(self.category2idx)

        for subcat in all_news["subcategory"].dropna().unique():
            if subcat not in self.subcategory2idx:
                self.subcategory2idx[subcat] = len(self.subcategory2idx)

        logger.info(f"Categories: {len(self.category2idx)}, Subcategories: {len(self.subcategory2idx)}")

        for _, row in tqdm(all_news.iterrows(), total=len(all_news), desc="Processing news"):
            news_id = row["news_id"]
            self.news2idx[news_id] = len(self.news2idx)

            self.news_data[news_id] = {
                "news_idx": self.news2idx[news_id],
                "category": self.category2idx.get(row["category"], 0),
                "subcategory": self.subcategory2idx.get(row["subcategory"], 0),
                "title": self.encode_text(row["title"] or "", self.max_title_len),
                "abstract": self.encode_text(row["abstract"] or "", self.max_abstract_len),
            }

        return self.news_data

    def process_behaviors(
        self,
        behaviors_df: pd.DataFrame,
        is_train: bool = True
    ) -> List[dict]:
        samples = []

        for _, row in tqdm(behaviors_df.iterrows(), total=len(behaviors_df), desc="Processing behaviors"):
            user_id = row["user_id"]

            if user_id not in self.user2idx:
                self.user2idx[user_id] = len(self.user2idx)

            history = []
            if pd.notna(row["history"]):
                history_items = row["history"].split()
                history = [self.news2idx.get(nid, 0) for nid in history_items]
                history = history[-self.max_history_len:]

            history = history + [0] * (self.max_history_len - len(history))

            if pd.notna(row["impressions"]):
                impressions = row["impressions"].split()

                for imp in impressions:
                    parts = imp.split("-")
                    if len(parts) == 2:
                        news_id, label = parts
                        label = int(label)
                    else:
                        news_id = parts[0]
                        label = 0

                    news_idx = self.news2idx.get(news_id, 0)

                    if is_train:
                        if label == 1:
                            samples.append({
                                "user_idx": self.user2idx[user_id],
                                "user_id": user_id,
                                "history": history.copy(),
                                "news_idx": news_idx,
                                "news_id": news_id,
                                "label": label,
                            })
                    else:
                        samples.append({
                            "user_idx": self.user2idx[user_id],
                            "user_id": user_id,
                            "history": history.copy(),
                            "news_idx": news_idx,
                            "news_id": news_id,
                            "label": label,
                            "impression_id": row["impression_id"],
                        })

        return samples

    def preprocess(self, output_dir: str = "data/processed") -> dict:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        train_path = self.data_dir / "MINDsmall_train"
        dev_path = self.data_dir / "MINDsmall_dev"

        logger.info("Processing news articles...")
        self.process_news(train_path, dev_path)

        logger.info("Processing training behaviors...")
        train_behaviors = self.parse_behaviors(train_path / "behaviors.tsv")
        train_samples = self.process_behaviors(train_behaviors, is_train=True)

        logger.info("Processing dev behaviors...")
        dev_behaviors = self.parse_behaviors(dev_path / "behaviors.tsv")
        dev_samples = self.process_behaviors(dev_behaviors, is_train=False)

        logger.info("Saving processed data...")

        mappings = {
            "word2idx": self.word2idx,
            "category2idx": self.category2idx,
            "subcategory2idx": self.subcategory2idx,
            "news2idx": self.news2idx,
            "user2idx": self.user2idx,
        }

        with open(output_path / "mappings.json", "w") as f:
            json.dump(mappings, f)

        with open(output_path / "news_data.pkl", "wb") as f:
            pickle.dump(self.news_data, f)

        with open(output_path / "train_samples.pkl", "wb") as f:
            pickle.dump(train_samples, f)

        with open(output_path / "dev_samples.pkl", "wb") as f:
            pickle.dump(dev_samples, f)

        stats = {
            "vocab_size": len(self.word2idx),
            "num_categories": len(self.category2idx),
            "num_subcategories": len(self.subcategory2idx),
            "num_news": len(self.news2idx),
            "num_users": len(self.user2idx),
            "num_train_samples": len(train_samples),
            "num_dev_samples": len(dev_samples),
        }

        with open(output_path / "stats.json", "w") as f:
            json.dump(stats, f, indent=2)

        logger.info(f"Preprocessing complete. Stats: {stats}")

        return stats


def load_processed_data(data_dir: str = "data/processed") -> dict:
    data_path = Path(data_dir)

    with open(data_path / "mappings.json", "r") as f:
        mappings = json.load(f)

    with open(data_path / "news_data.pkl", "rb") as f:
        news_data = pickle.load(f)

    with open(data_path / "train_samples.pkl", "rb") as f:
        train_samples = pickle.load(f)

    with open(data_path / "dev_samples.pkl", "rb") as f:
        dev_samples = pickle.load(f)

    with open(data_path / "stats.json", "r") as f:
        stats = json.load(f)

    return {
        "mappings": mappings,
        "news_data": news_data,
        "train_samples": train_samples,
        "dev_samples": dev_samples,
        "stats": stats,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="data/mind")
    parser.add_argument("--output-dir", type=str, default="data/processed")
    parser.add_argument("--max-title-len", type=int, default=30)
    parser.add_argument("--max-abstract-len", type=int, default=100)
    parser.add_argument("--max-history-len", type=int, default=50)
    parser.add_argument("--min-word-freq", type=int, default=2)
    args = parser.parse_args()

    preprocessor = MINDPreprocessor(
        data_dir=args.data_dir,
        max_title_len=args.max_title_len,
        max_abstract_len=args.max_abstract_len,
        max_history_len=args.max_history_len,
        min_word_freq=args.min_word_freq,
    )

    stats = preprocessor.preprocess(args.output_dir)
    print(f"\nPreprocessing complete!")
    print(f"Stats: {json.dumps(stats, indent=2)}")
