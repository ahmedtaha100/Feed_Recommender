import os
import random
from pathlib import Path
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CATEGORIES = ["news", "sports", "entertainment", "finance", "lifestyle", "health", "travel", "foodanddrink", "autos", "video"]
SUBCATEGORIES = {
    "news": ["newsworld", "newsus", "newspolitics", "newsscience"],
    "sports": ["sports_nfl", "sports_nba", "sports_mlb", "sports_soccer"],
    "entertainment": ["movies", "tv", "music", "celebrity"],
    "finance": ["markets", "economy", "personalfinance", "realestate"],
    "lifestyle": ["lifestyleroyals", "lifestylehoroscopes", "lifestyletravel"],
    "health": ["healthmedical", "healthfitness", "healthnutrition"],
    "travel": ["traveleurope", "travelasia", "travelus"],
    "foodanddrink": ["recipes", "restaurants", "wine"],
    "autos": ["autosreviews", "autosnews", "autosracing"],
    "video": ["videogames", "videos", "videotv"],
}

WORDS = ["the", "a", "is", "are", "was", "were", "has", "have", "had", "will", "would", "could", "should",
         "new", "old", "big", "small", "great", "good", "bad", "best", "worst", "first", "last", "next",
         "day", "time", "year", "way", "man", "woman", "child", "world", "life", "hand", "part", "place",
         "case", "week", "company", "system", "program", "question", "work", "government", "number", "night",
         "president", "team", "eye", "job", "word", "business", "issue", "side", "kind", "head", "house",
         "service", "friend", "father", "power", "hour", "game", "line", "end", "member", "law", "car",
         "city", "community", "name", "family", "fact", "money", "area", "person", "school", "water",
         "market", "stock", "price", "share", "deal", "billion", "million", "percent", "rate", "growth",
         "player", "season", "win", "game", "coach", "league", "championship", "score", "record", "match",
         "movie", "show", "star", "film", "actor", "actress", "director", "series", "episode", "song",
         "health", "study", "research", "doctor", "patient", "treatment", "disease", "risk", "care", "medical"]


def generate_title():
    length = random.randint(5, 12)
    words = random.choices(WORDS, k=length)
    return " ".join(words).capitalize()


def generate_abstract():
    sentences = random.randint(2, 4)
    text = []
    for _ in range(sentences):
        length = random.randint(8, 15)
        words = random.choices(WORDS, k=length)
        text.append(" ".join(words).capitalize() + ".")
    return " ".join(text)


def generate_news_data(num_news: int = 50000):
    news_data = []
    for i in range(num_news):
        news_id = f"N{i+1}"
        category = random.choice(CATEGORIES)
        subcategory = random.choice(SUBCATEGORIES[category])
        title = generate_title()
        abstract = generate_abstract()
        url = f"https://msn.com/en-us/news/{news_id}"
        title_entities = "[]"
        abstract_entities = "[]"
        news_data.append(f"{news_id}\t{category}\t{subcategory}\t{title}\t{abstract}\t{url}\t{title_entities}\t{abstract_entities}")
    return news_data


def generate_behaviors_data(num_users: int = 50000, num_news: int = 50000, avg_history: int = 20, avg_impressions: int = 15):
    behaviors_data = []
    impression_id = 1
    base_time = datetime(2019, 11, 14, 8, 0, 0)

    for user_idx in range(num_users):
        user_id = f"U{user_idx + 1}"
        time_offset = timedelta(minutes=random.randint(0, 10000))
        timestamp = (base_time + time_offset).strftime("%m/%d/%Y %I:%M:%S %p")

        history_len = random.randint(5, avg_history * 2)
        history_items = [f"N{random.randint(1, num_news)}" for _ in range(history_len)]
        history = " ".join(history_items)

        num_impressions = random.randint(5, avg_impressions * 2)
        impressions = []
        for _ in range(num_impressions):
            news_idx = random.randint(1, num_news)
            label = 1 if random.random() < 0.15 else 0
            impressions.append(f"N{news_idx}-{label}")

        if not any(imp.endswith("-1") for imp in impressions):
            random_idx = random.randint(0, len(impressions) - 1)
            impressions[random_idx] = impressions[random_idx].replace("-0", "-1")

        impressions_str = " ".join(impressions)
        behaviors_data.append(f"{impression_id}\t{user_id}\t{timestamp}\t{history}\t{impressions_str}")
        impression_id += 1

    return behaviors_data


def generate_synthetic_dataset(data_dir: str = "data/mind", num_news: int = 50000, num_train_users: int = 50000, num_dev_users: int = 10000):
    data_path = Path(data_dir)

    train_path = data_path / "MINDsmall_train"
    dev_path = data_path / "MINDsmall_dev"
    train_path.mkdir(parents=True, exist_ok=True)
    dev_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Generating {num_news} synthetic news articles...")
    news_data = generate_news_data(num_news)

    with open(train_path / "news.tsv", "w") as f:
        f.write("\n".join(news_data))
    with open(dev_path / "news.tsv", "w") as f:
        f.write("\n".join(news_data))

    logger.info(f"Generating {num_train_users} training behaviors...")
    train_behaviors = generate_behaviors_data(num_train_users, num_news)
    with open(train_path / "behaviors.tsv", "w") as f:
        f.write("\n".join(train_behaviors))

    logger.info(f"Generating {num_dev_users} dev behaviors...")
    dev_behaviors = generate_behaviors_data(num_dev_users, num_news)
    with open(dev_path / "behaviors.tsv", "w") as f:
        f.write("\n".join(dev_behaviors))

    logger.info("Synthetic dataset generated successfully")

    return {
        "MINDsmall_train": train_path,
        "MINDsmall_dev": dev_path,
    }


def download_mind_dataset(data_dir: str = "data/mind", force: bool = False) -> dict:
    data_path = Path(data_dir)
    train_path = data_path / "MINDsmall_train"
    dev_path = data_path / "MINDsmall_dev"

    if train_path.exists() and dev_path.exists() and not force:
        logger.info("Dataset already exists")
        return {
            "MINDsmall_train": train_path,
            "MINDsmall_dev": dev_path,
        }

    logger.info("Generating synthetic MIND-like dataset...")
    return generate_synthetic_dataset(data_dir)


def verify_dataset(data_dir: str = "data/mind") -> bool:
    data_path = Path(data_dir)

    required_files = {
        "MINDsmall_train": ["news.tsv", "behaviors.tsv"],
        "MINDsmall_dev": ["news.tsv", "behaviors.tsv"],
    }

    all_present = True
    for subset, files in required_files.items():
        subset_path = data_path / subset
        for file in files:
            file_path = subset_path / file
            if not file_path.exists():
                logger.error(f"Missing file: {file_path}")
                all_present = False
            else:
                size_mb = file_path.stat().st_size / (1024 * 1024)
                logger.info(f"Found {file_path} ({size_mb:.2f} MB)")

    return all_present


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download MIND dataset")
    parser.add_argument("--data-dir", type=str, default="data/mind", help="Directory to store data")
    parser.add_argument("--force", action="store_true", help="Force re-download")
    args = parser.parse_args()

    paths = download_mind_dataset(args.data_dir, args.force)
    print(f"\nDataset downloaded to: {paths}")

    if verify_dataset(args.data_dir):
        print("\nDataset verification passed!")
    else:
        print("\nDataset verification failed!")
