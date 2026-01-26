#!/usr/bin/env python3
import os
import sys
import json
import time
import signal
import subprocess
import argparse
from pathlib import Path
from typing import Optional
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_dependencies():
    logger.info("Checking dependencies...")
    try:
        import torch
        import faiss
        import lightgbm
        import fastapi
        import yaml
        import numpy
        import pandas
        logger.info("All dependencies available")
        return True
    except ImportError as e:
        logger.error(f"Missing dependency: {e}")
        logger.info("Run: pip install -r requirements.txt")
        return False


def download_data():
    logger.info("Downloading MIND dataset...")
    from data.download import download_mind_dataset, verify_dataset

    paths = download_mind_dataset("data/mind")

    if verify_dataset("data/mind"):
        logger.info("Dataset downloaded and verified")
        return True
    else:
        logger.error("Dataset verification failed")
        return False


def preprocess_data():
    logger.info("Preprocessing data...")
    from data.preprocessing import MINDPreprocessor

    preprocessor = MINDPreprocessor(
        data_dir="data/mind",
        max_title_len=30,
        max_abstract_len=100,
        max_history_len=50,
        min_word_freq=2,
    )

    stats = preprocessor.preprocess("data/processed")
    logger.info(f"Preprocessing complete: {stats}")
    return True


def train_model(num_epochs: int = 3, batch_size: int = 128):
    logger.info("Training model...")
    import yaml
    import torch
    import numpy as np

    from data.preprocessing import load_processed_data
    from data.dataset import create_data_loaders
    from models.two_tower import create_model_from_config
    from training.trainer import Trainer, WarmupCosineScheduler, extract_item_embeddings, get_device

    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    data = load_processed_data("data/processed")

    config["model"]["vocab_size"] = data["stats"]["vocab_size"]
    config["model"]["num_categories"] = data["stats"]["num_categories"]
    config["model"]["num_subcategories"] = data["stats"]["num_subcategories"]
    config["training"]["num_epochs"] = num_epochs
    config["training"]["batch_size"] = batch_size

    device = get_device()
    logger.info(f"Training on device: {device}")

    train_loader, eval_loader = create_data_loaders(
        train_samples=data["train_samples"][:50000],
        dev_samples=data["dev_samples"][:5000],
        news_data=data["news_data"],
        batch_size=batch_size,
        num_negatives=4,
        num_workers=0,
    )

    model = create_model_from_config(config)
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )

    total_steps = len(train_loader) * num_epochs
    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_steps=min(500, total_steps // 10),
        total_steps=total_steps,
    )

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        mixed_precision=device.type == "cuda",
        checkpoint_dir="checkpoints",
        log_interval=50,
    )

    history = trainer.train(
        train_loader=train_loader,
        eval_loader=eval_loader,
        num_epochs=num_epochs,
        early_stopping_patience=2,
    )

    logger.info("Extracting item embeddings...")
    trainer.load_checkpoint("best_model.pt")

    embeddings, idx_mapping = extract_item_embeddings(
        model=model,
        news_data=data["news_data"],
        device=device,
        batch_size=256,
    )

    artifacts_dir = Path("artifacts")
    artifacts_dir.mkdir(exist_ok=True)

    np.save(artifacts_dir / "item_embeddings.npy", embeddings)
    with open(artifacts_dir / "item_idx_mapping.json", "w") as f:
        json.dump({str(k): v for k, v in idx_mapping.items()}, f)

    torch.save(model.state_dict(), artifacts_dir / "two_tower_model.pt")

    with open(artifacts_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    with open(artifacts_dir / "model_config.json", "w") as f:
        json.dump(config, f, indent=2)

    logger.info(f"Training complete. Embeddings shape: {embeddings.shape}")
    return True


def build_faiss_index():
    logger.info("Building FAISS index...")
    from retrieval.index_builder import build_index_from_embeddings

    builder = build_index_from_embeddings(
        embeddings_path="artifacts/item_embeddings.npy",
        idx_mapping_path="artifacts/item_idx_mapping.json",
        output_path="artifacts/faiss_index",
        index_type="IVF",
        nlist=100,
        nprobe=10,
    )

    logger.info(f"Index built with {builder.index.ntotal} items")
    return True


def run_evaluation():
    logger.info("Running evaluation...")
    from evaluation.metrics import evaluate_retrieval

    results = evaluate_retrieval(
        model_path="artifacts/two_tower_model.pt",
        index_path="artifacts/faiss_index",
        data_path="data/processed",
        top_k=[10, 50, 100],
        num_samples=500,
    )

    logger.info("Evaluation Results:")
    for metric, value in results.items():
        logger.info(f"  {metric}: {value:.4f}")

    with open("artifacts/evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


def run_benchmark():
    logger.info("Running benchmarks...")
    from evaluation.benchmark import benchmark_retrieval, benchmark_model_inference

    results = {}

    results["retrieval"] = benchmark_retrieval(
        index_path="artifacts/faiss_index",
        embeddings_path="artifacts/item_embeddings.npy",
        num_queries=1000,
        top_k=100,
        batch_sizes=[1, 8, 32],
    )

    results["model_inference"] = benchmark_model_inference(
        model_path="artifacts/two_tower_model.pt",
        config_path="artifacts/model_config.json",
        num_samples=500,
        batch_sizes=[1, 8, 32],
    )

    logger.info("\nBenchmark Results:")
    logger.info("Retrieval (batch_size=1):")
    logger.info(f"  Mean: {results['retrieval']['batch_1']['per_query_mean_ms']:.2f}ms")
    logger.info(f"  P99:  {results['retrieval']['batch_1']['per_query_p99_ms']:.2f}ms")

    logger.info("Model Inference (batch_size=1):")
    logger.info(f"  Mean: {results['model_inference']['batch_1']['per_sample_mean_ms']:.2f}ms")
    logger.info(f"  P99:  {results['model_inference']['batch_1']['per_sample_p99_ms']:.2f}ms")

    with open("artifacts/benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


def start_server(port: int = 8000):
    logger.info(f"Starting API server on port {port}...")
    import uvicorn
    from serving.app import app

    uvicorn.run(app, host="0.0.0.0", port=port)


def demo_recommendations():
    logger.info("Running demo recommendations...")
    import pickle
    import torch
    import numpy as np
    from models.two_tower import create_model_from_config
    from retrieval.retriever import Retriever
    from training.trainer import get_device

    device = get_device()

    with open("artifacts/model_config.json", "r") as f:
        config = json.load(f)

    model = create_model_from_config(config)
    from training.trainer import load_checkpoint_safe
    checkpoint = load_checkpoint_safe("artifacts/two_tower_model.pt", device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()

    retriever = Retriever(
        index_path="artifacts/faiss_index",
        item_embeddings_path="artifacts/item_embeddings.npy",
    )

    with open("data/processed/news_data.pkl", "rb") as f:
        news_data = pickle.load(f)

    with open("data/processed/dev_samples.pkl", "rb") as f:
        dev_samples = pickle.load(f)

    with open("data/processed/mappings.json", "r") as f:
        mappings = json.load(f)

    idx2news = {v: k for k, v in mappings["news2idx"].items()}

    sample = dev_samples[0]
    history = sample["history"]

    logger.info("\nDemo: Getting recommendations for a user")
    logger.info(f"User history length: {sum(1 for h in history if h > 0)} items")

    history_titles = []
    history_categories = []

    for idx in history:
        if idx > 0 and idx in idx2news:
            news_id = idx2news[idx]
            news = news_data.get(news_id, {})
            history_titles.append(news.get("title", [0] * 30))
            history_categories.append(news.get("category", 0))
        else:
            history_titles.append([0] * 30)
            history_categories.append(0)

    history_title = torch.tensor([history_titles], dtype=torch.long).to(device)
    history_category = torch.tensor([history_categories], dtype=torch.long).to(device)
    history_mask = torch.tensor(
        [[1.0 if idx > 0 else 0.0 for idx in history]], dtype=torch.float
    ).to(device)

    start_time = time.perf_counter()

    with torch.no_grad():
        user_emb = model.encode_user(
            history_title=history_title,
            history_category=history_category,
            history_mask=history_mask,
        ).cpu().numpy()

    scores, item_ids = retriever.retrieve(user_emb, top_k=10)

    latency_ms = (time.perf_counter() - start_time) * 1000

    logger.info(f"\nTop 10 recommendations (latency: {latency_ms:.2f}ms):")
    for rank, (item_id, score) in enumerate(zip(item_ids[0], scores[0]), 1):
        if item_id >= 0 and item_id in idx2news:
            news_id = idx2news[item_id]
            logger.info(f"  {rank}. {news_id} (score: {score:.4f})")

    return True


def main():
    parser = argparse.ArgumentParser(description="Two-Tower Feed Recommender Demo")
    parser.add_argument("--skip-download", action="store_true", help="Skip data download")
    parser.add_argument("--skip-preprocess", action="store_true", help="Skip preprocessing")
    parser.add_argument("--skip-train", action="store_true", help="Skip training")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Training batch size")
    parser.add_argument("--serve", action="store_true", help="Start API server after demo")
    parser.add_argument("--port", type=int, default=8000, help="API server port")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Two-Tower Feed Recommender - Full Pipeline Demo")
    logger.info("=" * 60)

    if not check_dependencies():
        sys.exit(1)

    if not args.skip_download:
        if not Path("data/mind/MINDsmall_train").exists():
            if not download_data():
                sys.exit(1)
        else:
            logger.info("Data already exists, skipping download")

    if not args.skip_preprocess:
        if not Path("data/processed/train_samples.pkl").exists():
            if not preprocess_data():
                sys.exit(1)
        else:
            logger.info("Processed data exists, skipping preprocessing")

    if not args.skip_train:
        if not Path("artifacts/two_tower_model.pt").exists():
            if not train_model(num_epochs=args.epochs, batch_size=args.batch_size):
                sys.exit(1)
        else:
            logger.info("Model exists, skipping training")

    if not Path("artifacts/faiss_index").exists():
        if not build_faiss_index():
            sys.exit(1)
    else:
        logger.info("FAISS index exists, skipping build")

    run_evaluation()
    run_benchmark()
    demo_recommendations()

    logger.info("\n" + "=" * 60)
    logger.info("Demo Complete!")
    logger.info("=" * 60)
    logger.info("\nArtifacts saved to ./artifacts/")
    logger.info("To start the API server: python -m serving.app")

    if args.serve:
        logger.info(f"\nStarting API server on port {args.port}...")
        start_server(args.port)


if __name__ == "__main__":
    main()
