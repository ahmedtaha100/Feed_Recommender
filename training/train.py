import os
import sys
import json
import argparse
from pathlib import Path
import logging

import yaml
import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.download import download_mind_dataset, verify_dataset
from data.preprocessing import MINDPreprocessor, load_processed_data
from data.dataset import create_data_loaders
from models.two_tower import create_model_from_config
from training.trainer import Trainer, WarmupCosineScheduler, extract_item_embeddings, get_device

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main(args):
    config = load_config(args.config)

    device = get_device()
    logger.info(f"Using device: {device}")

    data_config = config["data"]

    if not Path(data_config["train_dir"]).exists():
        logger.info("Downloading MIND dataset...")
        download_mind_dataset(data_config["data_dir"])

    processed_dir = "data/processed"
    if not Path(processed_dir).exists() or args.reprocess:
        logger.info("Preprocessing data...")
        preprocessor = MINDPreprocessor(
            data_dir=data_config["data_dir"],
            max_title_len=data_config["max_title_len"],
            max_abstract_len=data_config["max_abstract_len"],
            max_history_len=data_config["max_history_len"],
            min_word_freq=data_config["min_word_freq"],
        )
        preprocessor.preprocess(processed_dir)

    logger.info("Loading processed data...")
    data = load_processed_data(processed_dir)

    config["model"]["vocab_size"] = data["stats"]["vocab_size"]
    config["model"]["num_categories"] = data["stats"]["num_categories"]
    config["model"]["num_subcategories"] = data["stats"]["num_subcategories"]

    logger.info("Creating data loaders...")
    train_loader, eval_loader = create_data_loaders(
        train_samples=data["train_samples"],
        dev_samples=data["dev_samples"][:10000],
        news_data=data["news_data"],
        batch_size=config["training"]["batch_size"],
        num_negatives=data_config["train_neg_samples"],
        num_workers=0,
    )

    logger.info("Creating model...")
    model = create_model_from_config(config)

    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {num_params:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )

    total_steps = len(train_loader) * config["training"]["num_epochs"]
    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_steps=config["training"]["warmup_steps"],
        total_steps=total_steps,
    )

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        mixed_precision=config["training"]["mixed_precision"] and device.type == "cuda",
        gradient_accumulation_steps=config["training"]["gradient_accumulation_steps"],
        max_grad_norm=config["training"]["max_grad_norm"],
        checkpoint_dir=config["training"]["checkpoint_dir"],
        log_interval=config["training"]["log_interval"],
    )

    if args.resume:
        trainer.load_checkpoint(args.resume)

    logger.info("Starting training...")
    history = trainer.train(
        train_loader=train_loader,
        eval_loader=eval_loader,
        num_epochs=config["training"]["num_epochs"],
        early_stopping_patience=config["training"]["early_stopping_patience"],
    )

    logger.info("Extracting item embeddings...")
    trainer.load_checkpoint("best_model.pt")
    embeddings, idx_mapping = extract_item_embeddings(
        model=model,
        news_data=data["news_data"],
        device=device,
        batch_size=512,
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

    logger.info("Training complete!")
    logger.info(f"Item embeddings shape: {embeddings.shape}")
    logger.info(f"Artifacts saved to {artifacts_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/config.yaml")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--reprocess", action="store_true")
    args = parser.parse_args()

    main(args)
