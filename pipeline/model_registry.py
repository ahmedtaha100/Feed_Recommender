import os
import json
import shutil
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelRegistry:
    def __init__(self, registry_path: str = "model_registry"):
        self.registry_path = Path(registry_path)
        self.registry_path.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.registry_path / "registry.json"
        self.metadata = self._load_metadata()

    def _load_metadata(self) -> Dict:
        if self.metadata_file.exists():
            with open(self.metadata_file, "r") as f:
                return json.load(f)
        return {"models": [], "production": None, "staging": None}

    def _save_metadata(self) -> None:
        with open(self.metadata_file, "w") as f:
            json.dump(self.metadata, f, indent=2)

    def _compute_checksum(self, file_path: Path) -> str:
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    def register_model(
        self,
        model_path: str,
        metrics: Dict[str, float],
        config: Dict,
        tags: Optional[List[str]] = None,
    ) -> str:
        model_file = Path(model_path)
        if not model_file.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        version = f"v{len(self.metadata['models']) + 1}"
        timestamp = datetime.now().isoformat()
        checksum = self._compute_checksum(model_file)

        version_dir = self.registry_path / version
        version_dir.mkdir(exist_ok=True)

        shutil.copy(model_file, version_dir / "model.pt")

        model_entry = {
            "version": version,
            "timestamp": timestamp,
            "checksum": checksum,
            "metrics": metrics,
            "config": config,
            "tags": tags or [],
            "status": "registered",
        }

        self.metadata["models"].append(model_entry)
        self._save_metadata()

        logger.info(f"Registered model {version} with metrics: {metrics}")
        return version

    def promote_to_staging(self, version: str) -> bool:
        for model in self.metadata["models"]:
            if model["version"] == version:
                self.metadata["staging"] = version
                model["status"] = "staging"
                self._save_metadata()
                logger.info(f"Promoted {version} to staging")
                return True
        return False

    def promote_to_production(self, version: str) -> bool:
        for model in self.metadata["models"]:
            if model["version"] == version:
                old_prod = self.metadata["production"]
                if old_prod:
                    for m in self.metadata["models"]:
                        if m["version"] == old_prod:
                            m["status"] = "archived"

                self.metadata["production"] = version
                model["status"] = "production"
                self._save_metadata()
                logger.info(f"Promoted {version} to production (archived {old_prod})")
                return True
        return False

    def get_production_model(self) -> Optional[Dict]:
        prod_version = self.metadata.get("production")
        if not prod_version:
            return None
        for model in self.metadata["models"]:
            if model["version"] == prod_version:
                return model
        return None

    def get_model_path(self, version: str) -> Optional[Path]:
        model_path = self.registry_path / version / "model.pt"
        if model_path.exists():
            return model_path
        return None

    def compare_models(self, version_a: str, version_b: str) -> Dict:
        model_a = None
        model_b = None
        for model in self.metadata["models"]:
            if model["version"] == version_a:
                model_a = model
            if model["version"] == version_b:
                model_b = model

        if not model_a or not model_b:
            return {}

        comparison = {"version_a": version_a, "version_b": version_b, "metrics_diff": {}}

        for metric in model_a["metrics"]:
            if metric in model_b["metrics"]:
                diff = model_b["metrics"][metric] - model_a["metrics"][metric]
                comparison["metrics_diff"][metric] = {
                    "a": model_a["metrics"][metric],
                    "b": model_b["metrics"][metric],
                    "diff": diff,
                    "improved": diff > 0,
                }

        return comparison

    def list_models(self) -> List[Dict]:
        return self.metadata["models"]

    def rollback_production(self) -> bool:
        models = sorted(
            [m for m in self.metadata["models"] if m["status"] == "archived"],
            key=lambda x: x["timestamp"],
            reverse=True,
        )
        if models:
            return self.promote_to_production(models[0]["version"])
        return False
