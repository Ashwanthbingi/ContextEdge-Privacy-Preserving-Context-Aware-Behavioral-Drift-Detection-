from __future__ import annotations

import shutil
from pathlib import Path

from contextedge.storage.database import Database

PRIVACY_NOTICE = (
    "All telemetry, feature processing, model training and inference stay on this laptop. "
    "Stored observations contain numerical resource features, category proportions, context labels and detection decisions. "
    "Executable basenames are categorized in memory; process identities are never persisted. "
    "No input, screen, content, URLs, addresses, destinations, command lines or full process paths are collected."
)


class PrivacyManager:
    def __init__(self, database: Database, model_root: Path):
        self.database, self.model_root = database, Path(model_root).resolve()

    def delete(self, include_models=False):
        if include_models:
            # Only the verified model directory owned by this database may be removed.
            expected = (self.database.path.parent / "profiles").resolve()
            if self.model_root != expected or not self.model_root.is_relative_to(
                self.database.path.parent.resolve()
            ):
                raise ValueError("Refusing deletion outside the application model directory")
            if self.model_root.exists():
                shutil.rmtree(self.model_root)
            self.model_root.mkdir()
        self.database.clear(include_profiles=include_models)
