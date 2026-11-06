from __future__ import annotations

import hashlib
import io
import importlib.metadata
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np

from contextedge.config import Context, Settings
from contextedge.detection.anomaly_scoring import CalibratedFusion, empirical_pvalue, evidence
from contextedge.detection.isolation_forest import IsolationDetector
from contextedge.detection.temporal_model import export_onnx, train_temporal
from contextedge.features.extraction import FEATURE_NAMES, FEATURE_SCHEMA
from contextedge.features.preprocessing import fit_scaler, sequences
from contextedge.inference.onnx_engine import ONNXEngine, verify_parity
from contextedge.profiles.calibration import split_calibration
from contextedge.storage.database import encode
from contextedge.storage.repositories import Repository


@dataclass
class Profile:
    context: str
    version: int
    scaler: object
    isolation: IsolationDetector
    temporal: ONNXEngine
    fusion: CalibratedFusion
    metadata: dict

    def score(self, vector, sequence, alpha):
        normalized = self.scaler.transform(np.asarray(vector, np.float32).reshape(1, -1))
        if_score = self.isolation.score(normalized)[0]
        temporal_score = None
        if len(sequence) == self.metadata["sequence_length"]:
            temporal_score = float(self.temporal.scores([self.scaler.transform(sequence)])[0])
        return self.fusion.score(if_score, temporal_score, alpha)


class ProfileManager:
    def __init__(self, root: Path, repository: Repository, settings: Settings):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.repo = repository
        self.settings = settings
        self.cache: dict[str, Profile] = {}
        self.errors: dict[str, str] = {}

    def _path(self, context, version):
        return self.root / Context(context).name.lower() / f"v{int(version)}"

    def active(self, context) -> Profile | None:
        context = str(context)
        rows = self.repo.db.rows("SELECT active_version FROM profiles WHERE context=?", (context,))
        if not rows or rows[0]["active_version"] is None:
            return None
        version = rows[0]["active_version"]
        if context in self.cache and self.cache[context].version == version:
            return self.cache[context]
        try:
            profile = self.load(context, version)
            self.cache[context] = profile
            self.errors.pop(context, None)
            return profile
        except (OSError, ValueError, RuntimeError, EOFError, KeyError, ImportError, TypeError) as exc:
            self.errors[context] = f"Profile unavailable ({type(exc).__name__}); recalibrate or roll back"
            return None
        except Exception as exc:
            # Native runtime and serializer failures must not become fabricated scores.
            self.errors[context] = f"Profile load failed ({type(exc).__name__}); recalibrate or roll back"
            return None

    def load(self, context, version) -> Profile:
        path = self._path(context, version)
        if not path.resolve().is_relative_to(self.root):
            raise ValueError("Invalid profile path")
        metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
        if metadata["feature_schema"] != FEATURE_SCHEMA or metadata["features"] != list(FEATURE_NAMES):
            raise ValueError("Incompatible feature schema")
        contents = {}
        for name in ("model.joblib", "temporal.onnx"):
            payload = (path / name).read_bytes()
            if hashlib.sha256(payload).hexdigest() != metadata["sha256"][name]:
                raise ValueError("Model checksum failed")
            contents[name] = payload
        # Only self-produced local artifacts are accepted. Never load downloaded joblib files.
        bundle = joblib.load(io.BytesIO(contents["model.joblib"]))
        return Profile(
            str(context),
            version,
            bundle["scaler"],
            bundle["isolation"],
            ONNXEngine(contents["temporal.onnx"]),
            bundle["fusion"],
            metadata,
        )

    def train(self, context, rows=None, candidate=False) -> Profile:
        context = str(Context(context))
        rows = self.repo.calibration(context) if rows is None else rows
        start = time.perf_counter()
        train, validation, train_segments, val_segments = split_calibration(rows, self.settings)
        scaler = fit_scaler(train)
        normalized_train = scaler.transform(train).astype(np.float32)
        normalized_val = scaler.transform(validation).astype(np.float32)
        train_seq = sequences(normalized_train, self.settings.sequence_length, train_segments)
        val_seq = sequences(normalized_val, self.settings.sequence_length, val_segments)
        isolation = IsolationDetector(self.settings.random_seed).fit(normalized_train)
        model, training = train_temporal(
            train_seq,
            hidden=self.settings.temporal_hidden,
            epochs=self.settings.temporal_epochs,
            seed=self.settings.random_seed,
        )
        onnx_bytes = export_onnx(model, self.settings.sequence_length, len(FEATURE_NAMES))
        temporal = ONNXEngine(onnx_bytes)
        parity = verify_parity(model, temporal, val_seq)
        if_reference = isolation.score(train_seq[:, -1])
        temporal_reference = temporal.scores(train_seq)
        if_val = isolation.score(val_seq[:, -1])
        temporal_val = temporal.scores(val_seq)
        weight = self.settings.temporal_weight
        fused_val = (1 - weight) * evidence(if_reference, if_val) + weight * evidence(
            temporal_reference, temporal_val
        )
        fusion = CalibratedFusion(
            if_reference, temporal_reference, fused_val, isolation.score(normalized_val), weight
        )
        previous = self.repo.versions(context)
        version = max([r["version"] for r in previous], default=0) + 1
        # Recover safely from an orphaned directory left before a database commit.
        while self._path(context, version).exists():
            version += 1
        metadata = {
            "context": context,
            "version": version,
            "feature_schema": FEATURE_SCHEMA,
            "features": list(FEATURE_NAMES),
            "sequence_length": self.settings.sequence_length,
            "settings": self.settings.to_dict(),
            "library_versions": {
                name: importlib.metadata.version(name)
                for name in ("scikit-learn", "torch", "onnx", "onnxruntime", "numpy", "river")
            },
            "feature_window": self.settings.feature_window,
            "created_at": time.time(),
            "observations": len(rows),
            "train_observations": len(train),
            "validation_observations": len(validation),
            "training_sequences": len(train_seq),
            "validation_sequences": len(val_seq),
            "alpha": self.settings.false_alarm_alpha,
            "threshold": 1 - self.settings.false_alarm_alpha,
            "temporal_weight": weight,
            "origins": sorted(set(r["origin"] for r in rows)),
            "training": training,
            "onnx_parity": parity,
            "baseline_mean": dict(zip(FEATURE_NAMES, train.mean(0).tolist(), strict=True)),
            "baseline_std": dict(zip(FEATURE_NAMES, train.std(0).tolist(), strict=True)),
            "reference_scores": [1 - empirical_pvalue(fused_val, float(s)) for s in fused_val],
            "training_seconds": time.perf_counter() - start,
        }
        buffer = io.BytesIO()
        joblib.dump({"scaler": scaler, "isolation": isolation, "fusion": fusion}, buffer, compress=3)
        payloads = {"model.joblib": buffer.getvalue(), "temporal.onnx": onnx_bytes}
        # Keep native weights for reproducible PyTorch inference/parity inspection.
        import torch

        weights = io.BytesIO()
        torch.save(model.state_dict(), weights)
        payloads["temporal.pt"] = weights.getvalue()
        metadata["sha256"] = {k: hashlib.sha256(v).hexdigest() for k, v in payloads.items()}
        metadata["model_bytes"] = sum(len(v) for v in payloads.values())
        metadata["onnx_bytes"] = len(onnx_bytes)
        path = self._path(context, version)
        path.mkdir(parents=True)
        for name, content in payloads.items():
            temp = path / (name + ".tmp")
            temp.write_bytes(content)
            os.replace(temp, path / name)
        (path / "metadata.json").write_text(json.dumps(metadata, indent=2, allow_nan=False), encoding="utf-8")
        with self.repo.db.connect() as conn:
            conn.execute("INSERT OR IGNORE INTO profiles VALUES (?,NULL,?)", (context, time.time()))
            conn.execute(
                "INSERT INTO model_versions VALUES (?,?,?,?,?,?)",
                (
                    context,
                    version,
                    "candidate" if candidate else "ready",
                    time.time(),
                    str(path.relative_to(self.root)),
                    encode(metadata),
                ),
            )
        profile = Profile(context, version, scaler, isolation, temporal, fusion, metadata)
        if not candidate:
            self.activate(context, version)
        return profile

    def activate(self, context, version):
        profile = self.load(context, version)
        with self.repo.db.connect() as conn:
            if not conn.execute(
                "SELECT 1 FROM model_versions WHERE context=? AND version=?", (str(context), version)
            ).fetchone():
                raise ValueError("Version is not registered")
            conn.execute(
                "UPDATE model_versions SET status='archived' WHERE context=? AND status='active'",
                (str(context),),
            )
            conn.execute(
                "UPDATE model_versions SET status='active' WHERE context=? AND version=?",
                (str(context), version),
            )
            conn.execute(
                "UPDATE profiles SET active_version=?,updated_at=? WHERE context=?",
                (version, time.time(), str(context)),
            )
        self.cache[str(context)] = profile
        self.errors.pop(str(context), None)

    def rollback(self, context, version):
        self.activate(context, version)
        self.repo.db.system_event(
            "rollback", f"{Context(context).value}: activated profile version {version}"
        )
