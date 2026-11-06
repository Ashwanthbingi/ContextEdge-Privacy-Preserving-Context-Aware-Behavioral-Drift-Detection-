from __future__ import annotations

import sqlite3
import threading
import time
import uuid
from collections import deque
from pathlib import Path

from contextedge.adaptation.baseline_manager import BaselineManager
from contextedge.collector.telemetry import TelemetryCollector, TelemetrySample
from contextedge.config import Settings, default_data_dir
from contextedge.context.mappings import DEFAULT_MAPPINGS, validate_mappings
from contextedge.context.recognition import ContextRecognizer
from contextedge.detection.state_machine import DRIFT, NORMAL, TEMPORARY, WARMUP, PersistenceTracker
from contextedge.features.extraction import FeatureExtractor
from contextedge.privacy.privacy_manager import PrivacyManager
from contextedge.profiles.manager import ProfileManager
from contextedge.storage.database import Database
from contextedge.storage.locking import MonitorLease
from contextedge.storage.repositories import Repository


class ContextEdgeEngine:
    """Single local writer shared by CLI and dashboard. Controls are thread-safe."""

    def __init__(self, data_dir: Path | None = None, settings: Settings | None = None):
        self.data_dir = Path(data_dir or default_data_dir()).resolve()
        self.db = Database(self.data_dir / "contextedge.sqlite3")
        self._lease = MonitorLease(self.data_dir / ".monitor.lock")
        self.settings = (
            settings.validate() if settings else Settings.from_dict(self.db.get_setting("settings", {}))
        )
        self.repo = Repository(self.db)
        mappings = validate_mappings(self.db.get_setting("mappings", DEFAULT_MAPPINGS))
        self.collector = TelemetryCollector(mappings, self.settings.foreground_enabled)
        self.recognizer = ContextRecognizer()
        self.features = FeatureExtractor(self.settings.feature_window)
        self.profiles = ProfileManager(self.data_dir / "profiles", self.repo, self.settings)
        self.adaptation = BaselineManager(self.profiles, self.settings)
        self.privacy = PrivacyManager(self.db, self.data_dir / "profiles")
        self.manual_context = self.db.get_setting("manual_context")
        self.calibrating = False  # Consent to calibration never silently survives restart.
        self._segment = uuid.uuid4().hex
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread = None
        self._context = None
        self._last_timestamp = None
        self._sequence = deque(maxlen=self.settings.sequence_length)
        self._trackers: dict[str, tuple[int, PersistenceTracker]] = {}
        self._grace = 0
        self._event_id = None
        self._event_record = None
        self.latest = None
        self.error = None
        self.started_at = None
        self._retained_at = 0.0

    @property
    def running(self):
        return self._thread is not None and self._thread.is_alive() and not self._stop.is_set()

    def start(self):
        with self._lock:
            if self.running:
                return
            if self._thread is not None and self._thread.is_alive():
                raise RuntimeError("Monitoring worker is still stopping")
            self._lease.acquire()
            # Close interrupted episodes only after securing exclusive monitoring access.
            try:
                self.db.execute(
                    "UPDATE events SET ended_at=updated_at,explanation=explanation || ' Monitoring interrupted.' WHERE ended_at IS NULL"
                )
                self.collector.start()
            except Exception:
                self._lease.release()
                raise
            self._context = None
            self._last_timestamp = None
            self._trackers.clear()
            self._sequence.clear()
            self.features.reset()
            self._stop.clear()
            self.error = None
            self.started_at = time.time()
            self._thread = threading.Thread(target=self._run, daemon=True, name="ContextEdgeMonitor")
            self._thread.start()

    def stop(self):
        self._stop.set()
        thread = self._thread
        if thread and thread is not threading.current_thread():
            thread.join(timeout=30)
            if thread.is_alive():
                raise RuntimeError("Monitoring still completing an operation; wait before changing settings")
        with self._lock:
            self.collector.stop()
            self._close_event("Monitoring stopped")
            for context in list(self.adaptation.pending):
                self.adaptation.reset_collection(context, "Monitoring stopped; sequence continuity ended")

    def _run(self):
        try:
            deadline = time.monotonic() + self.settings.sampling_interval
            while not self._stop.wait(max(0.0, deadline - time.monotonic())):
                self.process(self.collector.sample())
                deadline += self.settings.sampling_interval
                if deadline < time.monotonic():
                    # Skip missed deadlines rather than spin or collect bursts.
                    deadline = time.monotonic() + self.settings.sampling_interval
                if time.time() - self._retained_at >= 3600:
                    self.db.retain(self.settings.retention_days)
                    self._retained_at = time.time()
        except Exception as exc:
            self.error = f"Monitoring stopped: {type(exc).__name__}. Check storage availability and local configuration."
            self._stop.set()
            self.collector.stop()
            try:
                self.db.system_event("monitoring_error", self.error)
            except sqlite3.Error:
                pass
        finally:
            try:
                with self._lock:
                    self._close_event("Monitoring stopped or interrupted")
                    self.collector.stop()
            except sqlite3.Error:
                pass
            self._lease.release()

    def set_context(self, context):
        from contextedge.config import Context

        if context is not None:
            context = str(Context(context))
        with self._lock:
            self.manual_context = context
            self.db.set_setting("manual_context", context)

    def set_calibration(self, enabled):
        with self._lock:
            if enabled and self.manual_context is None:
                raise ValueError(
                    "Choose a manual context before confirming representative normal calibration"
                )
            if enabled != self.calibrating:
                self._segment = uuid.uuid4().hex
            self.calibrating = enabled

    def configure(self, settings: Settings, mappings=None):
        settings.validate()
        self.stop()
        with self._lock:
            if mappings is not None:
                self.collector.mappings = validate_mappings(mappings)
                self.db.set_setting("mappings", self.collector.mappings)
            self.settings = settings
            self.db.set_setting("settings", settings.to_dict())
            self.profiles.settings = settings
            self.adaptation.settings = settings
            self.collector.foreground_enabled = settings.foreground_enabled
            self.features = FeatureExtractor(settings.feature_window)
            self._sequence = deque(maxlen=settings.sequence_length)
            self._trackers.clear()
            self._context = None

    def train(self, context):
        self.stop()
        with self._lock, self._lease:
            profile = self.profiles.train(context)
            self._trackers.pop(str(context), None)
            self.calibrating = False
            return profile

    def _close_event(self, explanation):
        if self._event_id and self._event_record:
            record = dict(self._event_record)
            record["explanation"] += " " + explanation
            self.repo.update_event(self._event_id, record, close=True)
        self._event_id, self._event_record = None, None

    def process(self, sample: TelemetrySample, *, origin="real_laptop") -> dict:
        with self._lock:
            context = self.recognizer.recognize(sample, self.manual_context)
            name = str(context.context)
            gap = (
                self._last_timestamp is not None
                and sample.timestamp - self._last_timestamp > self.settings.max_context_gap_seconds
            )
            if self._last_timestamp is not None and sample.timestamp <= self._last_timestamp:
                raise ValueError("Samples must be in increasing timestamp order")
            switched = name != self._context
            if switched or gap:
                self._close_event(
                    "Context changed or observation gap; episode closed without cross-context persistence"
                )
                if self._context:
                    self.adaptation.reset_collection(self._context)
                self.features.reset()
                self._sequence.clear()
                self._trackers.pop(name, None)
                self._segment = uuid.uuid4().hex
                self._grace = self.settings.context_grace
                if switched:
                    self.db.execute(
                        "INSERT INTO context_history(timestamp,context,confidence,source) VALUES (?,?,?,?)",
                        (sample.timestamp, name, context.confidence, context.source),
                    )
                self._context = name
            self._last_timestamp = sample.timestamp
            feature = self.features.extract(sample)
            profile = self.profiles.active(name)
            ready = feature.valid and feature.window_ready and profile is not None
            record = {
                "timestamp": sample.timestamp,
                "context": name,
                "confidence": context.confidence,
                "source": context.source,
                "features": feature.vector,
                "valid": feature.valid,
                "missing_fraction": feature.missing_fraction,
                "classification": WARMUP,
                "anomaly_score": None,
                "drift_score": None,
                "persistence_seconds": 0.0,
                "explanation": self.profiles.errors.get(
                    name, "Collect representative normal data and train this context"
                ),
                "method": "uncalibrated; no ML prediction",
                "inference_ms": None,
                "profile_version": profile.version if profile else None,
            }
            if profile and profile.metadata["feature_window"] != self.settings.feature_window:
                ready = False
                record["explanation"] = "Feature-window setting changed; recalibration required"
            if not feature.valid or not feature.window_ready:
                record["explanation"] = "Waiting for a complete valid within-context feature window"
                self._sequence.clear()
            anomalous = False
            if ready:
                if self._sequence.maxlen != profile.metadata["sequence_length"]:
                    self._sequence = deque(maxlen=profile.metadata["sequence_length"])
                self._sequence.append(feature.vector)
                active = self._trackers.get(name)
                if active is None or active[0] != profile.version:
                    footprint = profile.metadata["feature_window"] + profile.metadata["sequence_length"] - 1
                    self._trackers[name] = (
                        profile.version,
                        PersistenceTracker(
                            self.settings, profile.metadata["reference_scores"], receptive_field=footprint
                        ),
                    )
                    self._grace = self.settings.context_grace
                tracker = self._trackers[name][1]
                started = time.perf_counter()
                anomaly = profile.score(feature.vector, list(self._sequence), self.settings.false_alarm_alpha)
                inference_ms = (time.perf_counter() - started) * 1000
                anomalous = anomaly.anomalous
                decision = tracker.update(
                    sample.timestamp, anomaly.score, anomalous, suppress=self._grace > 0
                )
                self._grace = max(0, self._grace - 1)
                record.update(
                    classification=decision.classification,
                    anomaly_score=anomaly.score,
                    drift_score=decision.drift_score,
                    persistence_seconds=decision.persistence_seconds,
                    explanation=decision.explanation,
                    method=anomaly.method,
                    inference_ms=inference_ms,
                )
                if decision.classification in {TEMPORARY, DRIFT}:
                    if self._event_id is None:
                        self._event_id = self.repo.open_event(record)
                    else:
                        self.repo.update_event(self._event_id, record)
                    self._event_record = dict(record)
                else:
                    self._close_event("Returned within baseline")
                if decision.classification == DRIFT:
                    self.adaptation.begin(name, sample.timestamp)
                elif decision.classification == NORMAL:
                    self.adaptation.reset_collection(name, "Recovered to baseline before candidate approval")
                if name in self.adaptation.pending:
                    self.adaptation.observe(name, sample.timestamp, feature.vector)
                    item = self.adaptation.pending[name]
                    if self.settings.auto_adapt and len(item["rows"]) >= self.settings.adaptation_samples:
                        try:
                            self.adaptation.stage(name)
                            self.adaptation.accept(item["id"], automatic=True)
                            self._trackers.pop(name, None)
                            self._sequence.clear()
                            self._close_event("Opt-in stable candidate accepted")
                        except ValueError:
                            pass  # Continue quarantine until stability criteria are met.
            if self.calibrating and feature.valid and feature.window_ready and not anomalous:
                self.repo.add_calibration(
                    name,
                    feature.timestamp,
                    feature.vector,
                    self._segment,
                    "synthetic_normal" if origin == "synthetic" else "user_normal",
                )
            elif self.calibrating and anomalous:
                record["explanation"] += (
                    " Excluded from normal calibration because active model marked it anomalous."
                )
                self._segment = uuid.uuid4().hex
            if not feature.valid:
                self._segment = uuid.uuid4().hex
                self._trackers.pop(name, None)
                self._close_event("Missing telemetry interrupted persistence")
            self.repo.observe(record)
            self.latest = record
            return record

    def delete_data(self, include_models=False):
        self.stop()
        with self._lock, self._lease:
            self.privacy.delete(include_models)
            self.latest = None
            self._trackers.clear()
            self._sequence.clear()
            self.features.reset()
            self._context = None
            self._last_timestamp = None
            self._event_id = self._event_record = None
            self.calibrating = False
            self.adaptation.pending.clear()
            if include_models:
                self.profiles.cache.clear()
                self.profiles.errors.clear()
                self.manual_context = None
                self.settings = Settings().validate()
                self.collector.mappings = dict(DEFAULT_MAPPINGS)
                self.collector.foreground_enabled = False
                self.profiles.settings = self.settings
                self.adaptation.settings = self.settings
                self.features = FeatureExtractor(self.settings.feature_window)
                self._sequence = deque(maxlen=self.settings.sequence_length)
