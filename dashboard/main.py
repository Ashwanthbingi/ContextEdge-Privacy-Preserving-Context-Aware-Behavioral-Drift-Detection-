from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime, timezone

import pandas as pd
import streamlit as st

from contextedge.config import Context
from contextedge.context.mappings import DEFAULT_MAPPINGS
from contextedge.detection.state_machine import DRIFT, NORMAL, TEMPORARY
from contextedge.engine import ContextEdgeEngine
from contextedge.features.extraction import FEATURE_NAMES
from contextedge.privacy.privacy_manager import PRIVACY_NOTICE


@st.cache_resource
def get_engine():
    return ContextEdgeEngine()


def format_time(value):
    return datetime.fromtimestamp(value, timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC") if value else "—"


def main():
    st.set_page_config(
        page_title="ContextEdge · Local behavioural intelligence", page_icon="◈", layout="wide"
    )
    st.markdown(
        """<style>
    .block-container {padding-top:2rem;max-width:1500px}
    [data-testid="stMetric"] {background:#152136;padding:16px;border-radius:12px;border:1px solid #26384D}
    [data-testid="stMetricLabel"] {color:#9EB3C9}
    .eyebrow {color:#43D9B8;letter-spacing:.17em;font-size:12px;font-weight:600}
    </style><div class="eyebrow">ON-DEVICE INTELLIGENCE / PRIVATE BY DESIGN</div>""",
        unsafe_allow_html=True,
    )
    st.title("ContextEdge")
    st.caption("Learn the rhythm of your laptop. Understand when it changes.")
    try:
        engine = get_engine()
    except Exception as exc:
        st.error(
            f"Local storage initialization failed ({type(exc).__name__}). Set CONTEXTEDGE_DATA_DIR to a writable local folder."
        )
        st.stop()
    with st.sidebar:
        st.header("Monitoring controls")
        start, stop = st.columns(2)
        if start.button("Start", type="primary", use_container_width=True):
            try:
                engine.start()
            except RuntimeError as exc:
                st.error(str(exc))
        if stop.button("Stop", use_container_width=True):
            engine.stop()
        options = ["Automatic"] + [str(c) for c in Context]
        chosen = st.selectbox(
            "Activity context",
            options,
            index=options.index(engine.manual_context) if engine.manual_context else 0,
        )
        selected = None if chosen == "Automatic" else chosen
        if selected != engine.manual_context:
            engine.set_calibration(False)
            engine.set_context(selected)
        calibrating = st.checkbox(
            "This is representative normal activity",
            value=engine.calibrating,
            disabled=selected is None,
            help="Explicitly adds valid, non-anomalous observations to this context's calibration set.",
        )
        if calibrating != engine.calibrating:
            engine.set_calibration(calibrating)
        st.caption(
            "Choose a manual context to calibrate. Models are trained only after enough contiguous observations."
        )
        with st.expander("Detection & retention settings"):
            cfg = engine.settings
            with st.form("settings"):
                interval = st.number_input(
                    "Sampling interval (seconds)", 0.25, 60.0, cfg.sampling_interval, 0.25
                )
                alpha = st.slider("Target anomaly tail probability", 0.01, 0.20, cfg.false_alarm_alpha, 0.01)
                persistence = st.number_input("Minimum sustained observations", 2, 100, cfg.min_persistence)
                window = st.number_input("Persistence window", 2, 500, cfg.persistence_window)
                confirmation = st.selectbox(
                    "Drift confirmation",
                    ["either", "both", "adwin", "page_hinkley"],
                    index=["either", "both", "adwin", "page_hinkley"].index(cfg.drift_confirmation),
                )
                adwin_delta = st.number_input("ADWIN delta", 0.0001, 0.1, cfg.adwin_delta, format="%.4f")
                ph_threshold = st.number_input(
                    "Page-Hinkley threshold", 0.5, 100.0, cfg.page_hinkley_threshold, 0.5
                )
                days = st.number_input("Observation retention (days)", 1, 3650, cfg.retention_days)
                foreground = st.checkbox("Use minimal foreground category (Windows)", cfg.foreground_enabled)
                automatic = st.checkbox(
                    "Automatically accept stable drift candidates",
                    cfg.auto_adapt,
                    help="Opt-in policy. A stable behavioural change may still be undesirable; review is the default.",
                )
                if st.form_submit_button("Apply settings & stop monitoring"):
                    try:
                        engine.configure(
                            replace(
                                cfg,
                                sampling_interval=interval,
                                false_alarm_alpha=alpha,
                                min_persistence=int(persistence),
                                persistence_window=int(window),
                                drift_confirmation=confirmation,
                                adwin_delta=adwin_delta,
                                page_hinkley_threshold=ph_threshold,
                                retention_days=int(days),
                                foreground_enabled=foreground,
                                auto_adapt=automatic,
                            )
                        )
                        st.success("Saved. Start monitoring when ready.")
                    except ValueError as exc:
                        st.error(str(exc))
            st.caption(
                "Alpha changes the cutoff on held-out calibrated scores. Smaller alpha needs a larger validation set."
            )
        with st.expander("Application category mappings"):
            mappings = st.text_area(
                "Executable basename → category (JSON)",
                json.dumps(engine.db.get_setting("mappings", DEFAULT_MAPPINGS), indent=2),
                height=200,
            )
            st.caption(
                "Categories: coding, browsing, media, compute, other. Add local game executables under compute."
            )
            if st.button("Save mappings & stop monitoring"):
                try:
                    engine.configure(engine.settings, json.loads(mappings))
                    st.success("Mappings saved locally")
                except (ValueError, TypeError) as exc:
                    st.error(str(exc))
        st.divider()
        st.caption(
            "Local processing • localhost access\n\nPersistent drift describes behavioural change, not a malware verdict."
        )

    @st.fragment(run_every="2s")
    def live_panels():
        rows = engine.repo.recent(300)
        latest = rows[-1] if rows else None
        status, context_col, readiness = st.columns([1, 2, 2])
        status.metric("Monitoring", "RUNNING" if engine.running else "STOPPED")
        context_col.metric("Activity context", latest["context"] if latest else "No samples")
        readiness.metric(
            "Context evidence",
            f"{latest['confidence']:.0%} · {latest['source']}" if latest else "Awaiting observation",
        )
        if engine.error:
            st.error(engine.error)
        monitor, profiles, events, adaptation, privacy, evaluation = st.tabs(
            [
                "Live overview",
                "Behavioural profiles",
                "Anomalies & drift",
                "Adaptation",
                "Privacy",
                "Evaluation",
            ]
        )
        with monitor:
            if latest:
                label = latest["classification"]
                if label == NORMAL:
                    st.success(label + " · " + latest["explanation"])
                elif label in {TEMPORARY, DRIFT}:
                    st.warning(label + " · " + latest["explanation"])
                else:
                    st.info(label + " · " + latest["explanation"])
                features = dict(zip(FEATURE_NAMES, latest["features"], strict=True))
                cols = st.columns(4)
                for col, label, value in zip(
                    cols,
                    ["CPU", "Memory", "Disk I/O", "Network I/O"],
                    [
                        f"{features['cpu_percent']:.1f}%",
                        f"{features['ram_percent']:.1f}%",
                        f"{features['disk_read_mb_s'] + features['disk_write_mb_s']:.2f} MB/s",
                        f"{features['net_recv_mb_s'] + features['net_sent_mb_s']:.2f} MB/s",
                    ],
                    strict=True,
                ):
                    col.metric(label, value)
                cols = st.columns(4)
                cols[0].metric(
                    "Anomaly score",
                    f"{latest['anomaly_score']:.3f}" if latest["anomaly_score"] is not None else "—",
                )
                cols[1].metric(
                    "Drift evidence index",
                    f"{latest['drift_score']:.3f}" if latest["drift_score"] is not None else "—",
                )
                cols[2].metric("Persistence", f"{latest['persistence_seconds']:.1f} s")
                cols[3].metric(
                    "Local inference",
                    f"{latest['inference_ms']:.2f} ms" if latest["inference_ms"] is not None else "—",
                )
                st.caption(
                    f"Observed {format_time(latest['timestamp'])} · {latest['method']} · "
                    f"Threshold {1 - engine.settings.false_alarm_alpha:.3f}. Drift index is a rule-based progress indicator, not a probability."
                )
                frame = pd.DataFrame([dict(zip(FEATURE_NAMES, r["features"], strict=True)) for r in rows])
                frame.index = pd.to_datetime([r["timestamp"] for r in rows], unit="s", utc=True)
                left, right = st.columns(2)
                with left:
                    st.subheader("Resource rhythm")
                    st.line_chart(frame[["cpu_percent", "ram_percent"]], height=240)
                with right:
                    st.subheader("I/O activity · MB/s")
                    st.line_chart(
                        frame[["disk_read_mb_s", "disk_write_mb_s", "net_recv_mb_s", "net_sent_mb_s"]],
                        height=240,
                    )
                scores = pd.DataFrame(
                    {
                        "Anomaly": [r["anomaly_score"] for r in rows],
                        "Drift": [r["drift_score"] for r in rows],
                    },
                    index=frame.index,
                )
                st.line_chart(scores, height=180)
                if not engine.running:
                    st.caption("Historical observations shown. Monitoring is stopped.")
            else:
                st.info(
                    "Start monitoring to collect actual laptop telemetry. Then select a context and calibrate its normal behaviour."
                )
        with profiles:
            summaries = []
            for context in Context:
                profile = engine.profiles.active(context)
                count = engine.repo.calibration_count(context)
                summaries.append(
                    {
                        "Context": str(context),
                        "Approved observations": count,
                        "Status": "Ready" if profile else "Calibrating" if count else "Not calibrated",
                        "Active version": profile.version if profile else None,
                        "Models": "Isolation Forest + GRU/ONNX" if profile else "No predictions",
                        "Issue": engine.profiles.errors.get(str(context), ""),
                    }
                )
            st.dataframe(pd.DataFrame(summaries), hide_index=True, use_container_width=True)
            selected_profile = st.selectbox("Inspect / train context", [str(c) for c in Context])
            count = engine.repo.calibration_count(selected_profile)
            st.progress(
                min(count / engine.settings.min_calibration_samples, 1.0),
                text=f"{count} / {engine.settings.min_calibration_samples} minimum observations; contiguous validation also required",
            )
            if st.button(
                "Stop monitoring & train context models",
                disabled=count < engine.settings.min_calibration_samples,
            ):
                with st.spinner("Training locally and verifying ONNX consistency…"):
                    try:
                        profile = engine.train(selected_profile)
                        st.success(f"Version {profile.version} activated. Start monitoring to use it.")
                    except Exception as exc:
                        st.error(f"Training failed: {type(exc).__name__}: {exc}")
            profile = engine.profiles.active(selected_profile)
            if profile:
                st.caption("Baseline statistics learned from approved training observations")
                st.dataframe(
                    pd.DataFrame(
                        {
                            "Mean": profile.metadata["baseline_mean"],
                            "Standard deviation": profile.metadata["baseline_std"],
                        }
                    ),
                    use_container_width=True,
                )
                with st.expander("Training evidence & ONNX verification"):
                    st.json(profile.metadata)
            versions = engine.repo.versions(selected_profile)
            if versions:
                st.dataframe(
                    pd.DataFrame(versions).drop(columns=["metadata", "artifact_dir"]), hide_index=True
                )
                version = st.selectbox("Version to restore", [v["version"] for v in versions])
                if st.button("Stop monitoring & restore version"):
                    engine.stop()
                    try:
                        engine.profiles.rollback(selected_profile, version)
                        st.success(f"Version {version} restored")
                    except Exception as exc:
                        st.error(f"Rollback failed: {type(exc).__name__}")
        with events:
            st.caption(
                "A temporary anomaly remains provisional while active. Recovery closes it; sustained confirmed change upgrades it to drift."
            )
            history = engine.repo.events()
            if history:
                frame = pd.DataFrame(history)
                for field in ("started_at", "updated_at", "ended_at"):
                    frame[field] = frame[field].apply(lambda x: format_time(x) if pd.notna(x) else "Active")
                st.dataframe(frame, hide_index=True, use_container_width=True)
            else:
                st.info("No anomaly or drift events recorded.")
        with adaptation:
            st.caption(
                "Confirmed shifts are quarantined. Stable candidate profiles keep the previous version available for rollback."
            )
            for context, item in list(engine.adaptation.pending.items()):
                st.write(
                    f"{context}: {len(item['rows'])}/{engine.settings.adaptation_samples} quarantined observations"
                )
                if st.button(
                    f"Stage stable {context} candidate",
                    key=f"stage-{context}",
                    disabled=len(item["rows"]) < engine.settings.adaptation_samples,
                ):
                    try:
                        with engine._lock, st.spinner("Verifying candidate stability and training…"):
                            engine.adaptation.stage(context)
                        st.success("Candidate ready for review")
                    except ValueError as exc:
                        st.warning(str(exc))
            history = engine.db.rows("SELECT * FROM adaptations ORDER BY id DESC LIMIT 100")
            if history:
                st.dataframe(pd.DataFrame(history), hide_index=True, use_container_width=True)
                for item in history:
                    if item["status"] == "review":
                        st.write(
                            f"Review {item['context']} v{item['candidate_version']} · stability {item['stability']:.3f}"
                        )
                        accept, reject = st.columns(2)
                        if accept.button("Accept legitimate change", key=f"accept-{item['id']}"):
                            engine.stop()
                            engine.adaptation.accept(item["id"])
                            st.success("Candidate activated; previous profile preserved")
                        if reject.button("Reject candidate", key=f"reject-{item['id']}"):
                            engine.adaptation.reject(item["id"])
                            st.info("Previous profile retained")
            else:
                st.info("No adaptation cycles have been started.")
        with privacy:
            st.success(
                "Local collection • Local training • Local inference • No application telemetry uploads"
            )
            st.write(PRIVACY_NOTICE)
            st.code(str(engine.data_dir), language=None)
            st.caption(
                f"Observations, calibration samples and event retention: {engine.settings.retention_days} days. "
                "Learned model versions and adaptation audit history remain until deleted explicitly."
            )
            st.write("Retained numerical feature schema")
            st.code(", ".join(FEATURE_NAMES), language=None)
            if st.button("Apply retention now"):
                engine.db.retain(engine.settings.retention_days)
                st.success("Expired observations removed")
            acknowledgement = st.text_input("Type DELETE to enable local deletion", key="delete_ack")
            left, right = st.columns(2)
            if left.button("Stop & delete observation history", disabled=acknowledgement != "DELETE"):
                engine.delete_data(False)
                st.success("Observation history deleted; learned models retained")
            if right.button("Stop & delete all data and profiles", disabled=acknowledgement != "DELETE"):
                engine.delete_data(True)
                st.success("Local observations, profiles, settings and history deleted")
            st.caption(
                "SQLite pages are cleared and compacted. OS backups, filesystem snapshots and SSD remapping can retain prior bytes."
            )
        with evaluation:
            st.caption(
                "Only executed experiments appear here. Synthetic accuracy is not evidence of real-world detection accuracy."
            )
            results = engine.db.rows("SELECT * FROM evaluations ORDER BY id DESC LIMIT 30")
            if results:
                for result in results:
                    with st.expander(
                        f"{result['source']} · {result['name']} · {format_time(result['timestamp'])}"
                    ):
                        st.json(json.loads(result["results"]))
            else:
                st.info(
                    "No evaluations recorded in this local store. Run the benchmark or import an executed evaluation report."
                )
            st.code(
                "python scripts/evaluate.py --output reports/synthetic\n"
                "python scripts/benchmark.py --seconds 30\n"
                "contextedge import-evaluation reports/synthetic/results.json",
                language="powershell",
            )

    live_panels()
