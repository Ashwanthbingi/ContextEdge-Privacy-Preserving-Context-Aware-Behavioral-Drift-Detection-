# Privacy boundaries and operational limitations

## Collected locally

CPU, RAM, available memory, disk and network byte counters, process counts and aggregate process resource statistics. Executable basenames may be categorized locally using an editable map. An optional Windows adapter retrieves a foreground process identity solely for that categorization. No full path or window content is read.

## Never collected by ContextEdge

Keystrokes, screenshots, screen/audio/video recordings, browser history, sites/URLs, packet contents, network endpoints or IP addresses, file/document contents or names, terminal histories, credentials, full command lines, process environments or full process paths.

Names are discarded after categorization. Transient PID/creation-time bookkeeping for resource deltas is in memory only. Persisted mappings contain only the user-configured executable basenames and categories. Persisted model distributions can reveal aggregate workload patterns, so these artifacts should still be treated as private.

## Local operation

The application has no cloud inference API, remote telemetry client or outbound data-upload path. Streamlit binds to 127.0.0.1 and its analytics flag is false. Models and resources do not download at runtime. Package installation is a separate setup activity requiring network access. The privacy test suite runs telemetry/processing with outbound `socket.connect` blocked and scans collector API usage; this supports the implementation claim but is not an OS-wide forensic network audit.

Local browser-to-Streamlit communication is necessary for the dashboard. Do not change the bind address or expose the port unless you intentionally design authentication and transport protection. Browser extensions, unrelated laptop software, package managers and operating-system telemetry are outside this application's control. No paid service is needed.

## Deletion and persistence

Raw observations and calibration samples expire under the configured retention. Profiles and version history remain intentionally until explicit deletion. Privacy controls show the actual storage directory and allow observation-only or full removal after typing DELETE. Deletion always stops monitoring and validates the model directory. It does not remove OS backups, cloud-synced filesystem copies or SSD-level remnants.

Calibration is opt-in and resets on restart. Auto-adaptation is off by default. A stable change is not necessarily a legitimate change; accepting one requires informed review or an explicitly enabled automatic policy. The app is a behavioural research tool, not an endpoint-security control.
