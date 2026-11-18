# Changelog

## 0.1.3 — 2026-11-18

Bug-fix and hardening release.

### Fixed
- `diagnose_quality` no longer flags `HUM` on pure 440 Hz tones.
- `analyze_audio` options (`identify_speakers`, `include_evidence`, `timestamps`) are honoured, and `options=None` is accepted.
  An explicit `speaker_diarization` task no longer conflicts with `identify_speakers=False`.
- Multilingual text: tokenisation and library full-text search keep non-Latin scripts and combining vowel marks
  (NFC-normalised; single CJK characters count as words; zero-width spaces split words). An existing FTS index
  built with the old tokenizer is rebuilt atomically.
- A timed-out stage thread can no longer mutate the context or cache of later stages.
- Diarization no longer crashes on all-zero speech windows.
- Streaming: word offsets in the transcript window, empty chunks, partial file init, and redacted reconcile.
- `export_audio_analysis` (srt/vtt/rttm) and `analyze_audio` with explicit tasks report an error / `partial`
  when a required backend is unavailable instead of succeeding.
- The quality cache key tracks quality rules, and the uncertainty key tracks quality when it ran.
- `put_variant` can no longer report success with stale variant metadata.
- Server worker threads no longer leak the job context between calls.
- A non-JSON LLM reply with stray braces, or an echoed evidence object, is parsed strictly; invalid
  `redact_extra_patterns` regexes are rejected at config load; `_pid_alive` handles 0, negative and overflowing pids.

### Security
- URL fetch: invalid ports are `INVALID_INPUT`; Teredo and NAT64 local-use prefixes embedding private/loopback
  IPv4 addresses are blocked.
- LLM loopback detection recognises `::ffff:127.0.0.1` and keeps a trailing dot when resolving the hostname.

## 0.1.2 — 2026-11-11

### Added
- `transcribe_audio` `hotwords`: names and jargon the decoder should favour (part of the transcript cache key).
  Needs faster-whisper >= 1.1.
- `LOW_RESOURCE_LANGUAGE` warning when Arabic, Hindi or Telugu is detected with a tiny/base/small Whisper model.
- MCP progress notifications (stage count) for synchronous tool calls when the client sends a progress token.
- Streamable-HTTP transport: `transport = "streamable-http"`, `http_host`, `http_port`, bearer token from
  `http_auth_token_env`. Refuses to start without a token or without an explicit `allowed_roots`.

### Security
- URL fetch connects to the vetted IP address instead of re-resolving the hostname (closes the DNS-rebinding window).

## 0.1.1 — 2026-10-19

Bug-fix release. Upgrading is recommended, especially on Windows and for anyone who sets `temp_dir`.

### Fixed
- **Data loss:** a blank `temp_dir` (e.g. `temp_dir = ""`, or `AUDIO_MCP_TEMP_DIR=""`) resolved to the working
  directory, which startup cleanup then emptied. Blank directory settings now mean "use the default", and
  startup cleanup only removes temp directories left by dead server instances (`proc_<pid>_<tag>`), never other files.
- **Privacy:** with `redact_outputs` / `redact: true`, timeline speech nodes kept the transcript in `label`, and
  numbers spoken across several words ("555 123 4567") were not redacted in word lists. Both are now redacted.
- **Windows:** concurrent requests for the same audio could fail with `PermissionError` while another thread
  replaced `meta.json` or a cached result.
- An LLM reply that was valid JSON with the wrong structure made `summarize_audio` / `ask_audio` /
  `compare_audio` fail instead of falling back to the rule-based result.
- `get_audio_timeline(diarize=True)` could return a cached timeline without speaker labels.
- The hourly sweep deleted cached results of audio that was still in use.
- `ask_audio`: passive questions ("When is the deadline mentioned?") were not recognised as time-of-mention
  questions; "anything odd?" never triggered the unusual-sounds intent (stray control characters in a regex).
- Async `analyze_audio` with `tasks="auto"` reported 0.99 progress at the start of the job.
- Stream pushes and `compare_transcription` ran models outside the `max_concurrent_jobs` limit.

### Performance
- Sound-event tagging and audio embeddings share one PANNs pass instead of two.
- `compare_audio` with a baseline no longer runs feature, event and embedding stages it does not use.

## 0.1.0 — 2026-10-19

First release: 13 MCP tools, local models, evaluation on public datasets (see `docs/eval/RESULTS.md`).
