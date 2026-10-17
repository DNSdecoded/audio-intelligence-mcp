# Audio Intelligence MCP Server — Design

Date: 2026-10-17
Status: draft for approval

## 1. Intent

A local-first MCP server that lets AI agents analyze audio and get back
structured, evidence-grounded JSON: transcripts, speakers, sound events,
acoustic measurements, quality diagnosis, timelines, answers to questions,
comparisons, and library search. Agents must be able to trust the output:
every claim points to timestamps, every confidence value comes from a real
model, and every skipped or failed stage is reported, never hidden.

The original brief (user's long-form request, 11 sections) is the
requirements source. This document records the decisions taken on top of it.

## 2. Decisions from the user

- **Local only.** Audio never leaves the machine. All models run locally.
- **Cloud LLM optional, via config.** Off by default. When on, it receives
  only text (transcript + event list), never audio. Every response states
  whether any data was sent off-machine.
- **Any audio format.** WAV, FLAC, MP3, OGG/Opus, M4A/AAC, WebM, AIFF, WMA,
  and anything else the installed ffmpeg can decode.
- **Priority: all 12 tools working evenly.** No tool may be a stub. Depth is
  traded for breadth: each tool must run end to end on real audio, with
  honest reporting of its limits.
- **I/O contract** as supplied by the user (section 5 below reproduces it).

## 3. Goals, non-goals, anti-goals

Goals
- All 12 MCP tools callable from Claude Code / Claude Desktop over stdio,
  each with a strict input schema and the common response envelope.
- Installable by others on Windows, macOS, and Linux (user decision).
  The development and reference machine is Windows 11, CPU only, 16 GB
  RAM. CPU is the baseline everywhere; CUDA is used automatically when
  available, and nothing requires it.
- Graceful degradation: a missing model or failed stage yields
  `status: "partial"` with the reason, never a crash.
- Tests: unit, MCP schema, integration, end-to-end on generated fixtures.
- README precise enough to install and connect from scratch.

Non-goals
- GPU optimization, distributed workers, multi-tenant auth.
- Real-time socket/WebRTC streaming (chunked sessions only; see 6.11).
- Training or fine-tuning models.
- A UI.

Anti-goals (failure even if it "works")
- Invented confidence values, timestamps, or quotes.
- Stating audio is authentic/fake, or an emotion as fact.
- Silently dropping a failed stage.
- Sending audio off-machine, or sending text off-machine without the
  config explicitly enabling it.
- Advertising a feature in README/tool descriptions that is not implemented.

## 4. Constraints

- Python. System Python is 3.14; ML wheels (ctranslate2, speechbrain,
  torch audio stack) may lag 3.14, so the project pins **Python 3.12 via
  `uv`** (`uv` 0.12 is installed). Builder may move to 3.13 if every
  dependency has wheels.
- ffmpeg/ffprobe found via config `ffmpeg_path`, then `PATH`. At startup
  the server checks they exist and reports `MODEL_UNAVAILABLE`-style
  capability errors if they are missing. README gives install steps per
  OS: winget (Windows), brew (macOS), apt/dnf (Linux). The dev machine
  has ffmpeg 8.1 from winget.
- Cross-platform code: `pathlib` everywhere, resolve paths before the
  allowed-roots check, case-insensitive comparison on Windows and macOS,
  handle Windows drive letters and UNC paths, no shell=True in
  subprocess calls, platform-specific default dirs via `platformdirs`.
- CPU inference only. Defaults chosen for CPU latency (see 6.2).
- First run downloads ~1 GB of model weights; after that, fully offline.
- Project root: `F:\audio_mcp` (empty, not yet a git repo).

## 5. Contract

### 5.1 Response envelope (every tool)

```json
{
  "status": "success | partial | processing | error",
  "audio_id": "aud_<16 hex>",
  "job_id": "job_<...>",
  "result": { "...": "tool-specific" },
  "warnings": [ { "code": "...", "message": "..." } ],
  "errors":   [ { "code": "...", "message": "...", "stage": "...", "retryable": false } ],
  "processing": {
    "duration_seconds": 2.4,
    "stages_run": ["decode", "transcription"],
    "stages_skipped": [ { "stage": "diarization", "reason": "..." } ],
    "models_used": [ { "stage": "transcription", "name": "faster-whisper", "version": "small", "local": true } ],
    "data_sent_off_machine": false,
    "schema_version": "1.0"
  }
}
```

`audio_id` omitted where not applicable; `job_id` only when
`status=processing`. `analyze_audio` names its payload `analysis` instead
of `result` to match the user's example; all other tools use `result`.

Error-only shape (validation failures before any work):

```json
{ "status": "error",
  "error": { "code": "UNSUPPORTED_AUDIO_FORMAT", "message": "...", "retryable": false } }
```

Error codes (minimum set): `INVALID_INPUT`, `PATH_NOT_ALLOWED`,
`FILE_NOT_FOUND`, `UNSUPPORTED_AUDIO_FORMAT`, `CORRUPT_AUDIO`,
`EMPTY_AUDIO`, `FILE_TOO_LARGE`, `DURATION_TOO_LONG`, `URL_FETCH_DISABLED`,
`URL_FETCH_FAILED`, `AUDIO_NOT_FOUND` (unknown audio_id), `JOB_NOT_FOUND`,
`MODEL_UNAVAILABLE`, `STAGE_FAILED`, `TIMEOUT`, `CANCELLED`, `LLM_DISABLED`.

### 5.2 Units and conventions

- Time: seconds, float, 3 decimals, relative to start of original file.
- Frequency: Hz. Level: dBFS. Amplitude: linear, -1..1 scale.
- `confidence` appears only when the model emits a score. A field
  `confidence_kind` says what it is: `"model_probability"` (e.g. whisper
  avg logprob → exp, PANNs sigmoid) or `"heuristic"` (rule-derived). Never
  labelled calibrated unless calibration was actually done.
- Speaker labels: `speaker_1..N`, consistent within one audio_id.

### 5.3 Audio source (input to every audio-taking tool)

```json
{ "type": "file_path", "path": "..." }
{ "type": "base64", "data": "...", "filename_hint": "x.mp3" }
{ "type": "url", "url": "https://..." }
{ "type": "audio_id", "audio_id": "aud_..." }
```

- `file_path`: resolved (symlinks too) and must sit under one of
  `allowed_roots` from config; else `PATH_NOT_ALLOWED`.
- `base64`: size-capped before decode (config `max_upload_mb`).
- `url`: disabled by default (`allow_url_fetch: false`); when enabled,
  http(s) only, size cap enforced while streaming, private/loopback IPs
  blocked, timeout.
- `audio_id`: reuse previously ingested audio and its cached results.

## 6. Architecture

```
src/audio_mcp/
  server.py            MCP (FastMCP, stdio) tool registration only
  config.py            pydantic-settings: TOML file + AUDIO_MCP_* env vars
  envelope.py          response envelope, error codes, helpers
  ingest/              source resolution, validation, ffprobe, ffmpeg decode
  store/               audio store (content-hash ids), result cache, retention
  backends/            model adapters behind small Protocols
    asr.py             faster-whisper adapter
    diarization.py     ECAPA+clustering adapter; pyannote adapter (optional)
    events.py          PANNs CNN14 adapter
    embeddings.py      sentence-transformers (text) + PANNs embedding (audio)
    llm.py             none | anthropic | openai-compatible | ollama
  dsp/                 features, quality, integrity, speech-rate (numpy/scipy)
  pipeline/            orchestrator: task planning, stage DAG, partial results
  timeline/            event graph build + relations
  reasoning/           ask (retrieval + rules | LLM), summarize, uncertainty
  compare/             feature/event/transcript diffs, baseline anomaly z-scores
  library/             SQLite + FTS5 index, vector search
  stream/              chunk sessions, provisional results, reconciliation
  export/              json, markdown, srt, vtt, csv, rttm
  jobs.py              async job manager: timeout, cancel, progress
tests/                 unit, schema, integration, e2e; fixtures generator
docs/                  README, config reference, tool reference, eval guide
config.example.toml
```

### 6.1 Ingest and preprocessing

1. Resolve source → bytes on disk in a private temp dir (random names;
   arbitrary user filenames never used as paths).
2. `ffprobe` (JSON, timeout) → format, codec, duration, rate, channels.
   No audio stream → `UNSUPPORTED_AUDIO_FORMAT`; probe failure →
   `CORRUPT_AUDIO`; zero duration → `EMPTY_AUDIO`; limits checked here.
3. `ffmpeg` decode to two arrays: native-rate float32 (for DSP) and
   16 kHz mono float32 (for models). PANNs needs 32 kHz; adapter resamples.
4. `audio_id` = `aud_` + first 16 hex of SHA-256 of the file bytes, so the
   same file always maps to the same id and cache.
5. Original bytes are never modified. Preprocessing (normalize, denoise via
   spectral gating, trim silence, mono, resample) writes a derived variant
   with a recorded transform list.

### 6.2 Backends (defaults, all local, CPU)

| Capability | Default | Notes |
|---|---|---|
| ASR + language ID | faster-whisper `small`, int8 | word timestamps; per-chunk language ID for code-switching; `model_size` configurable |
| Diarization | SpeechBrain ECAPA embeddings on VAD speech windows + agglomerative clustering | ungated, no token. Optional `pyannote` backend if `HF_TOKEN` set. Overlap detection only with pyannote; ECAPA path reports overlap as unsupported |
| Sound events | PANNs CNN14 (AudioSet 527 classes) | framewise → thresholded → merged intervals |
| Audio embedding | PANNs 2048-d clip embedding | used by compare + library similarity |
| Text embedding | sentence-transformers `all-MiniLM-L6-v2` | transcript/event search |
| VAD | silero-vad (ONNX) | speech regions for diarization and speech rate |
| LLM | none | optional; text only |

Each adapter implements a small Protocol (`load()`, `available() -> (bool,
reason)`, `run(...)`) and is chosen by config name. A failing `load()`
marks the capability unavailable; the orchestrator reports it as skipped.

### 6.3 DSP (no models)

- Features: duration, rate, channels, codec, RMS (linear + dBFS), peak,
  crest factor, clipping ratio, DC offset, silence ratio + intervals,
  SNR estimate (speech vs non-speech energy via VAD, fallback percentile
  method, labelled as estimate), spectral centroid/bandwidth/rolloff/
  flatness, band energy distribution, pitch (YIN/pYIN f0 stats on voiced
  frames), tempo/onset rate (onset envelope autocorrelation), zero-crossing
  rate.
- Quality diagnosis: clipping, low level, silence, noise floor, hum
  (50/60 Hz peaks), echo/reverb indicator (autocorrelation / decay
  heuristic), bandwidth limitation, DC offset → issues list with severity
  and recommended preprocessing.
- Integrity indicators: abrupt spectral discontinuities, zero-run splices,
  noise-floor jumps, band-limit inconsistencies. Output is a list of
  observations with timestamps plus a fixed limitations statement. No
  authenticity verdict, ever.
- Speech characteristics: words/min from ASR words, pause count/durations
  from VAD, f0 range/variability. Emotion/tone not computed by default; if
  a future adapter adds it, it is labelled `model_inference`.

`librosa` may be used if it installs cleanly on the pinned Python;
otherwise scipy implementations. Builder's call.

### 6.4 Orchestrator

- Input: requested tasks, or `"auto"`, plus optional `goal` text, `budget`
  (`fast | balanced | thorough`), and audio properties.
- `auto` planning: always decode + features + quality (cheap). Run VAD; if
  speech ratio > threshold → ASR (+ diarization if duration > 20 s or goal
  mentions speakers/meeting); run event detection unless budget=`fast` and
  audio is speech-dominant. Goal keywords (meeting, machine, compare,
  environment, music) adjust the plan. Plan and reasons are returned.
- Stages run as a dependency DAG; each stage result cached by
  (audio_id, stage, stage-config hash). Failures are captured per stage.
- `budget=thorough` enables cross-checks (6.6).
- Recovery: when quality issues are found and `auto_preprocess` is on,
  re-run ASR on the preprocessed variant and keep both; report which had
  higher mean word probability and fewer low-confidence words, as a
  measured comparison, not a claim of "better".

### 6.5 Timeline / event graph

Nodes: speech segments (with speaker), speaker turns, sound events,
silence, overlaps (when known), transitions (speaker change, onset of
loud event, level jump). Each node: `id`, `type`, `label`, `start`,
`end`, `speaker_id?`, `confidence?`, `confidence_kind?`, `source`.
Edges: `precedes`, `overlaps`, `during`, `interrupts` (event inside
speech), with gap seconds. Output both `nodes`/`edges` JSON and a `text`
rendering (`00:05–00:07  door (0.81)`).

### 6.6 Uncertainty

- Low-confidence words/segments flagged (threshold in config).
- Segments where ASR text exists but VAD says no speech (or vice versa)
  flagged as conflict.
- PANNs "Speech" vs VAD disagreement flagged.
- `thorough` budget: second ASR pass with a different decode setting
  (or model size) and a word-level disagreement rate.
- Quality issues propagate as warnings onto affected stages.
- Every `ask_audio` answer has `support: direct | inferred | insufficient`.

### 6.7 ask_audio

Default (no LLM): question parsed by rules into intents — time-of-mention
(keyword/semantic match over transcript segments via embeddings + FTS),
before/after/during an event (timeline relations), presence ("any X?"
over event labels incl. AudioSet synonyms), unusual sounds (rare-label
events + quality anomalies), change over time (first vs last third
feature/event diff), who/how many speakers. Answer text built from
templates + evidence list. Unrecognized intent → semantic segment
retrieval, `support: "insufficient"` if best score below threshold.
With LLM enabled: same retrieval produces an evidence pack; LLM answers
only from it and must cite evidence ids; uncited claims are dropped and
the response labels `interpretation` vs `detected`.

### 6.8 summarize_audio

Default: extractive — top segments by embedding centrality + keyword
salience, topics from keyword clustering, key moments from timeline
transitions and high-salience events. Every sentence carries evidence
ids. With LLM: abstractive summary constrained to the evidence pack.
Non-speech audio: summary built from events and features.

### 6.9 compare_audio

Two or more sources, or one source + `baseline` (one or many baseline
recordings). Outputs: feature deltas with units, spectral band deltas,
event set diff (label added/removed/rate change), transcript diff (if
speech), audio-embedding cosine similarity. Baseline mode: per-feature
mean/std over baselines → z-scores → anomalies over `z_threshold`,
also over 1 s windows to localize anomalies in time. With a single
baseline, std comes from its windows. Output separates
`measured_differences` from `possible_interpretations` (fixed, hedged
rules, or LLM if enabled).

### 6.10 search_audio_library

- `index` action: add files/folders (within allowed roots) → runs a
  configurable index pipeline (default: ASR + events + embeddings), stores
  segments in SQLite (FTS5 for text, float32 BLOB vectors).
- `search` action: query text + filters (language, duration range,
  event label, speaker count). Hybrid score: BM25 + cosine (text
  embedding) + optional query-by-example audio (PANNs embedding).
  Returns recording, segment start/end, excerpt, scores.
- `list` / `remove` actions.
- The library is persistent and outside the 24 h TTL: it has its own
  `library_retention` setting (default `keep`), and `retention: none`
  does not clear it. It stores derived data only (segments, transcript
  text, events, vectors, features) plus the source file path and
  content hash. It never stores a copy of the audio. When a result's
  source file is missing or its hash changed, the result is returned
  with `source_available: false`.
- Vector search is brute-force numpy; documented ceiling. Adapter
  interface allows swapping to FAISS/sqlite-vec later.
- Indexing large folders runs as a job.

### 6.11 analyze_audio_stream (chunked)

Actions: `start` (session id, sample format), `push` (base64 chunk or
file path chunk), `status`, `finish`, `cancel`.

The `start` call fixes the chunk format for the session:
- `pcm`: raw `s16le` or `f32le`, with declared `sample_rate_hz` and
  `channels`. Each chunk is decoded on its own.
- `file`: each chunk is a self-contained audio file in any format
  ffmpeg reads, for example an agent pushing successive segment files.

Continuous encoded streams (WebM/Opus or MP3 frames split across chunks)
are not supported and are documented as future work. Each `push` runs DSP +
events on the new chunk and ASR on a rolling window, returning
provisional results marked `provisional: true`. `finish` concatenates,
runs the full pipeline once, and returns a reconciliation diff
(provisional vs final). Live socket/WebRTC streaming documented as
future work.

### 6.12 export_audio_analysis

Formats: `json`, `markdown`, `srt`, `vtt`, `csv` (events/segments),
`rttm` (speaker turns). Writes to configured `export_dir`, returns path +
size + sha256. Optional redaction (emails, phone numbers, card-like
numbers, configurable regex list) applied on export and optionally on
all outputs.

### 6.13 Jobs

Every audio-taking tool accepts `async` (default: auto — async when
duration > `async_threshold_seconds`). Async returns `status: processing`
+ `job_id`. Progress = stages done / planned. Per-stage timeout; global
concurrency semaphore (`max_concurrent_jobs`, default 1 on CPU). Polling
and cancel: builder's call between a `job_id` input on each tool or a
small 13th `get_job` tool.

### 6.14 Privacy and retention

- Config `retention`: `none` (delete on exit), `ttl` (hours), `keep`.
  Default: ttl 24 h for stored audio and cached results.
- Temp dir cleaned on startup and shutdown.
- Logs: JSON lines to stderr/file, never transcript text or audio bytes
  unless `log_sensitive: true`.
- `processing.data_sent_off_machine` is true only when an LLM call went
  to a non-loopback host. Calls to a local Ollama or OpenAI-compatible
  server on localhost report `false`, and `models_used` records them with
  `local: true`. Every LLM backend, local or remote, is off unless the
  config enables it.
- README states which steps can send data out (only the LLM) and that
  model downloads contact Hugging Face / GitHub on first run only.

## 7. The 12 tools (input highlights → result)

| Tool | Input | Result |
|---|---|---|
| analyze_audio | source, tasks[] or "auto", goal?, budget?, options | combined analysis + plan |
| transcribe_audio | source, language?, word_timestamps, diarize, translate_to_en? | text, segments, words, languages[], speakers? |
| detect_audio_events | source, labels?, min_confidence | events[] intervals with labels |
| inspect_audio_quality | source, apply_preprocessing? | metrics, issues[], recommendations, before/after if applied |
| extract_audio_features | source, feature_groups? | metadata + features with units |
| ask_audio | audio_id or source, question | answer, support, evidence[], confidence? |
| get_audio_timeline | audio_id or source, include? | nodes, edges, text |
| compare_audio | sources[] or source+baseline[], aspects? | measured_differences, anomalies, similarity, interpretations |
| search_audio_library | action index/search/list/remove, query, filters | matches with timestamps + scores |
| summarize_audio | audio_id or source, length, focus? | summary, topics, key_moments with evidence |
| analyze_audio_stream | action, session_id, chunk | provisional or final results |
| export_audio_analysis | audio_id, format, redact? | path, size, sha256 |

`translate_to_en` uses whisper's translate task, aligned to segment
timestamps; translation into other languages is not supported locally.
Integrity indicators (6.3) are an `analyze_audio` task (`integrity`).

Tools that take `audio_id or source` run the needed stages on demand if
the cache lacks them.

## 8. Testing and evaluation

- Fixtures generated by a script (no licensing issues): sine tones,
  silence, white/pink noise, clipped signal, synthetic alarm (beeps),
  impulses (door/impact-like), spliced file, speech clips and a
  two-voice dialogue (generated once with local TTS and committed with
  reference transcripts, so every OS tests the same bytes; the generator
  script can regenerate them where `pyttsx3` TTS is available), corrupted bytes, non-audio file, zero-length file,
  a long (30 min) generated file for async/limits; each converted to
  wav/flac/mp3/ogg/m4a via ffmpeg for format coverage.
- Unit: DSP functions against analytic expectations (RMS of sine, clip
  ratio, silence intervals), envelope, path guard, config.
- Schema: every tool's input schema valid, invalid inputs → correct
  error codes, outputs match envelope schema.
- Integration: each backend adapter on fixtures (marked `slow`, skipped
  with reason if model unavailable).
- E2E: spawn server over stdio with the MCP client SDK, call all 12 tools.
- CI: GitHub Actions matrix on `windows-latest`, `macos-latest`, and
  `ubuntu-latest` with Python 3.12 and ffmpeg installed per OS. Fast
  tests (unit, schema, DSP, ingest, e2e with models stubbed) run on every
  push. Slow model tests run on one OS with a cached model directory.
- Eval scripts: WER (jiwer) vs reference transcripts, event P/R vs
  labeled intervals, DER vs RTTM, latency/real-time factor and peak RSS.
  Docs explain running on public datasets (LibriSpeech, ESC-50, AMI)
  the user downloads.

## 9. Deliverables

Source, `pyproject.toml` + `uv.lock`, `config.example.toml`,
`.env.example`, README (install, connect to Claude Code/Desktop, config,
troubleshooting), docs/tools.md (schemas + example success/error JSON),
docs/evaluation.md, docs/capabilities.md (implemented / partial / future),
tests, fixture generator, benchmark script.

Build order follows the brief's 4 phases; project runnable and tests
green at end of each:

1. **Foundation:** MCP server, ingest/validation, decode, transcription,
   sound-event detection, metadata, acoustic features, envelope, essential
   tests. Tools: `transcribe_audio`, `detect_audio_events`,
   `extract_audio_features`, first cut of `analyze_audio`.
2. **Audio intelligence:** `ask_audio`, `get_audio_timeline`,
   orchestrator (auto plan, budgets), uncertainty reporting,
   `inspect_audio_quality` with preprocessing, diarization,
   `summarize_audio`.
3. **Advanced analytics:** `compare_audio` + baseline anomalies,
   `search_audio_library`, multilingual/code-switching + translation,
   integrity indicators.
4. **Production readiness:** `analyze_audio_stream`, jobs (async,
   cancel, progress), retention/privacy/redaction,
   `export_audio_analysis`, optional LLM backend, benchmarks and eval
   scripts, full docs.

Brief's acceptance criteria (all must hold at the end): server starts and
connects from an MCP client; every tool has a documented schema and can
be called; representative samples analyzed end to end; consistent
timestamps and structured outputs; Q&A returns timestamped evidence;
errors, uncertainty, unsupported features and partial failures are
reported; code is modular, tested, documented, configurable; install and
example workflows reproducible from docs; every advertised feature is
implemented or explicitly marked incomplete; privacy, licensing,
operational limits, model constraints and evaluation method documented.

## 10. Left to the builder

- Exact library versions and Python 3.12 vs 3.13.
- librosa vs pure scipy for DSP.
- Thresholds and default parameter values (documented in config).
- TTS vs CC0 clips for speech fixtures.
- Job polling via per-tool `job_id` or a 13th `get_job` tool.
- Internal module boundaries beyond section 6.

## 11. Calls made by the designer (not discussed with user)

- Python + official `mcp` SDK (FastMCP), stdio transport.
- faster-whisper `small` int8 as default ASR.
- ECAPA + clustering as default diarization (no HF token needed);
  pyannote optional.
- PANNs CNN14 for events and audio embeddings; silero-vad for VAD.
- No LLM by default: rule-based / extractive ask + summarize.
- SQLite FTS5 + numpy vectors for search.
- Chunked-session streaming only; live sockets marked future.
- URL input off by default.
- Content-hash audio_id.
- Retention default: 24 h TTL.
- Library is persistent and outside the TTL. It stores derived data and
  file references only, never audio copies.
- Stream chunks are raw PCM or self-contained files. Continuous encoded
  streams are future work.
- `data_sent_off_machine` counts only non-loopback LLM hosts.
- Speech fixtures committed (TTS-generated once) for identical cross-OS
  tests. CI runs fast tests on all 3 OSes and slow model tests on one.
- CUDA used opportunistically; CPU is the supported baseline.
- Phases and acceptance criteria copied from the original brief into
  section 9.
