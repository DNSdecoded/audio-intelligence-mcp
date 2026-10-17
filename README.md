# Audio Intelligence MCP Server

A local-first [Model Context Protocol](https://modelcontextprotocol.io) server that lets AI agents analyze audio:
transcripts with word timings, speakers, sound events, acoustic measurements, quality diagnosis, timelines,
evidence-grounded answers, comparisons, library search, streaming, and exports.

Everything runs on your machine. Audio never leaves it. An LLM is optional, off by default, and only ever
receives text (transcript snippets and event labels); every response reports `data_sent_off_machine`.

Every tool returns the same JSON envelope:

```json
{
  "status": "success | partial | processing | error",
  "audio_id": "aud_0123456789abcdef",
  "result": { "...": "tool-specific (analyze_audio uses \"analysis\")" },
  "warnings": [{"code": "OVERLAP_UNSUPPORTED", "message": "..."}],
  "errors": [{"code": "STAGE_FAILED", "stage": "events", "message": "...", "retryable": false}],
  "processing": {"duration_seconds": 2.4, "stages_run": ["vad", "transcription"], "stages_skipped": [],
                 "cache_hits": [], "models_used": [{"stage": "transcription", "name": "faster-whisper",
                 "version": "small", "local": true}], "data_sent_off_machine": false, "schema_version": "1.0"}
}
```

Validation failures use `{"status": "error", "error": {"code", "message", "retryable"}}`. Times are seconds
(3 decimals), levels dBFS, frequencies Hz. `confidence` appears only when a model or rule produced it, always
with `confidence_kind` (`model_probability` or `heuristic`; none are calibrated).

## Requirements

- Python 3.12 and [uv](https://docs.astral.sh/uv/)
- ffmpeg and ffprobe on `PATH` (or set `ffmpeg_path` / `ffprobe_path`):
  - Windows: `winget install Gyan.FFmpeg`
  - macOS: `brew install ffmpeg`
  - Debian/Ubuntu: `sudo apt-get install ffmpeg`
- CPU is enough. CUDA is used automatically when available (`device = "auto"`).

## Install

```bash
git clone <this repo> audio_mcp && cd audio_mcp
uv sync --extra models            # local models (faster-whisper, PANNs, SpeechBrain, silero-vad, MiniLM)
uv sync --extra models --extra llm        # + optional Claude backend (anthropic SDK)
uv sync --extra models --extra pyannote   # + optional pyannote diarization (needs HF_TOKEN)
```

Without `--extra models` the server still starts: model-backed stages are reported in `stages_skipped` with the
reason, and the DSP tools (features, quality, integrity, comparison) work.

**First-run downloads** (cached in `models_dir`, then fully offline): Whisper small ≈ 460 MB, PANNs CNN14 ≈
330 MB, SpeechBrain ECAPA ≈ 80 MB, all-MiniLM-L6-v2 ≈ 90 MB. Silero VAD ships inside its package.

## Connect a client

Claude Code:

```bash
claude mcp add audio-intel -- uv --directory /path/to/audio_mcp run audio-mcp
```

Claude Desktop (`claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "audio-intel": {
      "command": "uv",
      "args": ["--directory", "/path/to/audio_mcp", "run", "audio-mcp"],
      "env": {"AUDIO_MCP_ALLOWED_ROOTS": "[\"/path/to/your/audio\"]"}
    }
  }
}
```

Only files under `allowed_roots` can be read (default: your home directory). Paths are resolved, symlinks
included, before the check.

## Tools

| Tool | What it does |
|---|---|
| `analyze_audio` | Task list or `"auto"` (plans from speech ratio, duration, goal text and budget; plan returned with reasons) |
| `transcribe_audio` | Segments + word timestamps, language runs, optional speakers, code-switching, English translation |
| `detect_audio_events` | AudioSet sound events (PANNs) as time intervals, filterable by category or label |
| `extract_audio_features` | Level, silence, SNR estimate, spectral shape, bands, pitch, tempo/onsets, with units |
| `inspect_audio_quality` | Clipping, low level, silence, noise, hum, echo, bandwidth, DC; optional preprocessed variant |
| `get_audio_timeline` | Event graph: speech, turns, events, silence, overlaps, transitions + relation edges |
| `ask_audio` | Question answering with timestamped evidence and `support: direct / inferred / insufficient` |
| `summarize_audio` | Extractive summary, topics, key moments, every sentence linked to evidence ids |
| `compare_audio` | Pairwise measured differences, or baseline z-score anomalies localized in time |
| `search_audio_library` | Persistent SQLite FTS5 + vector library: index files/folders, search by text/filters/example clip |
| `analyze_audio_stream` | Chunked sessions (PCM or self-contained files), provisional results, final reconciliation |
| `export_audio_analysis` | json, markdown, srt, vtt, csv, rttm (optional redaction) |
| `get_job` | Status/progress/cancel for async jobs (long audio runs as a job automatically) |

Full schemas and example success/error responses: [docs/tools.md](docs/tools.md). What is implemented,
partial, or future: [docs/capabilities.md](docs/capabilities.md).

## Example workflows

**Meeting.** `analyze_audio {"source": {"type": "file_path", "path": "/audio/standup.m4a"}, "goal": "meeting"}`
plans transcription, language, diarization, events, summary, timeline and uncertainty. Then reuse the returned
`audio_id` (cached, nothing recomputed): `ask_audio {"source": {"type": "audio_id", "audio_id": "aud_..."},
"question": "When does someone mention the deadline?"}` and `export_audio_analysis {"audio_id": "aud_...",
"format": "srt"}`.

**Machine baseline.** `compare_audio {"sources": [<today.wav>], "baseline": [<healthy_1.wav>, <healthy_2.wav>]}`
returns per-feature z-scores and the seconds where the target departs from the baseline distribution;
interpretations are labelled as such and never stated as causes.

**Environmental recording.** `detect_audio_events {"source": ..., "categories": ["animal", "vehicle"]}`, then
`get_audio_timeline` and `ask_audio "What happens immediately before the alarm sounds?"`.

**Library.** `search_audio_library {"action": "index", "paths": ["/audio/calls"]}`, then
`{"action": "search", "query": "refund request", "filters": {"language": "en"}}` or search by an example clip
with `"query_audio"`.

**Streaming.** `analyze_audio_stream` with `start` (`chunk_format: "pcm"` + rate/channels/sample format, or
`"file"`), repeated `push` (base64 `data` or `path`), then `finish` for the full analysis and a reconciliation of
provisional vs final results.

## Configuration

TOML file (`AUDIO_MCP_CONFIG=/path/config.toml`, see `config.example.toml`) and/or env vars
`AUDIO_MCP_<NAME>`. Full table: [docs/configuration.md](docs/configuration.md). Key settings:

| Setting | Default | |
|---|---|---|
| `allowed_roots` | home dir | folders audio may be read from |
| `max_file_mb` / `max_upload_mb` / `max_duration_seconds` | 500 / 50 / 7200 | input limits |
| `allow_url_fetch` | false | URL sources (private/loopback/metadata IPs always blocked) |
| `asr_model` | `small` | faster-whisper size (`tiny` … `large-v3`) |
| `async_threshold_seconds` | 600 | longer audio runs as a job (`async_mode: "auto"`) |
| `retention` / `retention_ttl_hours` | `ttl` / 24 | stored audio and results are deleted after the TTL |
| `llm_provider` | `none` | `anthropic`, `openai_compatible`, or `ollama` (text only, opt-in) |
| `redact_outputs` | false | redact emails, phone numbers, card numbers (+ your patterns) in all outputs |

## Privacy

- Audio is processed locally and never sent anywhere.
- The only outbound traffic is (a) model downloads on first use (Hugging Face, Zenodo, GitHub) and (b) the LLM,
  if you enable one. A local Ollama/OpenAI-compatible server on localhost counts as local; any other host sets
  `processing.data_sent_off_machine: true` on that response.
- Stored audio copies and cached results live under `data_dir` and are swept after `retention_ttl_hours`
  (`retention = "none"` deletes everything at shutdown). The search library is persistent
  (`library_retention = "keep"`) and stores derived data and file paths, never audio.
- Exports written by `export_audio_analysis` to `export_dir` are yours to keep: retention never deletes them.
  They are redacted when the call sets `redact: true` or the server sets `redact_outputs = true`.
- Logs go to stderr (and `log_file`) and never contain transcript text unless `log_sensitive = true`.
- Integrity indicators are signal observations only; the server never states that audio is authentic, edited
  or synthetic, and never infers emotion.

## Supported input

Any audio ffmpeg can decode: WAV, FLAC, MP3, OGG/Opus, M4A/AAC, WebM, AIFF, WMA, and more; any sample rate and
channel count (models get a 16 kHz mono mix; measurements use the native signal). Sources: `file_path`,
`base64`, `url` (off by default), `audio_id`. Playlist and concat inputs (HLS `.m3u8`, ffconcat, DASH, image
sequences) are rejected because they make ffmpeg read other files.

Limits and memory: audio longer than `max_duration_seconds` (default 7200 s = 2 h) is rejected. Decoded audio is
held in RAM as float32 at the native rate plus a 16 kHz mono copy: about 1.6 GB per hour of 48 kHz stereo
(about 0.9 GB per hour of 44.1 kHz mono), with transient peaks of roughly 2-3x that during decoding and feature
extraction. The two most recently used recordings stay decoded in memory. Raise the limit only with RAM to match.

Language coverage follows Whisper (≈ 99 languages); translation is to English only.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `MODEL_UNAVAILABLE: ffmpeg/ffprobe not found` | Install ffmpeg (above) or set `ffmpeg_path` / `ffprobe_path` |
| A stage in `stages_skipped` with "not installed (uv sync --extra models)" | Install the `models` extra |
| `stages_skipped` with a model load error | First-run download failed (network/proxy); check `models_dir` is writable and retry |
| SpeechBrain symlink / privilege error on Windows | Already handled (models are copied, not symlinked); update and retry |
| PANNs tries to run `wget` | The server pre-seeds `~/panns_data/class_labels_indices.csv`; make sure the home dir is writable |
| Slow on CPU | Use `asr_model = "base"` or `"tiny"`, `budget: "fast"`, or a CUDA GPU; long files run as jobs |
| `PATH_NOT_ALLOWED` | Add the folder to `allowed_roots` |
| Client reports a broken connection | Nothing may print to stdout; the server redirects stray library prints to stderr — report any that leak |

## Development

```bash
uv sync --extra dev --extra models
uv run pytest -m "not slow and not long"   # fast suite: fake backends, no downloads
uv run pytest -m slow                      # real models
uv run pytest -m long                      # 30-minute file / async path
uv run python scripts/gen_tool_docs.py --write-examples   # refresh docs/tools.md + examples
```

Evaluation and benchmarks: [docs/evaluation.md](docs/evaluation.md).

## Licensing

Code: see repository license. Models and data, each under its own license:
faster-whisper (MIT) and Whisper weights (MIT); PANNs code (MIT) and CNN14 weights (CC BY 4.0, trained on
AudioSet); SpeechBrain (Apache-2.0) and spkrec-ecapa-voxceleb (Apache-2.0); all-MiniLM-L6-v2 (Apache-2.0);
silero-vad (MIT); pyannote.audio (MIT) with gated model terms; AudioSet ontology labels (CC BY 4.0). Test
fixtures: [tests/fixtures/audio/SOURCES.md](tests/fixtures/audio/SOURCES.md).
