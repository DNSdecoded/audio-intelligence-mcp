# Audio Intelligence MCP Server

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://github.com/DNSdecoded/audio-intelligence-mcp/blob/main/LICENSE)
![Python 3.12 | 3.13](https://img.shields.io/badge/python-3.12%20%7C%203.13-blue)
![Platforms: Windows | macOS | Linux](https://img.shields.io/badge/platform-windows%20%7C%20macos%20%7C%20linux-lightgrey)

[![29-second overview of Audio Intelligence MCP](https://github.com/DNSdecoded/audio-intelligence-mcp/raw/main/docs/media/launch.gif)](https://cdn.jsdelivr.net/gh/DNSdecoded/audio-intelligence-mcp@main/docs/media/launch.mp4)

▶ [Watch with sound (MP4, 29 s)](https://cdn.jsdelivr.net/gh/DNSdecoded/audio-intelligence-mcp@main/docs/media/launch.mp4)

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

- Windows, macOS or Linux (CI runs the test suite on all three)
- Python 3.12 or 3.13 and [uv](https://docs.astral.sh/uv/)
- ffmpeg and ffprobe on `PATH` (or set `ffmpeg_path` / `ffprobe_path`):
  - Windows: `winget install Gyan.FFmpeg`
  - macOS: `brew install ffmpeg`
  - Debian/Ubuntu: `sudo apt-get install ffmpeg`
- CPU is enough. CUDA is used automatically when available (`device = "auto"`).

## Install

From PyPI (once released):

```bash
pip install "audio-intelligence-mcp[models]"     # or run without installing: uvx --from "audio-intelligence-mcp[models]" audio-mcp
```

From source:

```bash
git clone https://github.com/DNSdecoded/audio-intelligence-mcp.git
cd audio-intelligence-mcp
uv sync --extra models            # local models (faster-whisper, PANNs, SpeechBrain, silero-vad, MiniLM)
uv sync --extra models --extra llm        # + optional Claude backend (anthropic SDK)
uv sync --extra models --extra pyannote   # + optional pyannote diarization (needs HF_TOKEN)
```

Without `--extra models` the server still starts: model-backed stages are reported in `stages_skipped` with the
reason, and the DSP tools (features, quality, integrity, comparison) work.

**First-run downloads** (cached in `models_dir`, then fully offline): Whisper small ≈ 460 MB, PANNs CNN14 ≈
330 MB, SpeechBrain ECAPA ≈ 80 MB, all-MiniLM-L6-v2 ≈ 90 MB. Silero VAD ships inside its package.

## Connect a client

Claude Code (PyPI release):

```bash
claude mcp add audio-intel -- uvx --from "audio-intelligence-mcp[models]" audio-mcp
```

Claude Code (source checkout):

```bash
claude mcp add audio-intel -- uv --directory /path/to/audio-intelligence-mcp run audio-mcp
```

Claude Desktop (`claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "audio-intel": {
      "command": "uv",
      "args": ["--directory", "/path/to/audio-intelligence-mcp", "run", "audio-mcp"],
      "env": {"AUDIO_MCP_ALLOWED_ROOTS": "[\"/path/to/your/audio\"]"}
    }
  }
}
```

On Windows, use forward slashes or escaped backslashes in JSON paths (`"C:/audio"` or `"C:\audio"`).

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

Full schemas and example success/error responses: [docs/tools.md](https://github.com/DNSdecoded/audio-intelligence-mcp/blob/main/docs/tools.md). What is implemented,
partial, or future: [docs/capabilities.md](https://github.com/DNSdecoded/audio-intelligence-mcp/blob/main/docs/capabilities.md).

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
`AUDIO_MCP_<NAME>`. Full table: [docs/configuration.md](https://github.com/DNSdecoded/audio-intelligence-mcp/blob/main/docs/configuration.md). Key settings:

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

## Evaluation

Measured on public datasets with the default local models on a 4-core CPU (Whisper small int8, PANNs CNN14,
ECAPA). Full tables, per-file JSON and before/after numbers: [docs/eval/RESULTS.md](https://github.com/DNSdecoded/audio-intelligence-mcp/blob/main/docs/eval/RESULTS.md); method,
datasets and caveats: [docs/evaluation.md](https://github.com/DNSdecoded/audio-intelligence-mcp/blob/main/docs/evaluation.md).

| Capability | Dataset | Result |
|---|---|---|
| Transcription | LibriSpeech test-clean / test-other | **4.1 % / 6.1 % WER** (Whisper small published: 3.4 % / 7.6 %) |
| Noise robustness | LibriSpeech + real ambient noise, 20 / 10 / 5 / 0 dB SNR | 4.3 / 5.0 / 8.4 / 14.5 % WER |
| Formats | 12 containers/codecs/rates (wav 8–48 kHz, flac, mp3, ogg, opus, m4a, aac, webm, aiff, wma) | all decode; 0.6 % WER |
| Multilingual | FLEURS en / es / de / fr / zh / ar / hi / te | 6.7 / 5.7 / 12.9 / 13.7 / 24.6 (CER) / 37.7 / 58.8 / 103 % WER; language detected 88–100 % |
| Speaker diarization | AMI test meetings (real, with overlap) | 38.2 % DER (30.4 % with 0.25 s collar); most error is overlapped speech |
| Sound events | ESC-50, 200 clips, zero-shot | 73 % clip recall, 65 % top-1 (lenient AudioSet labels) |
| Quality diagnosis | 90 files with 8 injected defects | 88 % exact issue set; per-issue recall 100 % |
| Speed | 10-min file, CPU | transcription 0.58× real time, events 0.23×, diarization 0.57×; peak 2.1 GB RAM |

Low-resource languages (Arabic, Hindi, Telugu) are not usable with `asr_model = "small"`; use a larger model and
pass `language`.

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

Releasing: bump `version` in `pyproject.toml`, then push a `vX.Y.Z` tag. `.github/workflows/publish.yml` builds
the package and uploads it to PyPI through trusted publishing (no API token stored).

Evaluation and benchmarks: [docs/evaluation.md](https://github.com/DNSdecoded/audio-intelligence-mcp/blob/main/docs/evaluation.md). Design spec, implementation plan, and build
notes (design rulings and known limits): [docs/superpowers/](https://github.com/DNSdecoded/audio-intelligence-mcp/tree/main/docs/superpowers).

Project layout:

```
src/audio_mcp/
  server.py         MCP entry point (stdio), tool registration, lifespan
  tools/            request schemas and tool handlers
  ingest/           source resolution, path/URL guards, ffmpeg probe and decode
  store/            content-addressed audio store, result cache, retention
  pipeline/         stage DAG, auto planner, runner
  backends/         model adapters (Whisper, PANNs, ECAPA, Silero, MiniLM, LLMs) + fake backends for tests
  dsp/              features, quality, preprocessing, integrity, speech characteristics
  analysis/ timeline/ reasoning/ compare/ library/ stream/ export/ eval/
scripts/            benchmark, evaluate, docs generator
tests/              unit, integration, e2e (stdio)
```

## Known limitations

- Confidence values are raw model probabilities or heuristics, never calibrated.
- Default diarization (ECAPA clustering) does not detect overlapping speech; install the `pyannote` extra for that.
- Accuracy on low-resource languages with the default `small` model is poor (see Evaluation).
- Translation is to English only (Whisper).
- Integrity indicators are signal heuristics, not a deepfake or tamper detector.
- Malformed tool arguments are rejected by the MCP layer before the server sees them, so they come back as MCP
  tool errors rather than the JSON envelope.

Full list: [docs/capabilities.md](https://github.com/DNSdecoded/audio-intelligence-mcp/blob/main/docs/capabilities.md) and
[docs/superpowers/build-notes.md](https://github.com/DNSdecoded/audio-intelligence-mcp/blob/main/docs/superpowers/build-notes.md).

## License

The code in this repository is released under the [Apache License 2.0](https://github.com/DNSdecoded/audio-intelligence-mcp/blob/main/LICENSE) (see also [NOTICE](https://github.com/DNSdecoded/audio-intelligence-mcp/blob/main/NOTICE)).

Third-party models and data keep their own licenses, which the Apache License does not cover:
faster-whisper (MIT) and Whisper weights (MIT); PANNs code (MIT) and CNN14 weights (CC BY 4.0, trained on
AudioSet); SpeechBrain (Apache-2.0) and spkrec-ecapa-voxceleb (Apache-2.0); all-MiniLM-L6-v2 (Apache-2.0);
silero-vad (MIT); pyannote.audio (MIT) with gated model terms; AudioSet ontology labels (CC BY 4.0). Test
fixtures: `speech_es.ogg` is a LibriVox public-domain excerpt; the English fixtures are synthesized locally;
details in [tests/fixtures/audio/SOURCES.md](https://github.com/DNSdecoded/audio-intelligence-mcp/blob/main/tests/fixtures/audio/SOURCES.md). Check each model's terms before
commercial use.
