# Audio Intelligence MCP Server Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A local-first, cross-platform MCP server (stdio) exposing 12 audio-analysis tools plus a job tool. Every tool returns a common JSON envelope with evidence, honest confidence and explicit skipped/failed stages.

**Architecture:**
- `server.py` holds only FastMCP tool registration. Each tool calls a function in `tools/handlers.py` that receives an `AppContext` (settings, store, cache, backends, jobs, library, streams).
- Analysis runs as a stage DAG (`pipeline/`) over decoded audio (`ingest/`). Stage results are cached per `(audio_id, stage, config hash)`.
- Model adapters live in `backends/` behind small Protocols. A `fake` profile gives deterministic, model-free backends, so CI and e2e tests run without downloads.

**Tech Stack:**
- Base: Python 3.12 (uv), `mcp` (FastMCP), pydantic v2, pydantic-settings, numpy, scipy, soundfile, platformdirs, httpx, SQLite FTS5, ffmpeg/ffprobe subprocess.
- `models` extra: faster-whisper, silero-vad, panns-inference (+torch, librosa), speechbrain, sentence-transformers.
- `pyannote` extra: pyannote.audio.
- `dev` extra: pytest, pytest-asyncio, jiwer, psutil, pyttsx3.

**Spec:** `docs/superpowers/specs/2026-10-17-audio-intelligence-mcp-design.md`. Read it before any task. § numbers refer to it.

## Global Constraints

- Python `>=3.12,<3.14`. Pin with `uv python pin 3.12`.
- Every tool response uses the §5.1 envelope: `status` ∈ `success|partial|processing|error`, `audio_id`, `result` (or `analysis` for `analyze_audio`), `warnings`, `errors`, and `processing` with `schema_version: "1.0"`.
- Validation errors before any work use the error-only shape `{"status":"error","error":{"code","message","retryable"}}`.
- Error codes are exactly the §5.1 list. Do not invent others. Warning codes are free-form strings.
- Time is in seconds as float rounded to 3 decimals. Frequency is in Hz, level in dBFS, and amplitude linear on a −1..1 scale.
- `confidence` appears only when a model or rule produced it, always with `confidence_kind` ∈ `model_probability|heuristic`. Never write "calibrated".
- Speaker labels are `speaker_1..N`, numbered by first appearance.
- `audio_id` = `"aud_" + sha256(file_bytes).hexdigest()[:16]`.
- Original audio is never modified. Preprocessed variants are stored separately, with a transform list.
- No `shell=True`. Use `pathlib` everywhere. Paths are resolved (`Path.resolve(strict=True)`) before the allowed-roots check. Comparison is case-insensitive on Windows and macOS.
- Logs never include transcript text or audio bytes unless `log_sensitive=true`. Never write to stdout: it carries the MCP protocol.
- Any LLM backend is off by default. `data_sent_off_machine` is true only for an LLM call to a non-loopback host.
- No authenticity verdicts, and no emotion presented as fact.
- Each heavy model loads lazily on first use. When it can't load, the stage is reported in `stages_skipped` with the reason, and the server must not crash.
- Commit after every task. Commit messages carry no Claude/Anthropic attribution lines (user's CLAUDE.md).

## Review Focus

1. **Filenames and paths with spaces, unicode, `..`, or symlinks escaping an allowed root.** Expect `PATH_NOT_ALLOWED` or success, never a traceback. Test owner: Task 3.
2. **Very short audio (< 0.5 s) and all-silence audio passed to models.** Expect `success`/`partial` with empty segments/events and a warning, not a crash in windowing, VAD or diarization. Test owner: Task 11.
3. **Calling `ask_audio` or `get_audio_timeline` with an `audio_id` whose results are not cached yet.** Expect the stages to run on demand. An unknown id returns `AUDIO_NOT_FOUND`. Test owner: Task 18 (and Task 15).
4. **Multichannel/odd-rate inputs (6-channel 8 kHz, 96 kHz/24-bit FLAC).** Expect metadata to report the native values while models get 16 kHz mono. Test owner: Task 5.
5. **The same file ingested twice via different source types (path, then base64).** Expect the same `audio_id`, with the second call listing cached stages in `processing.cache_hits`. Test owner: Task 12.

---

## File Structure

```
pyproject.toml, uv.lock, .python-version, .gitignore, config.example.toml, .env.example
.github/workflows/ci.yml
src/audio_mcp/
  __init__.py            __version__ = "0.1.0"
  __main__.py            `python -m audio_mcp` → server.main()
  config.py              Settings, load_settings
  envelope.py            ErrorCode, AudioMCPError, ModelInfo, ProcessingInfo, build_response, error_response, warning
  types.py               Interval, Word, Segment, Event, SpeakerTurn, Evidence, r3()
  logging_setup.py       JSON stderr logging + sensitive-field filter
  context.py             AppContext, build_app_context
  ingest/sources.py      AudioSource union, check_path_allowed, materialize_source
  ingest/decode.py       ProbeInfo, DecodedAudio, find_ffmpeg, probe, decode, resample
  store/audio_store.py   AudioRecord, AudioStore
  store/cache.py         ResultCache, config_hash
  store/retention.py     sweep, cleanup_temp
  backends/base.py       Protocols, ASRResult, TagResult, Backends, UnavailableBackend, build_backends
  backends/audioset_labels.csv
  backends/energy_vad.py EnergyVAD
  backends/silero.py     SileroVAD
  backends/whisper.py    FasterWhisperASR, translate_segments
  backends/panns.py      PannsTagger
  backends/ecapa.py      EcapaEmbedder
  backends/pyannote_diar.py PyannoteDiarizer
  backends/text_embed.py MiniLMEmbedder
  backends/llm.py        NoLLM, HttpLLM, is_loopback
  backends/fake.py       fake_backends()
  dsp/features.py        FEATURE_GROUPS, extract_features, frame_rms
  dsp/quality.py         diagnose_quality
  dsp/preprocess.py      PREPROCESS_OPS, apply_preprocessing
  dsp/integrity.py       INTEGRITY_LIMITATIONS, integrity_indicators
  dsp/speech.py          speech_characteristics
  pipeline/stages.py     Stage, StageContext, STAGES
  pipeline/planner.py    TASK_TO_STAGES, plan_auto
  pipeline/runner.py     AnalysisOutcome, run_analysis
  analysis/events.py     EVENT_CATEGORIES, frames_to_events
  analysis/diarize.py    cluster_speakers, assign_speakers
  analysis/uncertainty.py find_uncertainties
  timeline/graph.py      build_timeline, render_text
  reasoning/ask.py       classify_intent, answer_question
  reasoning/summarize.py summarize
  reasoning/llm_grounded.py build_evidence_pack, llm_answer, llm_summarize
  compare/compare.py     window_features, compare_pair, baseline_anomalies
  library/index.py       SUPPORTED_EXTENSIONS, Library
  stream/sessions.py     StreamSessions
  export/exporters.py    EXPORT_FORMATS, export_analysis
  export/redact.py       redact_text, redact_analysis
  jobs.py                JobInfo, JobManager
  eval/metrics.py        der, load_rttm, word_error_rate, event_prf
  tools/schemas.py       request models
  tools/handlers.py      one function per tool: (AppContext, Request) -> dict
  server.py              FastMCP registration, lifespan, main()
scripts/benchmark.py, scripts/evaluate.py, scripts/gen_tool_docs.py
tests/conftest.py, tests/fixtures/synth.py, tests/fixtures/make_speech.py, tests/fixtures/audio/…
tests/unit/…, tests/integration/…, tests/e2e/…
docs/tools.md, docs/configuration.md, docs/capabilities.md, docs/evaluation.md, docs/examples/*.json, README.md
```

Test markers (register in `pyproject.toml`):
- `slow`: needs real models. A conftest hook skips it with the reason when `import faster_whisper` fails.
- `long`: the 30-min fixture.
- The CI fast job runs `-m "not slow and not long"`.

Shared test helpers live in `tests/conftest.py`:
- `fp(path) -> dict`: a `file_path` source dict.
- `seg(id, start, end, text, conf=None, words=())` and `w(start, end, text, conf)`, plus `evt(id, label, start, end, conf)`. Each builds the dict form of the types.
- `wait_until(pred, timeout=30)`.

---

# Phase 1 — Foundation

### Task 1: Project scaffold, config, CI skeleton

**Files:**
- Create: `pyproject.toml`, `.python-version`, `.gitignore`, `config.example.toml`, `.env.example`, `src/audio_mcp/__init__.py`, `src/audio_mcp/__main__.py`, `src/audio_mcp/config.py`, `.github/workflows/ci.yml`, `tests/conftest.py`
- Test: `tests/unit/test_config.py`

**Interfaces:**
- Produces: `Settings` (pydantic-settings, env prefix `AUDIO_MCP_`; list fields accept JSON strings) and `load_settings(config_path: Path | None = None) -> Settings`.
  - The TOML path comes from the argument, else env `AUDIO_MCP_CONFIG`, else none.
  - Precedence: env > TOML > defaults.
  - Fields and defaults:
    - `allowed_roots: list[Path] = [Path.home()]`
    - `data_dir: Path = platformdirs.user_data_path("audio-mcp")`
    - `temp_dir`, `export_dir`, `models_dir`: `Path | None = None`. None resolves to `data_dir/"tmp"`, `"exports"`, `"models"` via the properties `resolved_temp_dir`, `resolved_export_dir`, `resolved_models_dir`.
    - `max_file_mb: int = 500`, `max_upload_mb: int = 50`, `max_duration_seconds: float = 14400`
    - `allow_url_fetch: bool = False`, `url_timeout_seconds: float = 30`
    - `ffmpeg_path: Path | None = None`, `ffprobe_path: Path | None = None`
    - `backend_profile: Literal["real","fake"] = "real"`, `device: Literal["auto","cpu","cuda"] = "auto"`
    - `asr_model: str = "small"`, `asr_compute_type: str = "int8"`
    - `diarization_backend: Literal["ecapa","pyannote"] = "ecapa"`, `diarization_threshold: float = 0.55`, `max_speakers: int = 8`
    - `event_threshold: float = 0.3`, `low_confidence_threshold: float = 0.5`, `speech_ratio_threshold: float = 0.2`
    - `auto_preprocess: bool = False`
    - `async_threshold_seconds: float = 600`, `max_concurrent_jobs: int = 1`, `stage_timeout_seconds: float = 1800`
    - `retention: Literal["none","ttl","keep"] = "ttl"`, `retention_ttl_hours: float = 24`, `library_retention: Literal["keep","ttl"] = "keep"`
    - `log_level: str = "INFO"`, `log_sensitive: bool = False`, `log_file: Path | None = None`
    - `llm_provider: Literal["none","anthropic","openai_compatible","ollama"] = "none"`, `llm_model: str | None = None`, `llm_base_url: str | None = None`, `llm_api_key_env: str = "AUDIO_MCP_LLM_API_KEY"`, `llm_timeout_seconds: float = 60`
    - `redact_outputs: bool = False`, `redact_extra_patterns: list[str] = []`
  - Validator: `llm_provider != "none"` requires `llm_model`. `ollama` defaults `llm_base_url` to `http://localhost:11434/v1`.
  - `load_settings` creates the resolved dirs.

- [ ] **Step 1: Write `pyproject.toml`** with hatchling, `src` layout, script `audio-mcp = "audio_mcp.server:main"`, the extras from the Tech Stack line, pytest markers `slow`/`long`, and `asyncio_mode = "auto"`. Run `uv python pin 3.12` and `uv sync --extra dev`.
- [ ] **Step 2: Write failing tests**

```python
def test_defaults(monkeypatch):
    monkeypatch.delenv("AUDIO_MCP_CONFIG", raising=False)
    s = load_settings()
    assert s.allow_url_fetch is False and s.retention == "ttl" and s.retention_ttl_hours == 24
    assert s.llm_provider == "none" and s.backend_profile == "real"

def test_toml_then_env_precedence(tmp_path, monkeypatch):
    p = tmp_path / "c.toml"; p.write_text('max_upload_mb = 7\nasr_model = "base"\n')
    monkeypatch.setenv("AUDIO_MCP_ASR_MODEL", "tiny")
    s = load_settings(p); assert s.max_upload_mb == 7 and s.asr_model == "tiny"

def test_llm_requires_model(monkeypatch):
    monkeypatch.setenv("AUDIO_MCP_LLM_PROVIDER", "anthropic")
    with pytest.raises(ValueError): load_settings()

def test_ollama_default_base_url(monkeypatch):
    monkeypatch.setenv("AUDIO_MCP_LLM_PROVIDER", "ollama"); monkeypatch.setenv("AUDIO_MCP_LLM_MODEL", "llama3")
    assert load_settings().llm_base_url == "http://localhost:11434/v1"
```

- [ ] **Step 3: Run** `uv run pytest tests/unit/test_config.py -v`. Expected: FAIL (module missing).
- [ ] **Step 4: Implement `config.py`.**
  - `config.example.toml` lists every field with its default and a one-line comment.
  - `.env.example` lists `AUDIO_MCP_CONFIG`, `AUDIO_MCP_LLM_API_KEY` and `HF_TOKEN`.
  - `tests/conftest.py` gets the `settings(tmp_path)` fixture: `allowed_roots=[tmp_path, FIXTURES_DIR]`, `data_dir=tmp_path/"data"`, `backend_profile="fake"`.
- [ ] **Step 5: Run** the same command. Expected: 4 passed.
- [ ] **Step 6: Write `.github/workflows/ci.yml`.**
  - Job `fast`: matrix `windows-latest, macos-latest, ubuntu-latest`. Uses `astral-sh/setup-uv`, `uv python install 3.12`, and installs ffmpeg (`choco install ffmpeg -y` / `brew install ffmpeg` / `sudo apt-get install -y ffmpeg`). Then `uv sync --extra dev` and `uv run pytest -m "not slow and not long"`.
  - Job `slow`: ubuntu only, `uv sync --extra dev --extra models`, `actions/cache` on `~/.local/share/audio-mcp/models` and `~/panns_data`, then `uv run pytest -m slow`.
- [ ] **Step 7: Commit** `feat: project scaffold, settings, CI skeleton`.

### Task 2: Envelope, error codes, shared types

**Files:**
- Create: `src/audio_mcp/envelope.py`, `src/audio_mcp/types.py`
- Test: `tests/unit/test_envelope.py`, `tests/unit/test_types.py`

**Interfaces:**
- Produces from `envelope.py`:
  - `class ErrorCode(StrEnum)`: the 17 §5.1 codes.
  - `class AudioMCPError(Exception)`: `__init__(code: ErrorCode | str, message: str, *, retryable: bool = False, stage: str | None = None)`, `.to_dict() -> dict`.
  - `class ModelInfo(BaseModel)`: `stage: str, name: str, version: str, local: bool = True`.
  - `class ProcessingInfo(BaseModel)`:
    - `duration_seconds: float = 0`
    - `stages_run: list[str] = []`, `stages_skipped: list[dict] = []` (each `{"stage","reason"}`), `cache_hits: list[str] = []`
    - `models_used: list[ModelInfo] = []`
    - `data_sent_off_machine: bool = False`, `schema_version: str = "1.0"`
    - `plan: list[dict] | None = None`
  - `build_response(*, result: dict | None, audio_id: str | None = None, payload_key: str = "result", warnings=(), errors=(), processing: ProcessingInfo | None = None, job_id: str | None = None, status: str | None = None) -> dict`
    - When `status` is None it is derived: `processing` if `job_id`; `error` if errors and no result; `partial` if errors and a result; else `success`.
    - The `audio_id`/`job_id` keys are omitted when None.
  - `error_response(err: AudioMCPError) -> dict`: the error-only shape.
  - `warning(code: str, message: str) -> dict`.
- Produces from `types.py`:
  - `r3(x) -> float`.
  - Pydantic models. Time fields are rounded via a validator, and dumps use `exclude_none=True`.
    - `Interval(start, end)`
    - `Word(start, end, text, confidence=None, confidence_kind=None)`
    - `Segment(id, start, end, text, language=None, speaker_id=None, confidence=None, confidence_kind=None, words=[], translation=None)`
    - `Event(id, label, start, end, confidence=None, confidence_kind=None, source)`
    - `SpeakerTurn(speaker_id, start, end)`
    - `Evidence(id, start, end, source, text=None, label=None, kind: Literal["detected","interpretation"]="detected", confidence=None, confidence_kind=None)`
  - Ids: segments `seg_0001`, events `evt_0001`. Evidence reuses the referenced node id.

- [ ] **Step 1: Write failing tests**

```python
def test_status_derivation():
    assert build_response(result={"a": 1})["status"] == "success"
    assert build_response(result={"a": 1}, errors=[{"code": "STAGE_FAILED"}])["status"] == "partial"
    assert build_response(result=None, errors=[{"code": "STAGE_FAILED"}])["status"] == "error"
    r = build_response(result=None, job_id="job_1"); assert r["status"] == "processing" and r["job_id"] == "job_1"

def test_payload_key_and_schema_version():
    r = build_response(result={"x": 1}, payload_key="analysis", audio_id="aud_0123456789abcdef")
    assert "analysis" in r and "result" not in r and r["processing"]["schema_version"] == "1.0"
    assert r["warnings"] == [] and r["errors"] == []

def test_error_response_shape():
    r = error_response(AudioMCPError(ErrorCode.UNSUPPORTED_AUDIO_FORMAT, "nope"))
    assert r == {"status": "error", "error": {"code": "UNSUPPORTED_AUDIO_FORMAT", "message": "nope", "retryable": False}}

def test_segment_rounds_and_omits_none():
    d = Segment(id="seg_0001", start=1.23456, end=2.0, text="hi").model_dump(exclude_none=True)
    assert d["start"] == 1.235 and "confidence" not in d and "speaker_id" not in d
```

- [ ] **Step 2: Run** `uv run pytest tests/unit/test_envelope.py tests/unit/test_types.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement** both modules.
- [ ] **Step 4: Run** the same command. Expected: PASS.
- [ ] **Step 5: Commit** `feat: response envelope, error codes, shared types`.

### Task 3: Source resolution and path guard

**Files:**
- Create: `src/audio_mcp/ingest/__init__.py`, `src/audio_mcp/ingest/sources.py`
- Test: `tests/unit/test_sources.py`

**Interfaces:**
- Consumes: `Settings`, `AudioMCPError`, `ErrorCode`.
- Produces:
  - Pydantic discriminated union `AudioSource` on `type`:
    - `FilePathSource(type="file_path", path: str)`
    - `Base64Source(type="base64", data: str, filename_hint: str | None = None)`
    - `UrlSource(type="url", url: str)`
    - `AudioIdSource(type="audio_id", audio_id: str = Field(pattern=r"^aud_[0-9a-f]{16}$"))`
  - `check_path_allowed(path: str | Path, roots: list[Path]) -> Path`:
    - Resolves with `strict=True`. Missing → `FILE_NOT_FOUND`. Not under any resolved root → `PATH_NOT_ALLOWED`. A directory → `INVALID_INPUT`.
    - Comparison uses `os.path.normcase` on Windows and `casefold` on macOS.
  - `@dataclass MaterializedSource(path: Path, original_name: str, is_temp: bool)`.
  - `materialize_source(src: FilePathSource | Base64Source | UrlSource, settings) -> MaterializedSource`:
    - **File:** the checked path. Size > `max_file_mb` → `FILE_TOO_LARGE`.
    - **Base64:**
      - Precheck `len(data) * 3 / 4 > max_upload_mb * 2**20` → `FILE_TOO_LARGE` before decoding. Invalid base64 → `INVALID_INPUT`.
      - Writes `resolved_temp_dir/<uuid4 hex><ext>`, where `ext` is the `filename_hint` suffix if it matches `^\.[A-Za-z0-9]{1,5}$`, else `.bin`.
    - **URL:**
      - Disabled → `URL_FETCH_DISABLED`. Non-http(s) scheme → `INVALID_INPUT`.
      - Resolve the host with `socket.getaddrinfo`. Any private, loopback, link-local, reserved or multicast IP → `URL_FETCH_FAILED` "blocked address".
      - Stream with httpx (`follow_redirects=False`, timeout `url_timeout_seconds`) and abort past `max_file_mb` → `FILE_TOO_LARGE`. Other failures → `URL_FETCH_FAILED` (retryable=True).
  - `AudioIdSource` is not handled here (store resolves it).

- [ ] **Step 1: Write failing tests**

```python
def test_path_inside_root(tmp_path):
    f = tmp_path / "a b ü.wav"; f.write_bytes(b"x")
    assert check_path_allowed(str(f), [tmp_path]) == f.resolve()

def test_dotdot_escape(tmp_path):
    root = tmp_path / "root"; root.mkdir(); (tmp_path / "secret.wav").write_bytes(b"x")
    with pytest.raises(AudioMCPError) as e: check_path_allowed(str(root / ".." / "secret.wav"), [root])
    assert e.value.code == ErrorCode.PATH_NOT_ALLOWED

def test_symlink_escape(tmp_path):
    root = tmp_path / "root"; root.mkdir(); target = tmp_path / "out.wav"; target.write_bytes(b"x")
    link = root / "link.wav"
    try: link.symlink_to(target)
    except OSError: pytest.skip("symlinks not permitted")
    with pytest.raises(AudioMCPError) as e: check_path_allowed(str(link), [root])
    assert e.value.code == ErrorCode.PATH_NOT_ALLOWED

def test_missing_file(tmp_path):
    with pytest.raises(AudioMCPError) as e: check_path_allowed(str(tmp_path / "nope.wav"), [tmp_path])
    assert e.value.code == ErrorCode.FILE_NOT_FOUND

def test_base64_too_large(settings):
    settings.max_upload_mb = 1
    with pytest.raises(AudioMCPError) as e: materialize_source(Base64Source(type="base64", data="A" * (2 * 1024 * 1024)), settings)
    assert e.value.code == ErrorCode.FILE_TOO_LARGE

def test_base64_bad_hint_is_neutralised(settings):
    m = materialize_source(Base64Source(type="base64", data=base64.b64encode(b"abc").decode(), filename_hint="../../evil.sh;rm"), settings)
    assert m.path.parent == settings.resolved_temp_dir and m.path.suffix == ".bin" and m.is_temp

def test_url_disabled(settings):
    with pytest.raises(AudioMCPError) as e: materialize_source(UrlSource(type="url", url="https://x.org/a.mp3"), settings)
    assert e.value.code == ErrorCode.URL_FETCH_DISABLED

def test_url_loopback_blocked(settings):
    settings.allow_url_fetch = True
    with pytest.raises(AudioMCPError) as e: materialize_source(UrlSource(type="url", url="http://127.0.0.1/a.wav"), settings)
    assert e.value.code == ErrorCode.URL_FETCH_FAILED
```

- [ ] **Step 2: Run** `uv run pytest tests/unit/test_sources.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement `sources.py`.**
- [ ] **Step 4: Run** the same command. Expected: PASS (the symlink test may skip on Windows).
- [ ] **Step 5: Commit** `feat: audio source resolution with path guard and safe uploads`.

### Task 4: Test fixtures (synthetic generator + committed speech)

**Files:**
- Create: `tests/fixtures/__init__.py`, `tests/fixtures/synth.py`, `tests/fixtures/make_speech.py`, `tests/fixtures/audio/{speech_en.wav, speech_en.txt, dialogue_2spk.wav, dialogue_2spk.rttm, dialogue_2spk.txt, speech_es.ogg, speech_es.txt, SOURCES.md}`
- Modify: `tests/conftest.py`
- Test: `tests/unit/test_synth.py`

**Interfaces:**
- Produces:
  - `make_synthetic(out_dir: Path) -> dict[str, Path]`. Deterministic (`np.random.default_rng(0)`), soundfile 16-bit PCM WAV unless noted:
    - `sine`: 440 Hz amp 0.5, 2.0 s, 44100 Hz, 2 ch
    - `silence`: 3.0 s zeros, 16000 mono
    - `noise`: white uniform amp 0.1, 3.0 s, 16000 mono
    - `clipped`: 440 Hz amp 2.0 clipped to ±1, 2.0 s, 44100 mono
    - `alarm`: 32000 mono. Noise amp 0.001 for 0–1 s; 1000 Hz beeps (0.2 s on / 0.1 s off, amp 0.5) for 1.0–3.0 s; noise amp 0.001 for 3–4 s.
    - `impulses`: 3.0 s, 16000, 0.001 noise floor. Decaying 50 ms noise bursts, peak 0.8, at 0.5/1.5/2.5 s.
    - `spliced`: 16000 mono, 3.0 s.
      - 0–1.0 s: 220 Hz amp 0.3
      - 1.0–1.05 s: exact zeros
      - 1.05–2.0 s: 220 Hz amp 0.3
      - 2.0–3.0 s: white noise amp 0.05
    - `hum`: 50 Hz amp 0.3 + noise 0.01, 3 s, 16000
    - `low_level`: 440 Hz amp 0.005, 2 s, 16000
    - `dc_offset`: 440 Hz amp 0.3 + 0.05, 2 s, 16000
    - `multichannel`: 6 ch, 8000 Hz, 1 s, 440 Hz amp 0.3
    - `hires`: FLAC `PCM_24`, 96000 Hz, 1 s, 1000 Hz amp 0.5
    - `short`: 0.3 s 440 Hz amp 0.3, 16000
    - `corrupt`: `.mp3` = first 2048 bytes of an mp3 conversion of `speech_en`, then 4096 random bytes
    - `not_audio`: `.wav` holding text "hello"
    - `empty`: zero-byte `.wav`
    - `speech_en.{flac,mp3,ogg,m4a,webm,aiff}` via `ffmpeg -y -i` (skipped if ffmpeg is missing)
  - `make_long(out_dir) -> Path`: 30 min, 16000 mono FLAC, 220 Hz amp 0.1.
  - conftest session fixtures: `synth` (built once into a session tmp dir), `FIXTURES_DIR`, `speech_en`, `dialogue`, `speech_es`.
- `make_speech.py` is run manually:
  - It uses pyttsx3 with the SAPI voices "Microsoft David Desktop" and "Microsoft Zira Desktop", then ffmpeg-resamples to 16000 mono s16.
  - `speech_en`: "We need to finish the report by Friday. The alarm test is scheduled for Monday morning."
  - `dialogue_2spk`: four alternating turns David/Zira, separated by 0.4 s silence:
    1. "Did you send the budget numbers?"
    2. "Yes, I sent them this morning."
    3. "Great, then we can close the review."
    4. "I will book the room for Thursday."
  - It writes the `.rttm` from the clip lengths and the `.txt` references.
- `speech_es.ogg`: 6–10 s of public-domain Spanish speech cut from a LibriVox recording with ffmpeg. Record the URL and license in `SOURCES.md`, and type the transcript into `speech_es.txt`. Also note the AudioSet label CSV (CC-BY 4.0) in `SOURCES.md`.

- [ ] **Step 1: Write failing tests**

```python
def test_synth_set(synth):
    info = sf.info(synth["sine"]); assert info.samplerate == 44100 and info.channels == 2 and abs(info.duration - 2.0) < 1e-3
    assert sf.info(synth["hires"]).subtype == "PCM_24" and synth["empty"].stat().st_size == 0

def test_committed_speech_present():
    for n in ["speech_en.wav", "speech_en.txt", "dialogue_2spk.wav", "dialogue_2spk.rttm", "speech_es.ogg", "SOURCES.md"]:
        assert (FIXTURES_DIR / n).exists(), n
    assert sf.info(FIXTURES_DIR / "speech_en.wav").samplerate == 16000
```

- [ ] **Step 2: Run** `uv run pytest tests/unit/test_synth.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement `synth.py`.** Run `uv run python tests/fixtures/make_speech.py` (pyttsx3 is in the dev extra), cut the LibriVox clip, and write `SOURCES.md`.
- [ ] **Step 4: Run** the same command. Expected: PASS.
- [ ] **Step 5: Commit** `test: synthetic and committed speech fixtures`.

### Task 5: Probe and decode via ffmpeg

**Files:**
- Create: `src/audio_mcp/ingest/decode.py`
- Test: `tests/unit/test_decode.py`

**Interfaces:**
- Produces:
  - `find_ffmpeg(settings) -> tuple[Path, Path]`: from config, else `shutil.which`. Missing → `AudioMCPError(MODEL_UNAVAILABLE, "ffmpeg/ffprobe not found; see README install steps")`.
  - `@dataclass ProbeInfo`: `format_name, codec, duration_seconds: float, sample_rate_hz: int, channels: int, bit_depth: int | None, size_bytes: int`.
    - `to_dict()` returns keys `format, codec, duration_seconds, sample_rate_hz, channels, bit_depth, size_bytes`.
  - `probe(path, settings) -> ProbeInfo`:
    - Runs `ffprobe -v error -print_format json -show_format -show_streams` as an arg list, timeout 30 s.
    - Size 0 → `EMPTY_AUDIO`. Non-zero exit → `CORRUPT_AUDIO`. No audio stream → `UNSUPPORTED_AUDIO_FORMAT`.
    - Missing or zero duration → `EMPTY_AUDIO`. Duration > `max_duration_seconds` → `DURATION_TOO_LONG`.
    - `bit_depth` comes from `bits_per_raw_sample` or `bits_per_sample`, with 0 → None.
  - `@dataclass DecodedAudio`: `native: np.ndarray` (float32 `(channels, n)`), `native_sr: int`, `mono16k: np.ndarray` (float32 `(n16,)`). Property `duration`.
  - `decode(path, info, settings) -> DecodedAudio`:
    - Two ffmpeg runs to stdout: `-f f32le -acodec pcm_f32le` at the native rate/channels, and `-ac 1 -ar 16000`. Timeout `max(60, 2*duration)`.
    - Failure → `CORRUPT_AUDIO`. Empty output → `EMPTY_AUDIO`.
  - `resample(x, sr_from, sr_to) -> np.ndarray`: `scipy.signal.resample_poly`.

- [ ] **Step 1: Write failing tests**

```python
@pytest.mark.parametrize("ext", ["flac", "mp3", "ogg", "m4a", "webm", "aiff"])
def test_formats_decode(synth, settings, ext):
    p = synth[f"speech_en.{ext}"]; info = probe(p, settings); d = decode(p, info, settings)
    assert d.mono16k.dtype == np.float32 and abs(len(d.mono16k) / 16000 - info.duration_seconds) < 0.1

def test_odd_layouts(synth, settings):  # Review Focus #4
    i = probe(synth["multichannel"], settings); d = decode(synth["multichannel"], i, settings)
    assert (i.channels, i.sample_rate_hz) == (6, 8000) and d.native.shape[0] == 6 and d.mono16k.ndim == 1
    h = probe(synth["hires"], settings); assert h.sample_rate_hz == 96000 and h.bit_depth == 24

@pytest.mark.parametrize("name,code", [("corrupt", "CORRUPT_AUDIO"), ("not_audio", "CORRUPT_AUDIO"), ("empty", "EMPTY_AUDIO")])
def test_bad_inputs(synth, settings, name, code):
    with pytest.raises(AudioMCPError) as e: decode(synth[name], probe(synth[name], settings), settings)
    assert e.value.code == code

def test_duration_limit(synth, settings):
    settings.max_duration_seconds = 1.0
    with pytest.raises(AudioMCPError) as e: probe(synth["sine"], settings)
    assert e.value.code == "DURATION_TOO_LONG"
```

- [ ] **Step 2: Run** `uv run pytest tests/unit/test_decode.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement `decode.py`.**
- [ ] **Step 4: Run** the same command. Expected: PASS.
- [ ] **Step 5: Commit** `feat: ffprobe/ffmpeg probe and decode with error mapping`.

### Task 6: Audio store and result cache

**Files:**
- Create: `src/audio_mcp/store/__init__.py`, `src/audio_mcp/store/audio_store.py`, `src/audio_mcp/store/cache.py`
- Test: `tests/unit/test_store.py`

**Interfaces:**
- Consumes: `materialize_source`, `probe`, `decode`, the `AudioSource` types.
- Produces:
  - `@dataclass AudioRecord`: `audio_id, path: Path, probe: ProbeInfo, original_name: str, ingested_at: float`.
  - `class AudioStore(settings)`:
    - `ingest(src) -> AudioRecord`:
      - Materializes the source, hashes the file, and copies it to `data_dir/audio/<id>/source<ext>` if absent.
      - Writes `meta.json` and deletes the temp file.
      - `AudioIdSource` → `get`.
    - `ingest_path(path: Path, original_name: str) -> AudioRecord`: internal; skips the root check, for stream finishing.
    - `get(audio_id) -> AudioRecord`: unknown → `AUDIO_NOT_FOUND`.
    - `load(audio_id) -> DecodedAudio`: LRU of 2.
    - `put_variant(audio_id, name, native, sr, transforms: list[dict]) -> Path`: writes `audio/<id>/variants/<name>.flac` + `.json`.
    - `delete(audio_id)`, `list_ids() -> list[str]`.
  - `config_hash(d: dict) -> str`: the first 12 hex of sha256 of `json.dumps(d, sort_keys=True)`.
  - `class ResultCache(settings)`:
    - `get(audio_id, stage, cfg_hash) -> dict | None`
    - `put(audio_id, stage, cfg_hash, value: dict)`: atomic temp + `os.replace` into `data_dir/results/<id>/<stage>-<hash>.json`
    - `clear(audio_id)`

- [ ] **Step 1: Write failing tests**

```python
def test_same_bytes_same_id_across_source_types(settings, speech_en):
    store = AudioStore(settings)
    a = store.ingest(FilePathSource(type="file_path", path=str(speech_en)))
    b = store.ingest(Base64Source(type="base64", data=base64.b64encode(speech_en.read_bytes()).decode(), filename_hint="x.wav"))
    assert a.audio_id == b.audio_id and re.fullmatch(r"aud_[0-9a-f]{16}", a.audio_id)
    assert not any(settings.resolved_temp_dir.iterdir())

def test_original_untouched(settings, speech_en):
    before = hashlib.sha256(speech_en.read_bytes()).hexdigest()
    AudioStore(settings).ingest(FilePathSource(type="file_path", path=str(speech_en)))
    assert hashlib.sha256(speech_en.read_bytes()).hexdigest() == before

def test_unknown_id(settings):
    with pytest.raises(AudioMCPError) as e: AudioStore(settings).get("aud_0000000000000000")
    assert e.value.code == "AUDIO_NOT_FOUND"

def test_cache_roundtrip(settings):
    c = ResultCache(settings); h = config_hash({"m": "small"})
    assert c.get("aud_0000000000000000", "transcription", h) is None
    c.put("aud_0000000000000000", "transcription", h, {"text": "x"})
    assert c.get("aud_0000000000000000", "transcription", h) == {"text": "x"}
```

- [ ] **Step 2: Run** `uv run pytest tests/unit/test_store.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement** the store and cache.
- [ ] **Step 4: Run** the same command. Expected: PASS.
- [ ] **Step 5: Commit** `feat: content-addressed audio store and stage result cache`.

### Task 7: Acoustic feature extraction (DSP)

**Files:**
- Create: `src/audio_mcp/dsp/__init__.py`, `src/audio_mcp/dsp/features.py`
- Test: `tests/unit/test_features.py`

**Interfaces:**
- Produces:
  - `FEATURE_GROUPS = ("level", "silence", "spectral", "pitch", "rhythm", "bands")`.
  - `frame_rms(x, sr, win=0.025, hop=0.010) -> np.ndarray`.
  - `extract_features(native, sr, groups=FEATURE_GROUPS, speech_regions: list[Interval] | None = None) -> dict`, returning `{"units": {...}, "values": {...}}`.
- Values are computed on the channel mean, except `per_channel_peak`. Units vocabulary: `linear, dBFS, dB, Hz, ratio, BPM, per_second, seconds`.
  - **level:**
    - `rms`, `rms_dbfs` (None if rms == 0), `peak_amplitude`, `per_channel_peak`, `crest_factor_db`
    - `clipping_ratio`: fraction of samples with |x| ≥ 0.999
    - `dc_offset`
    - `snr_db_estimate` + `snr_method`: `"vad"` = 10·log10(mean power in speech regions / outside) when regions are given; else `"percentile"` = 10·log10(p95 / p10 frame power). None if the denominator is 0.
  - **silence:** `silence_ratio`, plus `silence_intervals` (≥ 0.3 s) from frames with RMS < −50 dBFS.
  - **spectral:** means over STFT 2048/512 Hann frames of `spectral_centroid_hz`, `spectral_bandwidth_hz`, `spectral_rolloff_hz` (85%), `spectral_flatness`, `zero_crossing_rate`.
  - **bands:** `band_energy`, the fractions for edges `[0,250,500,1000,2000,4000,8000,nyquist]`, keyed `"0-250"`… and summing to 1.
  - **pitch:**
    - YIN in numpy on 16 kHz mono, 50–600 Hz, threshold 0.15.
    - Outputs `f0_mean_hz, f0_median_hz, f0_min_hz, f0_max_hz, f0_std_hz, voiced_ratio`.
    - The f0 keys are None when `voiced_ratio < 0.05`.
  - **rhythm:**
    - `tempo_bpm`: autocorrelation of the onset-strength envelope, 40–240 BPM; None if the peak prominence < 0.1.
    - `onset_rate_per_s`.

- [ ] **Step 1: Write failing tests**

```python
def load(p): x, sr = sf.read(p, always_2d=True); return x.T.astype("float32"), sr

def test_sine_level(synth):
    v = extract_features(*load(synth["sine"]))["values"]
    assert abs(v["rms"] - 0.5 / np.sqrt(2)) < 1e-3 and abs(v["peak_amplitude"] - 0.5) < 1e-3
    assert abs(v["crest_factor_db"] - 3.01) < 0.05 and v["clipping_ratio"] == 0
    assert abs(v["spectral_centroid_hz"] - 440) < 30 and abs(v["f0_median_hz"] - 440) < 5

def test_clipped(synth):
    assert extract_features(*load(synth["clipped"]), ["level"])["values"]["clipping_ratio"] > 0.5

def test_silence(synth):
    v = extract_features(*load(synth["silence"]))["values"]
    assert v["silence_ratio"] == 1.0 and v["rms_dbfs"] is None and v["f0_mean_hz"] is None

def test_bands_sum_and_units(synth):
    f = extract_features(*load(synth["noise"]), ["bands"])
    assert abs(sum(f["values"]["band_energy"].values()) - 1) < 1e-6 and f["units"]["band_energy"] == "ratio"

def test_beeps_onsets(synth):
    v = extract_features(*load(synth["alarm"]), ["rhythm"])["values"]
    assert 4 <= v["onset_rate_per_s"] * 4.0 <= 10
```

- [ ] **Step 2: Run** `uv run pytest tests/unit/test_features.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement `features.py`** with numpy/scipy only.
- [ ] **Step 4: Run** the same command. Expected: PASS.
- [ ] **Step 5: Commit** `feat: acoustic feature extraction with documented units`.

### Task 8: Backend protocols, registry, energy VAD, fakes, Silero VAD

**Files:**
- Create: `src/audio_mcp/backends/__init__.py`, `base.py`, `energy_vad.py`, `silero.py`, `fake.py`, `audioset_labels.csv` (the AudioSet `class_labels_indices.csv`, 527 rows)
- Test: `tests/unit/test_backends_base.py`, `tests/integration/test_silero.py`

**Interfaces:**
- Produces in `base.py`:
  - `Backend` Protocol: `info: ModelInfo`; `available() -> tuple[bool, str | None]` (cheap import check); `load() -> None` (idempotent; raises `MODEL_UNAVAILABLE`).
  - Capability protocols:
    - `VAD.speech_regions(mono16k) -> list[Interval]`
    - `ASR.transcribe(mono16k, *, language=None, word_timestamps=True, task: Literal["transcribe","translate"]="transcribe", per_chunk_language=False, chunks: list[Interval] | None = None, **decode_kwargs) -> ASRResult`
    - `Tagger.tag_windows(mono16k) -> TagResult` and `Tagger.embed(mono16k) -> np.ndarray`
    - `SpeakerEmbedder.embed(windows: list[np.ndarray]) -> np.ndarray` (n, d)
    - `Diarizer.diarize(mono16k, num_speakers: int | None) -> tuple[list[SpeakerTurn], list[Interval]]` (turns, overlaps)
    - `TextEmbedder.embed(texts: list[str]) -> np.ndarray`, L2-normalized
    - `LLM.complete(system: str, prompt: str) -> str`, with attribute `is_remote: bool`
  - Result types:
    - `@dataclass ASRResult`: `segments: list[Segment], language: str | None, language_probability: float | None, languages: list[dict], dropped_low_speech: int = 0`
    - `@dataclass TagResult`: `labels: list[str], scores: np.ndarray (windows×527), window_starts: np.ndarray, window_seconds: float, embeddings: np.ndarray`
  - `@dataclass UnavailableBackend(name: str, reason: str)`: `available() -> (False, reason)`; every other method raises `MODEL_UNAVAILABLE`.
  - `@dataclass Backends`: `vad, vad_fallback, asr, tagger, speaker, diarizer, text, llm`.
  - `build_backends(settings) -> Backends`:
    - `fake` → `fake_backends()`.
    - `real` → construct adapters without loading them. A failed import gives `UnavailableBackend` with reason `"<module> not installed (uv sync --extra models)"`.
    - `diarizer` is `PyannoteDiarizer` only when `diarization_backend == "pyannote"`, else None.
    - `llm` is `NoLLM` until Task 28.
- `EnergyVAD`:
  - 30 ms frames. Speech = frame RMS > max(−40 dBFS, p10 floor + 12 dB). Gaps < 0.3 s are merged and regions < 0.25 s dropped.
  - `ModelInfo(stage="vad", name="energy-vad", version="1")`.
- `SileroVAD`: `silero_vad.load_silero_vad(onnx=True)` + `get_speech_timestamps(torch.from_numpy(x), model, sampling_rate=16000, return_seconds=True)`.
- `fake_backends()`, all with `ModelInfo(name="fake-…")`:
  - `vad`: `EnergyVAD`.
  - `FakeASR`:
    - One segment per VAD region (or per `chunks`). Text cycles through `["we need to finish the report by friday", "the alarm test is scheduled for monday morning"]`.
    - Language `"en"` (with `per_chunk_language`: chunk i is `["en","es"][i % 2]`), probability 0.9.
    - Words spread evenly, word confidence 0.8, segment confidence 0.85, `confidence_kind="model_probability"`.
    - `task="translate"` returns the same text prefixed `"[en] "`.
  - `FakeTagger`:
    - Real labels from the CSV. Windows of 1.0 s / 0.5 s hop (a short file gets 1 window).
    - Scores per window:
      - "Speech" 0.9 when the window overlaps EnergyVAD speech and the 1 kHz band is not dominant
      - "Beep, bleep" 0.8 when the 900–1100 Hz energy is > 50% of the window energy
      - "Silence" 0.7 when the window RMS < −50 dBFS
      - otherwise "Noise" 0.5
    - Embeddings: 64 log-band energies zero-padded to 2048.
  - `FakeSpeaker`: a normalized 64-bin mean log-spectrum.
  - `FakeText`: hashing bag-of-words, 256-d, L2-normalized.
  - `NoLLM` (stub class here: `available() -> (False, "llm_provider is none")`, `is_remote=False`).

- [ ] **Step 1: Write failing tests**

```python
def test_energy_vad_finds_beeps(synth):
    x, _ = sf.read(synth["alarm"], dtype="float32"); r = EnergyVAD().speech_regions(resample(x, 32000, 16000))
    assert r and r[0].start == pytest.approx(1.0, abs=0.1) and r[-1].end == pytest.approx(3.0, abs=0.15)

def test_energy_vad_silence(synth):
    x, _ = sf.read(synth["silence"], dtype="float32"); assert EnergyVAD().speech_regions(x) == []

def test_real_profile_degrades_without_models(settings, monkeypatch):
    settings.backend_profile = "real"; monkeypatch.setitem(sys.modules, "faster_whisper", None)
    ok, reason = build_backends(settings).asr.available()
    assert ok is False and "faster_whisper" in reason

def test_fake_tagger_shape(speech_en):
    x, _ = sf.read(speech_en, dtype="float32"); t = fake_backends().tagger.tag_windows(x)
    assert t.scores.shape[1] == 527 == len(t.labels)
```

  `tests/integration/test_silero.py` (`@pytest.mark.slow`): the regions on `speech_en` cover ≥ 60% of its duration.
- [ ] **Step 2: Run** `uv run pytest tests/unit/test_backends_base.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement** it.
- [ ] **Step 4: Run** the unit tests (PASS) and, with models installed, `uv run pytest tests/integration/test_silero.py -v` (PASS).
- [ ] **Step 5: Commit** `feat: backend protocols, registry, energy VAD, fakes, silero VAD`.

### Task 9: faster-whisper ASR adapter

**Files:**
- Create: `src/audio_mcp/backends/whisper.py`
- Modify: `backends/base.py` (real profile → `FasterWhisperASR`)
- Test: `tests/integration/test_whisper.py`

**Interfaces:**
- Produces: `FasterWhisperASR(settings)`.
  - Lazy `WhisperModel(settings.asr_model, device=dev, compute_type=settings.asr_compute_type, download_root=str(models_dir/"whisper"))`, where `dev = "cuda"` if `device=="cuda"` or (`"auto"` and `ctranslate2.get_cuda_device_count() > 0`), else `"cpu"`.
  - `transcribe` → `model.transcribe(x, language=language, task=task, word_timestamps=word_timestamps, vad_filter=False, beam_size=decode_kwargs.get("beam_size", 5), temperature=decode_kwargs.get("temperature", 0.0))`.
  - Mapping:
    - Segment confidence `= min(1, exp(avg_logprob))`; word confidence `= word.probability`; both `confidence_kind="model_probability"`.
    - Segments with `no_speech_prob > 0.8 and avg_logprob < -1.0` are dropped and counted.
    - `languages = [{"language", "start": 0, "end": duration, "probability"}]`.
  - `per_chunk_language=True` raises `NotImplementedError` until Task 22.
  - `info = ModelInfo(stage="transcription", name="faster-whisper", version=settings.asr_model)`.

- [ ] **Step 1: Write the failing test** (`@pytest.mark.slow`)

```python
def test_transcribes_reference(settings):
    settings.backend_profile = "real"
    x, _ = sf.read(FIXTURES_DIR / "speech_en.wav", dtype="float32")
    r = FasterWhisperASR(settings).transcribe(x)
    text = " ".join(s.text for s in r.segments).lower()
    assert "report" in text and "friday" in text and r.language == "en"
    assert all(0 <= w.confidence <= 1 for s in r.segments for w in s.words)
    assert r.segments[0].start < 1.0 and r.segments[-1].end <= len(x) / 16000 + 0.5
```

- [ ] **Step 2: Run** `uv run pytest tests/integration/test_whisper.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement** the adapter.
- [ ] **Step 4: Run** the same command. Expected: PASS (the first run downloads ~460 MB).
- [ ] **Step 5: Commit** `feat: faster-whisper ASR adapter`.

### Task 10: PANNs tagger and event extraction

**Files:**
- Create: `src/audio_mcp/backends/panns.py`, `src/audio_mcp/analysis/__init__.py`, `src/audio_mcp/analysis/events.py`
- Modify: `backends/base.py` (real profile → `PannsTagger`)
- Test: `tests/unit/test_events.py`, `tests/integration/test_panns.py`

**Interfaces:**
- Produces:
  - `PannsTagger(settings)`. `load()`:
    1. Copy the bundled CSV to `~/panns_data/class_labels_indices.csv` if absent. This must happen before importing `panns_inference`: its `config.py` shells out to `wget`, which is missing on Windows.
    2. Download `Cnn14_mAP=0.431.pth` from `https://zenodo.org/record/3987831/files/Cnn14_mAP%3D0.431.pth?download=1` via httpx to `models_dir/panns/` (`.part` + rename).
    3. Create `AudioTagging(checkpoint_path=..., device=...)`.
  - `tag_windows`: resample to 32 kHz; windows of 1.0 s / 0.5 s hop (a short file is padded to one window); batch 16; `clipwise_output` → `scores`, `embedding` → `embeddings`. `embed(x)` is the mean of the window embeddings.
  - `info = ModelInfo(stage="events", name="panns-cnn14", version="mAP0.431")`.
  - `frames_to_events(tag: TagResult, *, duration: float, threshold: float, labels: list[str] | None = None, exclude: frozenset[str] = frozenset()) -> list[Event]`:
    - Per label, runs of windows with score ≥ threshold, where consecutive windows overlap or touch, merge to `[first_start, min(last_start + window_seconds, duration)]`.
    - `confidence` = the max score in the run, `confidence_kind="model_probability"`, `source="panns"` (or the tagger's name).
    - Ids `evt_0001…` in (start, label) order. The `labels` filter is case-insensitive.
  - `EVENT_CATEGORIES: dict[str, list[str]]`: keys `alarm, impact, footsteps, door, animal, machinery, music, environment, speech, vehicle`, each mapped to AudioSet label names. `alarm` must include "Alarm", "Beep, bleep", "Siren", "Smoke detector, smoke alarm", "Buzzer".

- [ ] **Step 1: Write failing tests**

```python
TAG = TagResult(labels=["Speech", "Beep, bleep"], scores=np.array([[0.9, 0], [0.9, 0], [0, 0.8], [0, 0.1]]),
                window_starts=np.array([0, .5, 1, 1.5]), window_seconds=1.0, embeddings=np.zeros((4, 2)))

def test_merge_runs():
    ev = frames_to_events(TAG, duration=2.5, threshold=0.3)
    assert [(e.label, e.start, e.end, e.confidence) for e in ev] == [("Speech", 0.0, 1.5, 0.9), ("Beep, bleep", 1.0, 2.0, 0.8)]
    assert ev[0].id == "evt_0001" and ev[0].confidence_kind == "model_probability"

def test_label_filter_case_insensitive():
    assert [e.label for e in frames_to_events(TAG, duration=2.5, threshold=0.3, labels=["speech"])] == ["Speech"]
```

  `tests/integration/test_panns.py` (`@pytest.mark.slow`):
  - `alarm` at threshold 0.2 has an event whose label is in `EVENT_CATEGORIES["alarm"]`, with start within 0.6 s of 1.0.
  - `silence` has no event other than "Silence".
- [ ] **Step 2: Run** `uv run pytest tests/unit/test_events.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement** it.
- [ ] **Step 4: Run** the unit tests and the slow test. Expected: PASS.
- [ ] **Step 5: Commit** `feat: PANNs tagger and event interval extraction`.

### Task 11: Stage registry and runner (explicit tasks)

**Files:**
- Create: `src/audio_mcp/pipeline/__init__.py`, `stages.py`, `planner.py`, `runner.py`, `src/audio_mcp/context.py`
- Test: `tests/unit/test_runner.py`

**Interfaces:**
- Produces:
  - `@dataclass AppContext`: `settings, store, cache, backends, jobs=None, library=None, streams=None`, and `build_app_context(settings) -> AppContext`. Later tasks fill the optional fields there.
  - `@dataclass StageContext`: `app, record: AudioRecord, audio: DecodedAudio, options: dict, results: dict[str, dict], models_used: list[ModelInfo], warnings: list[dict], cancel_event: threading.Event | None`.
  - `@dataclass Stage`: `name, deps: tuple[str, ...], fn: Callable[[StageContext], dict], cfg: Callable[[AppContext, dict], dict], backend: str | None`. `backend` is the `Backends` field name that must be available.
  - `STAGES: dict[str, Stage]`. Phase-1 stages:
    - `metadata` → `probe.to_dict()` plus `"duration_seconds"`.
    - `vad` → `{"speech_regions", "speech_ratio", "backend"}`. Uses `vad`; if unavailable, `vad_fallback` with the warning `VAD_FALLBACK`.
    - `features` (deps `vad`) → `extract_features(native, sr, options.get("feature_groups") or FEATURE_GROUPS, speech_regions)`.
    - `transcription` (deps `vad`, backend `asr`) → `{"text", "segments", "language", "language_probability", "languages"}`. No VAD speech → empty segments + the warning `NO_SPEECH_DETECTED`, without calling ASR.
    - `language` (deps `transcription`) → `{"primary", "confidence", "confidence_kind", "segments": languages}`.
    - `events` (backend `tagger`) → `{"events", "threshold"}`, using `options.event_threshold` (default `settings.event_threshold`) and `options.event_labels`.
  - `planner.TASK_TO_STAGES`:
    - Phase 1 maps `acoustic_features→["metadata","features"]`, `transcription→["transcription"]`, `language_detection→["language"]`, `sound_event_detection→["events"]`.
    - Also map the not-yet-registered tasks: `audio_quality→["quality"]`, `speaker_diarization→["diarization"]`, `speech_characteristics→["speech"]`, `integrity→["integrity"]`, `timeline→["timeline"]`, `summarization→["summary"]`, `uncertainty→["uncertainty"]`.
    - A stage missing from `STAGES` goes to `stages_skipped` with reason `"not implemented in this version"`.
  - `@dataclass AnalysisOutcome`: `record, results: dict[str, dict], processing: ProcessingInfo, warnings: list[dict], errors: list[dict]`.
  - `run_analysis(app, source: AudioSource, stages: list[str], options: dict, *, cancel_event=None, progress=None) -> AnalysisOutcome`:
    - Ingest, load, expand deps, topo-sort.
    - Per stage, in order:
      1. A cache hit (key `config_hash(stage.cfg(app, options))`) → `cache_hits`.
      2. A failed dependency → skipped with `"dependency <name> failed"`.
      3. An unavailable backend → skipped with its reason.
      4. Otherwise run it, appending `models_used`.
      - An exception → error `{"code": e.code if AudioMCPError else "STAGE_FAILED", "stage", "message", "retryable"}`.
    - Check `cancel_event` between stages and raise `AudioMCPError(CANCELLED)`. Call `progress(done, total)` after each stage.
    - Results are JSON-serializable dicts (pydantic dumps with `exclude_none`).

- [ ] **Step 1: Write failing tests** (fixture `app(settings) = build_app_context(settings)`, fake profile)

```python
def test_runs_with_deps_and_caches(app, speech_en):
    o1 = run_analysis(app, FilePathSource(type="file_path", path=str(speech_en)), ["transcription"], {})
    assert o1.processing.stages_run == ["vad", "transcription"] and o1.results["transcription"]["segments"]
    o2 = run_analysis(app, FilePathSource(type="file_path", path=str(speech_en)), ["transcription"], {})
    assert o2.processing.cache_hits == ["vad", "transcription"] and o2.processing.stages_run == []

def test_failed_stage_reports_and_skips_dependents(app, speech_en, monkeypatch):
    monkeypatch.setattr(app.backends.asr, "transcribe", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    o = run_analysis(app, FilePathSource(type="file_path", path=str(speech_en)), ["transcription", "language", "features"], {})
    assert o.errors[0]["stage"] == "transcription" and "features" in o.results
    assert {"stage": "language", "reason": "dependency transcription failed"} in o.processing.stages_skipped

def test_unavailable_backend_skipped(app, speech_en):
    app.backends.asr = UnavailableBackend("faster-whisper", "faster_whisper not installed")
    o = run_analysis(app, FilePathSource(type="file_path", path=str(speech_en)), ["transcription"], {})
    assert o.processing.stages_skipped[0]["reason"].startswith("faster_whisper not installed")

def test_unregistered_stage_reported(app, speech_en):
    o = run_analysis(app, FilePathSource(type="file_path", path=str(speech_en)), ["summary"], {})
    assert {"stage": "summary", "reason": "not implemented in this version"} in o.processing.stages_skipped

@pytest.mark.parametrize("name", ["short", "silence"])
def test_tiny_and_silent_inputs(app, synth, name):  # Review Focus #2
    o = run_analysis(app, FilePathSource(type="file_path", path=str(synth[name])), ["transcription", "events", "features"], {})
    assert o.errors == [] and o.results["transcription"]["segments"] == []
```

  `synth` paths live in a session tmp dir. In conftest, the `settings` fixture adds the `synth` dir to `allowed_roots` (`synth` is session-scoped and built first).
- [ ] **Step 2: Run** `uv run pytest tests/unit/test_runner.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement** it.
- [ ] **Step 4: Run** the same command. Expected: PASS.
- [ ] **Step 5: Commit** `feat: stage DAG runner with caching and partial-failure reporting`.

### Task 12: MCP server with Phase-1 tools

**Files:**
- Create: `src/audio_mcp/tools/__init__.py`, `schemas.py`, `handlers.py`, `src/audio_mcp/server.py`, `src/audio_mcp/logging_setup.py`
- Test: `tests/unit/test_handlers_phase1.py`, `tests/e2e/test_stdio.py`

**Interfaces:**
- Produces:
  - `schemas.py`: pydantic models with `extra="forbid"`; `source: AudioSource` is required where present.
    - `AnalyzeOptions`: `timestamps=True, word_level_timestamps=True, identify_speakers=True, detect_emotions=False, include_confidence=True, include_evidence=True, language: str | None = None, num_speakers: int | None = Field(None, ge=1, le=20)`.
    - `AnalyzeRequest`: `source, tasks: list[TaskName] | Literal["auto"] = "auto", goal: str | None = None, budget: Literal["fast","balanced","thorough"] = "balanced", options: AnalyzeOptions = AnalyzeOptions(), output_format: Literal["json"] = "json", async_mode: Literal["auto","always","never"] = "auto"`. `TaskName` is a Literal of the `TASK_TO_STAGES` keys.
    - `TranscribeRequest`: `source, language=None, word_timestamps=True, diarize=False, translate_to_en=False, multilingual=False, async_mode="auto"`.
    - `DetectEventsRequest`: `source, categories: list[CategoryName] | None = None, labels: list[str] | None = None, min_confidence: float = Field(0.3, ge=0, le=1), async_mode="auto"`.
    - `FeaturesRequest`: `source, feature_groups: list[FeatureGroup] | None = None, async_mode="auto"`.
  - `handlers.py`: `analyze_audio(app, req) -> dict`, `transcribe_audio`, `detect_audio_events`, `extract_audio_features`.
    - A shared decorator `@tool_handler` maps `AudioMCPError` → `error_response`. A `ValidationError` → `INVALID_INPUT` with the first error message. Anything else → `STAGE_FAILED`, logged without data.
    - `async_mode` is accepted and ignored until Task 24.
    - `analyze_audio` returns payload key `analysis` with:
      - `audio_metadata` (from `metadata`)
      - `language` (`{"primary","confidence","confidence_kind"}`)
      - `transcription` (`{"text","segments"}`)
      - `sound_events` (the event list)
      - `acoustic_features`
      - later tasks add `speakers`, `audio_quality`, `summary`, `timeline`, `integrity`, `speech_characteristics`, `uncertainty`, `recovery`
    - `detect_emotions=true` adds the warning `UNSUPPORTED_FEATURE` with the message "emotion detection is not implemented".
    - `include_confidence=false` strips `confidence`/`confidence_kind` recursively. `word_level_timestamps=false` drops `words`.
    - Phase 1 `"auto"` expands to `acoustic_features, transcription, language_detection, sound_event_detection`; Task 17 replaces it.
    - Other handlers return `result`:
      - `transcribe_audio`: `{"text","segments","language","languages"}`
      - `detect_audio_events`: `{"events","threshold"}`, with `categories` expanded via `EVENT_CATEGORIES` into `event_labels`
      - `extract_audio_features`: `{"metadata","features"}`
  - `server.py`:
    - `mcp = FastMCP("audio-intelligence", lifespan=lifespan)`. The lifespan sets the module-level `_app = build_app_context(load_settings())` and calls `setup_logging`.
    - Each tool is registered with `@mcp.tool(name=..., description=...)`, typed params mirroring the request fields, and a body `return await anyio.to_thread.run_sync(lambda: handlers.x(_app, Request(**locals_dict)))`.
    - Descriptions are one or two sentences: what it does and what it returns.
    - `main()` → `mcp.run()`.
  - `logging_setup.setup_logging(settings)`: a JSON formatter on stderr (and `log_file`), plus a filter replacing the record attrs `text`, `transcript`, `question`, `answer` with `"[redacted]"` unless `log_sensitive`.

- [ ] **Step 1: Write failing tests**

```python
def test_analyze_shape(app, speech_en):
    r = analyze_audio(app, AnalyzeRequest(source=fp(speech_en), tasks=["transcription", "language_detection", "sound_event_detection", "acoustic_features"]))
    a = r["analysis"]; assert r["status"] == "success" and r["audio_id"].startswith("aud_")
    assert {"audio_metadata", "language", "transcription", "sound_events", "acoustic_features"} <= a.keys()
    assert a["audio_metadata"]["sample_rate_hz"] == 16000 and r["processing"]["models_used"]

def test_bad_path_error_shape(app):
    r = transcribe_audio(app, TranscribeRequest(source={"type": "file_path", "path": str(Path.home().parent)}))
    assert r["status"] == "error" and r["error"]["code"] in {"PATH_NOT_ALLOWED", "INVALID_INPUT"} and r["error"]["retryable"] is False

def test_unsupported_format(app, synth):
    r = extract_audio_features(app, FeaturesRequest(source=fp(synth["not_audio"])))
    assert r["error"]["code"] in {"CORRUPT_AUDIO", "UNSUPPORTED_AUDIO_FORMAT"}

def test_events_categories(app, synth):
    r = detect_audio_events(app, DetectEventsRequest(source=fp(synth["alarm"]), categories=["alarm"]))
    assert r["status"] == "success" and r["result"]["events"] and all(e["label"] in EVENT_CATEGORIES["alarm"] for e in r["result"]["events"])

def test_emotions_unsupported_warning(app, speech_en):
    r = analyze_audio(app, AnalyzeRequest(source=fp(speech_en), tasks=["acoustic_features"], options={"detect_emotions": True}))
    assert any(w["code"] == "UNSUPPORTED_FEATURE" for w in r["warnings"])

def test_cache_reuse_across_source_types(app, speech_en):  # Review Focus #5
    transcribe_audio(app, TranscribeRequest(source=fp(speech_en)))
    r = transcribe_audio(app, TranscribeRequest(source={"type": "base64", "data": base64.b64encode(speech_en.read_bytes()).decode()}))
    assert "transcription" in r["processing"]["cache_hits"]
```

  `tests/e2e/test_stdio.py`:
  - Starts the server through `mcp.client.stdio.stdio_client(StdioServerParameters(command=sys.executable, args=["-m", "audio_mcp"], env={**os.environ, "AUDIO_MCP_BACKEND_PROFILE": "fake", "AUDIO_MCP_DATA_DIR": str(tmp), "AUDIO_MCP_ALLOWED_ROOTS": json.dumps([str(FIXTURES_DIR)])}))`.
  - `list_tools()` names must equal `EXPECTED_TOOLS` (a module constant, here the four phase-1 names), and every `inputSchema["type"] == "object"`.
  - `call_tool("transcribe_audio", {"source": fp(FIXTURES_DIR/"speech_en.wav")})` content[0] JSON has `status == "success"`.
- [ ] **Step 2: Run** `uv run pytest tests/unit/test_handlers_phase1.py tests/e2e/test_stdio.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement** it.
- [ ] **Step 4: Run** the same command. Expected: PASS.
- [ ] **Step 5: Manual check.** `uv run audio-mcp` waits on stdin with nothing on stdout. `claude mcp add audio-intel -- uv --directory F:\audio_mcp run audio-mcp`, then `/mcp` lists 4 tools.
- [ ] **Step 6: Commit** `feat: MCP server with analyze/transcribe/events/features tools`.

**Phase 1 gate:** `uv run pytest -m "not slow and not long"` is green, and so is `uv run pytest -m slow` with `--extra models`.

---

# Phase 2 — Audio intelligence

### Task 13: Quality diagnosis, preprocessing, inspect_audio_quality

**Files:**
- Create: `src/audio_mcp/dsp/quality.py`, `src/audio_mcp/dsp/preprocess.py`
- Modify: `pipeline/stages.py` (stage `quality`, deps `features`, `vad`), `tools/schemas.py`, `tools/handlers.py`, `server.py`, `tests/e2e/test_stdio.py` (`EXPECTED_TOOLS`)
- Test: `tests/unit/test_quality.py`, `tests/unit/test_preprocess.py`, `tests/unit/test_handlers_quality.py`

**Interfaces:**
- Produces:
  - `diagnose_quality(native, sr, feature_values: dict, speech_regions: list[Interval]) -> dict` with `{"metrics", "issues", "recommendations", "overall"}`.
  - Issue = `{code, severity, message, measured, threshold, start?, end?}`. Rules:

    | Code | Rule | Recommendation |
    |---|---|---|
    | `CLIPPING` | `clipping_ratio > 0.001` | none; message says "not recoverable by preprocessing" |
    | `LOW_LEVEL` | `rms_dbfs < -35` | `normalize` |
    | `SILENCE` | `silence_ratio > 0.5` | `trim_silence` |
    | `HIGH_NOISE` | `snr_db_estimate < 15` | `denoise` |
    | `HUM` | the Welch PSD (nperseg = sr) peak at 50 or 60 Hz ≥ 20 dB above the median PSD of 30–120 Hz | `highpass_80hz` |
    | `ECHO` | the onset-envelope normalized autocorrelation has a peak > 0.5 at lags 30–300 ms | none; "heuristic" in message |
    | `BANDWIDTH_LIMITED` | `sr >= 16000` and 99% rolloff < 3800 Hz | none |
    | `DC_OFFSET` | `abs(dc_offset) > 0.01` | `remove_dc` |
    | `MULTICHANNEL` | channels > 1 | `mono` |

  - Severity: `high` if the measured value exceeds the threshold by ≥ 2× (for dB rules, by ≥ 10 dB); `MULTICHANNEL` is always `low`; else `medium`.
  - `overall`: `poor` if any high, `fair` if any medium, else `good`. `recommendations` is the deduplicated op list.
  - `PREPROCESS_OPS = ("remove_dc","mono","highpass_80hz","denoise","trim_silence","resample_16k","normalize")`. This tuple is also the canonical execution order.
  - `apply_preprocessing(native, sr, ops) -> tuple[np.ndarray, int, list[dict]]`, with transforms as `{"op","params"}`. Requested ops run in canonical order, whatever order they're given in.
    - `normalize`: RMS → −20 dBFS, peak-limited to −1 dBFS.
    - `denoise`: STFT spectral gating with a noise profile = per-bin 10th-percentile magnitude, gate at +6 dB, 3-frame smoothing.
    - `trim_silence` records `{"offset_seconds"}`.
    - `highpass_80hz`: 4th-order Butterworth, `sosfiltfilt`.
  - `InspectQualityRequest`: `source, apply_preprocessing: bool | list[PreprocessOp] = False, compare_transcription: bool = False`.
  - `inspect_audio_quality` handler:
    - Returns the quality report as `result`.
    - With preprocessing (`True` = the recommended ops), it stores the variant (`put_variant(audio_id, "preprocessed", …)`) and adds `result.preprocessing = {"variant_path", "transforms", "before": metrics, "after": metrics}`. `metrics` = `level`+`silence` feature values.
    - With `compare_transcription` and ASR available, it adds `transcription_comparison = {"before": {"mean_word_confidence", "low_confidence_words"}, "after": {...}, "note": "measured comparison only; not a claim of improved accuracy"}`, with variant times shifted by `offset_seconds`.
    - Shared helper `compare_preprocessed_asr(app, record, audio, ops) -> dict`, reused by Task 17.

- [ ] **Step 1: Write failing tests**

```python
@pytest.mark.parametrize("name,code", [("clipped", "CLIPPING"), ("low_level", "LOW_LEVEL"), ("silence", "SILENCE"),
    ("hum", "HUM"), ("dc_offset", "DC_OFFSET"), ("noise", "HIGH_NOISE")])
def test_issue_detected(synth, name, code):
    x, sr = load(synth[name]); q = diagnose_quality(x, sr, extract_features(x, sr)["values"], [])
    assert code in {i["code"] for i in q["issues"]}

def test_clean_sine(synth):
    x, sr = load(synth["sine"]); q = diagnose_quality(x, sr, extract_features(x, sr)["values"], [])
    assert not {"CLIPPING", "LOW_LEVEL", "DC_OFFSET"} & {i["code"] for i in q["issues"]}

def test_normalize_and_dc(synth):
    x, sr = load(synth["dc_offset"]); y, _, t = apply_preprocessing(x, sr, ["normalize", "remove_dc"])
    assert abs(y.mean()) < 1e-3 and abs(20 * np.log10(np.sqrt((y ** 2).mean())) + 20) < 0.5
    assert [s["op"] for s in t] == ["remove_dc", "normalize"]

def test_handler_preserves_original(app, synth):
    p = synth["low_level"]; before = p.read_bytes()
    r = inspect_audio_quality(app, InspectQualityRequest(source=fp(p), apply_preprocessing=True))["result"]["preprocessing"]
    assert p.read_bytes() == before and Path(r["variant_path"]).exists() and r["after"]["rms_dbfs"] > r["before"]["rms_dbfs"]
```

  `load` moves into conftest.
- [ ] **Step 2: Run** `uv run pytest tests/unit/test_quality.py tests/unit/test_preprocess.py tests/unit/test_handlers_quality.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement** it. Register the tool and add the `audio_quality` mapping in `analyze_audio`.
- [ ] **Step 4: Run** the same command. Expected: PASS.
- [ ] **Step 5: Commit** `feat: quality diagnosis, preprocessing variants, inspect_audio_quality`.

### Task 14: Speaker diarization (+ DER metric)

**Files:**
- Create: `src/audio_mcp/backends/ecapa.py`, `src/audio_mcp/backends/pyannote_diar.py`, `src/audio_mcp/analysis/diarize.py`, `src/audio_mcp/eval/__init__.py`, `src/audio_mcp/eval/metrics.py`
- Modify: `backends/base.py`, `pipeline/stages.py`, `tools/handlers.py`
- Test: `tests/unit/test_diarize.py`, `tests/unit/test_metrics.py`, `tests/integration/test_diarization_real.py`

**Interfaces:**
- Produces:
  - `EcapaEmbedder(settings)`: `EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir=str(models_dir/"ecapa"), run_opts={"device": dev}, local_strategy=LocalStrategy.COPY)`. COPY avoids the Windows symlink privilege error. `embed(windows) -> (n,192)`, L2-normalized.
  - `PyannoteDiarizer(settings)`: available only if `pyannote.audio` imports and `HF_TOKEN` is set. Uses `Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=token)`; returns turns + `annotation.get_overlap()` intervals.
  - `cluster_speakers(mono16k, regions, embedder, *, threshold, num_speakers, max_speakers) -> list[SpeakerTurn]`:
    - Windows of 1.5 s / 0.75 s hop within regions (a region < 1.5 s gets one window).
    - `linkage(emb, "average", metric="cosine")`, then `fcluster(t=num_speakers, criterion="maxclust")` if given, else `fcluster(t=threshold, criterion="distance")`; if the cluster count > `max_speakers`, redo with `maxclust=max_speakers`. Fewer than 2 windows gives one speaker.
    - Consecutive same-label windows merge into turns; the boundary between differing overlapping windows is their overlap midpoint.
    - Labels are relabelled `speaker_1..N` by first appearance.
  - `assign_speakers(segments, turns) -> list[Segment]`: max overlap per segment and per word; no overlap → None.
  - Stages:
    - `diarization` (deps `vad`, backend `speaker`, or `diarizer` when configured) → `{"turns", "speakers": [{"speaker_id","first_seen","last_seen","total_speech_seconds"}], "overlaps": list | None, "overlap_supported": bool, "backend"}`. The ECAPA path adds the warning `OVERLAP_UNSUPPORTED`.
    - `speaker_assign` (deps `transcription`, `diarization`) → `{"segments": assigned}`.
    - The handlers present `speaker_assign.segments` as the transcript when present.
  - `TASK_TO_STAGES["speaker_diarization"] = ["diarization", "speaker_assign"]`. `transcribe_audio(diarize=True)` adds both. `analyze_audio` maps `analysis.speakers = diarization` (speakers + turns + overlap fields).
  - `eval/metrics.py`:
    - `load_rttm(path) -> list[dict]`
    - `der(ref, hyp, *, collar=0.25, frame=0.01) -> {"der","missed","false_alarm","confusion"}`: frame-based; excludes ±collar around ref boundaries; optimal mapping via `linear_sum_assignment` on the overlap matrix; denominator = ref speech frames.

- [ ] **Step 1: Write failing tests**

```python
def test_two_clusters_synthetic():
    class E:
        def embed(self, ws): return np.array([[1, 0] if w.mean() > 0 else [0, 1] for w in ws], dtype="float32")
    x = np.concatenate([np.full(48000, .1), np.full(48000, -.1), np.full(48000, .1)]).astype("float32")
    turns = cluster_speakers(x, [Interval(start=0, end=9)], E(), threshold=0.5, num_speakers=None, max_speakers=8)
    assert [t.speaker_id for t in turns] == ["speaker_1", "speaker_2", "speaker_1"] and turns[1].start == pytest.approx(3.0, abs=0.8)

def test_assign_by_overlap():
    segs = [Segment(id="seg_0001", start=0, end=2, text="a"), Segment(id="seg_0002", start=2.5, end=4, text="b")]
    out = assign_speakers(segs, [SpeakerTurn(speaker_id="speaker_1", start=0, end=2.2), SpeakerTurn(speaker_id="speaker_2", start=2.2, end=5)])
    assert [s.speaker_id for s in out] == ["speaker_1", "speaker_2"]

def test_der_perfect_and_swapped_labels():  # tests/unit/test_metrics.py
    ref = [{"speaker_id": "A", "start": 0, "end": 5}, {"speaker_id": "B", "start": 5, "end": 10}]
    hyp = [{"speaker_id": "s2", "start": 0, "end": 5}, {"speaker_id": "s1", "start": 5, "end": 10}]
    assert der(ref, hyp)["der"] == pytest.approx(0, abs=1e-6)

def test_der_missed():
    assert der([{"speaker_id": "A", "start": 0, "end": 10}], [], collar=0)["der"] == pytest.approx(1.0)
```

  `tests/integration/test_diarization_real.py` (`@pytest.mark.slow`): ECAPA on `dialogue_2spk.wav` gives 2 speakers and DER < 0.35 against `dialogue_2spk.rttm`.
- [ ] **Step 2: Run** `uv run pytest tests/unit/test_diarize.py tests/unit/test_metrics.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement** it.
- [ ] **Step 4: Run** the unit tests and the slow test. Expected: PASS.
- [ ] **Step 5: Commit** `feat: ECAPA clustering diarization, optional pyannote, DER metric`.

### Task 15: Timeline / event graph + get_audio_timeline

**Files:**
- Create: `src/audio_mcp/timeline/__init__.py`, `src/audio_mcp/timeline/graph.py`
- Modify: `pipeline/stages.py` (stage `timeline`, deps `metadata`, `features`, `transcription`, `events`; it uses `diarization`/`speaker_assign` from `ctx.results` when present), `tools/*`, `server.py`, e2e `EXPECTED_TOOLS`
- Test: `tests/unit/test_timeline.py`

**Interfaces:**
- Produces:
  - `build_timeline(results: dict, duration: float, *, include: set[str] | None = None) -> dict` with `{"nodes", "edges", "text"}`.
  - Node `{id, type, label, start, end, speaker_id?, confidence?, confidence_kind?, source}`. Types and ids:
    - `speech_segment`: segment id. Uses `speaker_assign.segments` if present.
    - `speaker_turn`: `turn_0001`.
    - `sound_event`: event id. "Speech"-labelled events are excluded when transcription exists.
    - `silence`: `sil_0001`, from `features.values.silence_intervals`.
    - `overlap`: `ovl_0001`.
    - `transition`: `trn_0001`, zero-length, one of:
      - a speaker change at turn boundaries
      - the onset of a non-speech event with confidence ≥ 0.5
      - a level jump (adjacent 1 s windows differing in RMS by > 12 dB)
  - Edges `{source, target, relation, gap_seconds?}`. Built over nodes sorted by (start, end) with a sweep, so not all pairs:
    - `precedes`: each node to its nearest following node with `start >= node.end`, with `gap_seconds`.
    - `overlaps`: partial overlap.
    - `during`: A ⊂ B.
    - `interrupts`: a `sound_event` ⊂ a `speech_segment` (instead of `during`).
  - `render_text(tl) -> str`: one line per node, `MM:SS.s–MM:SS.s  <type>  <label>[ (speaker_x)][ (0.81)]`.
  - `TimelineRequest`: `source, include: list[NodeType] | None = None, diarize: bool = True`.
  - `get_audio_timeline` runs `["timeline"]`, plus `diarization`, `speaker_assign` when `diarize` and the `speaker` backend is available.

- [ ] **Step 1: Write failing tests**

```python
def test_sequence_and_relations():
    results = {"transcription": {"segments": [seg("seg_0001", 0, 5, "hello"), seg("seg_0002", 9, 12, "again")]},
               "events": {"events": [evt("evt_0001", "Door", 5, 7, .8), evt("evt_0002", "Alarm", 7, 9, .9), evt("evt_0003", "Cough", 3, 3.5, .6)]},
               "features": {"values": {"silence_intervals": []}}}
    tl = build_timeline(results, 12.0)
    ids = [n["id"] for n in tl["nodes"] if n["type"] in {"speech_segment", "sound_event"}]
    assert ids == ["seg_0001", "evt_0003", "evt_0001", "evt_0002", "seg_0002"]
    rel = {(e["source"], e["target"], e["relation"]) for e in tl["edges"]}
    assert ("evt_0001", "evt_0002", "precedes") in rel and ("evt_0003", "seg_0001", "interrupts") in rel
    assert "00:05.0–00:07.0" in tl["text"]

def test_audio_id_on_demand(app, speech_en):  # Review Focus #3
    aid = extract_audio_features(app, FeaturesRequest(source=fp(speech_en)))["audio_id"]
    r = get_audio_timeline(app, TimelineRequest(source={"type": "audio_id", "audio_id": aid}))
    assert r["status"] == "success" and any(n["type"] == "speech_segment" for n in r["result"]["nodes"])
```

- [ ] **Step 2: Run** `uv run pytest tests/unit/test_timeline.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement** it. `analyze_audio` maps `timeline` → `analysis.timeline`.
- [ ] **Step 4: Run** the same command. Expected: PASS.
- [ ] **Step 5: Commit** `feat: temporal event graph and get_audio_timeline tool`.

### Task 16: Uncertainty analysis

**Files:**
- Create: `src/audio_mcp/analysis/uncertainty.py`
- Modify: `pipeline/stages.py` (stage `uncertainty`, deps `transcription`, `vad`; it reads `events`/`quality` from results if present), `eval/metrics.py` (add `word_error_rate`)
- Test: `tests/unit/test_uncertainty.py`

**Interfaces:**
- Produces:
  - `word_error_rate(ref: str, hyp: str) -> float`: word-level Levenshtein / len(ref words), lowercased, punctuation stripped. 0 when both are empty.
  - `find_uncertainties(results, *, low_conf: float, second_pass: list[dict] | None = None) -> dict` with `{"items", "summary": {"low_confidence_word_ratio", "conflict_count", "asr_cross_check": dict | None}}`.
  - Item = `{type, start, end, detail, evidence_ids}`. Types:
    - `LOW_CONFIDENCE_SEGMENT`: segment conf < `low_conf`.
    - `LOW_CONFIDENCE_WORDS`: runs of ≥ 2 consecutive words < `low_conf`.
    - `ASR_WITHOUT_VAD`: segment with < 20% overlap with VAD.
    - `VAD_WITHOUT_ASR`: VAD region ≥ 1 s with no segment overlap.
    - `TAGGER_VAD_DISAGREEMENT`: a "Speech" event ≥ 1 s with < 20% VAD overlap, or a VAD region ≥ 1 s with no "Speech" event overlap (only when events exist).
    - `QUALITY_AFFECTED`: a high-severity quality issue.
  - The stage: when `options.budget == "thorough"` and ASR is available, it runs a second pass with `beam_size=1, temperature=0.4`. `asr_cross_check = {"word_disagreement_rate": word_error_rate(primary, second), "method": "beam-5 vs greedy t=0.4 decode"}`.
  - `TASK_TO_STAGES["uncertainty"] = ["uncertainty"]`. `analyze_audio` maps → `analysis.uncertainty`.

- [ ] **Step 1: Write failing tests**

```python
def test_flags():
    r = {"transcription": {"segments": [seg("seg_0001", 0, 2, "a b c", conf=.3, words=[w(0, .5, "a", .2), w(.5, 1, "b", .3), w(1, 2, "c", .9)]),
                                        seg("seg_0002", 5, 6, "x", conf=.9)]},
         "vad": {"speech_regions": [{"start": 0, "end": 2}, {"start": 8, "end": 10}]}}
    types = {i["type"] for i in find_uncertainties(r, low_conf=.5)["items"]}
    assert {"LOW_CONFIDENCE_SEGMENT", "LOW_CONFIDENCE_WORDS", "ASR_WITHOUT_VAD", "VAD_WITHOUT_ASR"} <= types

def test_wer():
    assert word_error_rate("a b c d", "a x c") == pytest.approx(0.5)

def test_thorough_cross_check(app, speech_en):
    o = run_analysis(app, FilePathSource(type="file_path", path=str(speech_en)), ["uncertainty"], {"budget": "thorough"})
    assert o.results["uncertainty"]["summary"]["asr_cross_check"]["word_disagreement_rate"] >= 0
```

- [ ] **Step 2: Run** `uv run pytest tests/unit/test_uncertainty.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement** it.
- [ ] **Step 4: Run** the same command. Expected: PASS.
- [ ] **Step 5: Commit** `feat: uncertainty and conflict reporting`.

### Task 17: Adaptive orchestration (auto plan, budgets, recovery)

**Files:**
- Modify: `src/audio_mcp/pipeline/planner.py`, `tools/handlers.py`
- Test: `tests/unit/test_planner.py`

**Interfaces:**
- Produces: `plan_auto(app, record, audio, *, goal: str | None, budget: str) -> list[dict]` of `{"task", "reason"}`. It runs/caches the `vad` stage via `run_analysis` to get `speech_ratio`, then applies these rules in order:
  1. `acoustic_features` and `audio_quality` always ("cheap; always run").
  2. `speech = speech_ratio >= settings.speech_ratio_threshold`. If true → `transcription`, `language_detection`.
  3. `speaker_diarization` when speech and (`duration > 20` or goal ~ `meeting|interview|speaker|conversation|call`) and `budget != "fast"`.
  4. `sound_event_detection` unless (`budget == "fast"` and `speech_ratio > 0.8`). The goal ~ `machine|industrial|environment|alarm|noise|sound` forces it.
  5. `summarization` when speech or the goal ~ `summar|key points|meeting`.
  6. `timeline` when events or diarization are planned.
  7. `integrity` when the goal ~ `authentic|edit|tamper|splice|fake|synthetic`.
  8. `speech_characteristics` when speech and `budget != "fast"`.
  9. `uncertainty` when `budget != "fast"` (the thorough cross-check comes via Task 16's budget option).
- Unavailable tasks stay in the plan, so the runner reports their skip reason.
- `analyze_audio` with `tasks="auto"` sets `processing.plan`. Recovery runs when `settings.auto_preprocess`, the quality has a high-severity `LOW_LEVEL|HIGH_NOISE|HUM|DC_OFFSET` issue, and transcription ran: `analysis.recovery = compare_preprocessed_asr(...)` (Task 13).

- [ ] **Step 1: Write failing tests**

```python
def plan_tasks(app, path, goal, budget):
    rec = app.store.ingest(FilePathSource(type="file_path", path=str(path)))
    return {p["task"] for p in plan_auto(app, rec, app.store.load(rec.audio_id), goal=goal, budget=budget)}

def test_speech_plan(app, speech_en):
    assert {"transcription", "speaker_diarization", "summarization"} <= plan_tasks(app, speech_en, "meeting notes", "balanced")

def test_noise_plan(app, synth):
    t = plan_tasks(app, synth["impulses"], "machine noise", "balanced")
    assert "sound_event_detection" in t and "transcription" not in t

def test_fast_budget(app, speech_en):
    assert "speaker_diarization" not in plan_tasks(app, speech_en, None, "fast")

def test_plan_reported(app, speech_en):
    r = analyze_audio(app, AnalyzeRequest(source=fp(speech_en), tasks="auto"))
    assert r["processing"]["plan"] and all({"task", "reason"} <= p.keys() for p in r["processing"]["plan"])
```

  `impulses` under EnergyVAD gives a low speech ratio. If the ratio crosses the threshold, adjust the fixture's burst length, not the rule.
- [ ] **Step 2: Run** `uv run pytest tests/unit/test_planner.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement** it.
- [ ] **Step 4: Run** the same command. Expected: PASS.
- [ ] **Step 5: Commit** `feat: adaptive analysis planning with budgets and recovery`.

### Task 18: ask_audio (rule-based, evidence-grounded)

**Files:**
- Create: `src/audio_mcp/reasoning/__init__.py`, `src/audio_mcp/reasoning/ask.py`, `src/audio_mcp/backends/text_embed.py`
- Modify: `backends/base.py` (real → `MiniLMEmbedder`), `tools/*`, `server.py`, e2e `EXPECTED_TOOLS`
- Test: `tests/unit/test_ask.py`

**Interfaces:**
- Produces:
  - `MiniLMEmbedder(settings)`: `SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", cache_folder=str(models_dir/"st"), device=dev)`, `encode(texts, normalize_embeddings=True)`. `info.stage="text_embedding"`.
  - `classify_intent(q) -> tuple[str, dict]`. Lowercased regexes, checked in this order:
    1. `speakers`: `how many (speakers|people|voices)|who (speaks|talks)`
    2. `time_of_mention`: `when (does|did|is|do) .*?(mention|say|says|talk about|discuss)(es)? (?P<topic>.+?)\??$`
    3. `relative_to_event`: `(?P<rel>before|after|during|while) (the )?(?P<target>[a-z ,]+?)( sound| sounds| starts| begins| goes off| rings)?\??$`
    4. `presence`: `^(is|are) there (any |a |an )?(?P<target>.+?)\??$`
    5. `unusual`: `unusual|strange|odd|anomal|unexpected`
    6. `change`: `chang|differ` combined with `(beginning|start)` and `end`
    7. else `fallback`
  - `resolve_labels(target: str, labels: list[str]) -> list[str]`: union of the `EVENT_CATEGORIES` entries whose key or any label word appears in the target, plus labels containing the target (case-insensitive).
  - `answer_question(question, results, duration, text: TextEmbedder | None) -> dict` with `{"answer", "support": "direct|inferred|insufficient", "intent", "evidence": [Evidence dict], "confidence"?, "confidence_kind"?}`.
    - `relative_to_event`:
      - Events matching `resolve_labels`.
      - `before`: timeline nodes (non-transition) with `end` in `[event.start - 5, event.start + 0.05]`.
      - `after`: symmetric.
      - `during`/`while`: nodes linked by `during|interrupts|overlaps`.
      - No matching event → `insufficient`, "No <target> was detected in the recording."
      - No neighbour nodes → `direct`, "Nothing was detected in the 5 s before <label> at M:SS.s."
    - `time_of_mention`:
      - Score = 0.5·keyword overlap (lowercased, stopword-stripped, suffix-stemmed `s|es|ed|ing`) + 0.5·cosine (when `text` exists; else keyword only).
      - Keep segments with score ≥ 0.35, max 5.
      - `direct` if keyword overlap > 0, else `inferred`. None → `insufficient`.
    - `presence`: matching events → `direct`. None → `direct` with "No <target> detected (threshold <t>)." and an evidence entry `{"id": "detector", "source": "events", "label": "<none>", "start": 0, "end": duration}`.
    - `unusual`: events whose label is outside the top 3 labels by total duration and whose total duration < 10% of the file, plus quality issues and integrity observations if cached. Evidence `kind="interpretation"`; `support="inferred"`.
    - `change`: first vs last third: `rms_dbfs`, `spectral_centroid_hz` (via `window_features`-style framing, implemented locally with `frame_rms` + STFT centroid), event label sets, speaker sets. Measured deltas only; `support="direct"`.
    - `speakers`: from diarization; missing → `insufficient` "Diarization unavailable: <reason>".
    - `fallback`: top 3 segments/events by cosine (or keyword). Best < 0.35 → `insufficient`, "The recording does not contain enough evidence to answer this question."
  - `confidence`: the max `confidence` among `detected` evidence, with `confidence_kind` copied; omitted when absent. Timestamps in answers are formatted `M:SS.s`.
  - `AskRequest`: `source, question: str = Field(min_length=3, max_length=1000), include_evidence: bool = True, use_llm: bool | None = None`. `use_llm` is honoured in Task 28; before that, `True` → `LLM_DISABLED`.
  - The handler runs `["timeline", "uncertainty"]` (+ `diarization`, `speaker_assign` for `speakers`) on demand. `include_evidence=False` drops `evidence`.

- [ ] **Step 1: Write failing tests**

```python
def test_mention(app, speech_en):
    r = ask_audio(app, AskRequest(source=fp(speech_en), question="When does the speaker mention the deadline Friday?"))["result"]
    assert r["support"] == "direct" and "friday" in r["evidence"][0]["text"].lower()
    assert r["evidence"][0]["start"] < r["evidence"][0]["end"] and r["evidence"][0]["source"] == "transcription"

def test_before_alarm(app, synth):
    r = ask_audio(app, AskRequest(source=fp(synth["alarm"]), question="What happens immediately before the alarm sounds?"))["result"]
    assert r["intent"] == "relative_to_event" and r["support"] == "direct" and all(e["end"] <= 1.1 for e in r["evidence"] if e["source"] != "events")

def test_absent_event(app, speech_en):
    r = ask_audio(app, AskRequest(source=fp(speech_en), question="Is there a dog barking?"))["result"]
    assert r["answer"].startswith("No") and r["support"] == "direct"

def test_unanswerable(app, synth):
    r = ask_audio(app, AskRequest(source=fp(synth["noise"]), question="What is the speaker's favourite colour?"))["result"]
    assert r["support"] == "insufficient" and "confidence" not in r

def test_unknown_audio_id(app):  # Review Focus #3
    r = ask_audio(app, AskRequest(source={"type": "audio_id", "audio_id": "aud_0000000000000000"}, question="what happens?"))
    assert r["error"]["code"] == "AUDIO_NOT_FOUND"

def test_audio_id_runs_on_demand(app, speech_en):  # Review Focus #3
    aid = extract_audio_features(app, FeaturesRequest(source=fp(speech_en)))["audio_id"]
    r = ask_audio(app, AskRequest(source={"type": "audio_id", "audio_id": aid}, question="When does the speaker mention the report?"))
    assert r["status"] == "success" and "transcription" in r["processing"]["stages_run"]
```

  In `test_before_alarm`, the evidence before the beep at ~1.0 s is the silence node at 0–1 s (fake profile; the 0.001 noise floor is < −50 dBFS).
- [ ] **Step 2: Run** `uv run pytest tests/unit/test_ask.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement** it.
- [ ] **Step 4: Run** the same command. Expected: PASS.
- [ ] **Step 5: Commit** `feat: evidence-grounded ask_audio`.

### Task 19: summarize_audio (extractive) and speech characteristics

**Files:**
- Create: `src/audio_mcp/reasoning/summarize.py`, `src/audio_mcp/dsp/speech.py`
- Modify: `pipeline/stages.py` (stage `summary`, deps `timeline`; stage `speech`, deps `transcription`, `vad`, `features`), `tools/*`, `server.py`, e2e `EXPECTED_TOOLS`
- Test: `tests/unit/test_summarize.py`, `tests/unit/test_speech.py`

**Interfaces:**
- Produces:
  - `summarize(results, duration, text: TextEmbedder | None, *, length="short", focus=None) -> dict` with `{"text", "sentences": [{"text","evidence_ids"}], "topics": [{"label","evidence_ids"}], "key_moments": [{"timestamp","description","evidence_ids"}], "method": "extractive"}`.
    - Sentence count: short 2, medium 4, long 8 (capped at the segment count).
    - Segment score = 0.6·centrality (mean cosine to the others; 0 without an embedder) + 0.4·TF-IDF salience (stopwords removed). With `focus`: + 0.5·similarity to the focus. Chosen segments are emitted in time order; sentence text = the segment text.
    - `topics`: top 5 TF-IDF terms, each with the ids of the segments containing it.
    - `key_moments` (max 10, time order): the first speech, speaker changes, and non-speech events with confidence ≥ 0.5.
    - Without speech, `text` is built from events and features, e.g. "4.0 s recording. Detected: Beep, bleep (1.0–3.0 s). Silence covers 48% of the recording."; `sentences` link to the event ids.
  - `speech_characteristics(results) -> dict` with `{"words_per_minute", "articulation_rate_wpm", "pause_count", "mean_pause_seconds", "long_pauses", "f0_range_hz", "f0_std_hz", "per_speaker"?, "notes": "emotion and tone are not inferred"}`.
    - `wpm` = words / VAD speech seconds × 60.
    - A pause is an inter-word gap ≥ 0.25 s; long pauses are ≥ 1.0 s.
    - `articulation_rate_wpm` = words / (speech seconds − pause seconds) × 60.
    - f0 from the `features` values.
    - `per_speaker` uses `speaker_assign` if present.
  - `SummarizeRequest`: `source, length: Literal["short","medium","long"] = "short", focus: str | None = None, use_llm: bool | None = None`.
  - `analyze_audio` maps `summary` → `analysis.summary` and `speech` → `analysis.speech_characteristics`.

- [ ] **Step 1: Write failing tests**

```python
def test_extractive_grounded(app, dialogue):
    r = summarize_audio(app, SummarizeRequest(source=fp(dialogue), length="short"))
    seg_ids = {s["id"] for s in transcribe_audio(app, TranscribeRequest(source=fp(dialogue)))["result"]["segments"]}
    res = r["result"]; assert len(res["sentences"]) == 2 and res["method"] == "extractive"
    assert all(set(s["evidence_ids"]) <= seg_ids for s in res["sentences"])

def test_nonspeech_summary(app, synth):
    r = summarize_audio(app, SummarizeRequest(source=fp(synth["alarm"])))["result"]
    assert "Beep" in r["text"] and r["key_moments"]

def test_wpm():
    res = {"transcription": {"segments": [seg("seg_0001", 0, 6, "", words=[w(i * .5, i * .5 + .4, "x", .9) for i in range(12)])]},
           "vad": {"speech_regions": [{"start": 0, "end": 6}]}, "features": {"values": {}}}
    s = speech_characteristics(res); assert s["words_per_minute"] == pytest.approx(120, abs=1) and s["pause_count"] == 0
```

- [ ] **Step 2: Run** `uv run pytest tests/unit/test_summarize.py tests/unit/test_speech.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement** it.
- [ ] **Step 4: Run** the same command. Expected: PASS.
- [ ] **Step 5: Commit** `feat: extractive summaries and speech characteristics`.

**Phase 2 gate:** fast tests green. The e2e `EXPECTED_TOOLS` holds 8 tools.

---

# Phase 3 — Advanced analytics

### Task 20: compare_audio and baseline anomalies

**Files:**
- Create: `src/audio_mcp/compare/__init__.py`, `src/audio_mcp/compare/compare.py`
- Modify: `pipeline/stages.py` (stage `audio_embedding`, backend `tagger` → `{"embedding": list[float]}`), `tools/*`, `server.py`, e2e `EXPECTED_TOOLS`
- Test: `tests/unit/test_compare.py`

**Interfaces:**
- Produces:
  - `window_features(native, sr, window_seconds=1.0) -> tuple[np.ndarray, list[str]]`. Per non-overlapping window: `rms_dbfs` (floor −100), `spectral_centroid_hz`, `spectral_flatness`, `zero_crossing_rate`, and the 7 band fractions. Names are returned alongside.
  - `compare_pair(a: dict, b: dict) -> dict`. Bundles are `{"audio_id", "features": features_result, "events": [..], "duration", "transcript": str | None, "embedding": list | None}`. Output:
    - `measured_differences.features`: `{key: {"a","b","delta","unit"}}` for the numeric scalar keys.
    - `measured_differences.bands`: per-band delta.
    - `measured_differences.events`: `{"only_in_a", "only_in_b", "rate_change": {label: {"a_per_min","b_per_min"}}}`.
    - `measured_differences.transcript`: `{"word_difference_rate", "a_only_words"[:20], "b_only_words"[:20]}`, only when both transcripts exist.
    - `similarity.audio_embedding_cosine`, when both embeddings exist; otherwise `similarity = {}` and the warning `EMBEDDING_UNAVAILABLE`.
    - `possible_interpretations`: `{"statement", "based_on", "kind": "interpretation", "source": "rule"}` from fixed rules:
      - |rms delta| > 6 dB → "<B|A> is louder; this may reflect gain, distance, or a change in the source."
      - |centroid delta| > 25% → "Spectral balance shifted toward <higher|lower> frequencies; this may reflect a different source, microphone, or filtering."
      - `only_in_b` non-empty → "Sound types present only in B: …; their cause is not determined."
  - `baseline_anomalies(target, baselines, *, z_threshold=3.0, window_seconds=1.0) -> dict` with `{"feature_z", "anomalous_features", "anomalous_windows": [{"start","end","features": {name: z}}], "baseline_windows": int}`.
    - Baseline mean/std come from all baseline windows pooled; std floor = 1e-6·|mean| + 1e-9.
    - `feature_z[name]` = (target window mean − baseline mean) / baseline std.
    - A window is anomalous when any feature's |z| ≥ threshold. Adjacent anomalous windows merge, and their per-feature z is the max |z|.
  - `CompareRequest`: `sources: list[AudioSource] = Field(min_length=1, max_length=10)`, `baseline: list[AudioSource] | None = None`, `aspects: list[Literal["features","events","transcript","embedding"]] | None = None`, `z_threshold: float = Field(3.0, gt=0)`. A model validator requires ≥ 2 sources without a baseline.
  - `compare_audio` handler:
    - Per recording, runs `metadata`, `features`, `events`, `audio_embedding`, plus `transcription` when the aspect `transcript` is requested.
    - No baseline: `comparisons = [compare_pair(sources[0], s) for s in sources[1:]]`.
    - With a baseline: `baseline = {audio_id: baseline_anomalies(...)}` for each source.
    - `result.audio_ids` lists all ids; the top-level `audio_id` is omitted.

- [ ] **Step 1: Write failing tests**

```python
def test_pair_level_delta(app, synth):
    r = compare_audio(app, CompareRequest(sources=[fp(synth["sine"]), fp(synth["low_level"])]))["result"]["comparisons"][0]
    d = r["measured_differences"]["features"]["rms_dbfs"]; assert d["unit"] == "dBFS" and d["delta"] < -30
    assert any(i["kind"] == "interpretation" for i in r["possible_interpretations"])

def test_baseline_localizes_anomaly():
    base = (np.sin(2 * np.pi * 220 * np.arange(16000 * 5) / 16000) * .3).astype("float32")[None]
    tgt = base.copy(); tgt[0, 32000:48000] += np.random.default_rng(0).normal(0, .3, 16000).astype("float32")
    out = baseline_anomalies(window_features(tgt, 16000), [window_features(base, 16000)])
    assert out["anomalous_windows"][0]["start"] == pytest.approx(2.0, abs=0.01) and out["anomalous_windows"][0]["end"] == pytest.approx(3.0, abs=0.01)

def test_requires_two_without_baseline():
    with pytest.raises(ValidationError): CompareRequest(sources=[{"type": "file_path", "path": "x.wav"}])

def test_baseline_mode(app, synth):
    r = compare_audio(app, CompareRequest(sources=[fp(synth["noise"])], baseline=[fp(synth["sine"])]))["result"]
    assert r["baseline"] and next(iter(r["baseline"].values()))["anomalous_features"]
```

- [ ] **Step 2: Run** `uv run pytest tests/unit/test_compare.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement** it.
- [ ] **Step 4: Run** the same command. Expected: PASS.
- [ ] **Step 5: Commit** `feat: compare_audio with baseline anomaly localization`.

### Task 21: Library index and search

**Files:**
- Create: `src/audio_mcp/library/__init__.py`, `src/audio_mcp/library/index.py`
- Modify: `context.py` (`library = Library(settings.data_dir/"library.sqlite", backends.text if available else None)`), `tools/*`, `server.py`, e2e `EXPECTED_TOOLS`
- Test: `tests/unit/test_library.py`

**Interfaces:**
- Produces:
  - `SUPPORTED_EXTENSIONS = {".wav",".flac",".mp3",".ogg",".opus",".m4a",".aac",".webm",".aiff",".aif",".wma"}`.
  - `class Library(db_path, text: TextEmbedder | None)`. The schema is created on init:

    ```sql
    CREATE TABLE IF NOT EXISTS recordings(audio_id TEXT PRIMARY KEY, path TEXT, sha256 TEXT, size_bytes INTEGER, duration REAL,
      language TEXT, speaker_count INTEGER, labels TEXT, indexed_at REAL, embedding BLOB);
    CREATE TABLE IF NOT EXISTS segments(rowid INTEGER PRIMARY KEY, id TEXT UNIQUE, audio_id TEXT, kind TEXT, start REAL, end REAL,
      text TEXT, label TEXT, speaker_id TEXT, embedding BLOB);
    CREATE VIRTUAL TABLE IF NOT EXISTS segments_fts USING fts5(text, content='segments', content_rowid='rowid');
    ```

    plus insert/delete triggers syncing `segments_fts`.
  - Methods:
    - `add(record, source_path, results, audio_embedding) -> int`:
      - Replaces existing rows. Speech segments come from the transcription (with `speaker_assign` if present); event segments use `kind="event"`, `text=label`.
      - Text embeddings are stored as float32 blobs. Returns the count.
    - `search(query, *, filters: dict, limit=10, query_audio_embedding=None) -> list[dict]`:
      - Candidates: the FTS `MATCH` top 200 (query tokens quoted and OR-joined), plus cosine over all segment vectors when an embedder exists (numpy, 50k-row chunks), plus recording-level audio cosine when `query_audio_embedding` is given.
      - Scores: `bm25 = 1/(1+rank_index)`, `text_cosine`, `audio_cosine`; `combined` = the mean of the present scores.
      - Filters: `language`, `min_duration`, `max_duration`, `event_label` (in recording labels), `speaker_count`, `audio_ids`.
      - Match: `{audio_id, path, segment_id, kind, start, end, excerpt, label?, speaker_id?, scores, source_available}`. `source_available` = the path exists and size + sha256 match.
    - `list(limit=100, offset=0) -> list[dict]`, `remove(audio_id) -> bool`.
  - `SearchLibraryRequest`: `action: Literal["index","search","list","remove"]`, `paths: list[str] | None`, `query: str | None`, `query_audio: AudioSource | None`, `filters: LibraryFilters | None`, `limit: int = Field(10, ge=1, le=100)`, `audio_id: str | None`, `async_mode="auto"`. A validator enforces the per-action required fields.
  - Handler:
    - `index` expands folders recursively by extension. Each path is checked with `check_path_allowed`.
    - Per file: `run_analysis([... "language","events","audio_embedding","transcription"])`, then `library.add`.
    - Returns `{"indexed": [{"path","audio_id","segments"}], "failed": [{"path","error"}]}`.
    - `search` → `{"matches"}`, `list` → `{"recordings"}`, `remove` → `{"removed": bool}`.

- [ ] **Step 1: Write failing tests**

```python
def test_index_and_keyword_search(app, speech_en, dialogue):
    out = search_audio_library(app, SearchLibraryRequest(action="index", paths=[str(speech_en), str(dialogue)]))["result"]
    assert len(out["indexed"]) == 2 and out["failed"] == []
    m = search_audio_library(app, SearchLibraryRequest(action="search", query="report"))["result"]["matches"]
    assert m and "combined" in m[0]["scores"] and m[0]["start"] < m[0]["end"] and "report" in m[0]["excerpt"].lower()

def test_event_filter(app, synth, speech_en):
    search_audio_library(app, SearchLibraryRequest(action="index", paths=[str(synth["alarm"]), str(speech_en)]))
    m = search_audio_library(app, SearchLibraryRequest(action="search", query="beep", filters={"event_label": "Beep, bleep"}))["result"]["matches"]
    assert m and all(x["path"].endswith("alarm.wav") for x in m)

def test_missing_source_flag(app, tmp_path, speech_en):
    p = tmp_path / "copy.wav"; shutil.copy(speech_en, p)
    search_audio_library(app, SearchLibraryRequest(action="index", paths=[str(p)])); p.unlink()
    m = search_audio_library(app, SearchLibraryRequest(action="search", query="report"))["result"]["matches"]
    assert any(x["source_available"] is False for x in m)

def test_index_rejects_outside_root(app):
    r = search_audio_library(app, SearchLibraryRequest(action="index", paths=[str(Path.home().parent / "nope.wav")]))
    assert r["result"]["failed"][0]["error"]["code"] in {"PATH_NOT_ALLOWED", "FILE_NOT_FOUND"}
```

- [ ] **Step 2: Run** `uv run pytest tests/unit/test_library.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement** it.
- [ ] **Step 4: Run** the same command. Expected: PASS.
- [ ] **Step 5: Commit** `feat: SQLite FTS5 + vector audio library search`.

### Task 22: Multilingual transcription and translation

**Files:**
- Modify: `src/audio_mcp/backends/whisper.py`, `pipeline/stages.py` (the `transcription` cfg includes `multilingual`; new stage `translation`, deps `transcription`, backend `asr`), `tools/handlers.py`
- Test: `tests/unit/test_multilingual.py`, `tests/integration/test_whisper_multilingual.py`

**Interfaces:**
- Produces:
  - `FasterWhisperASR.transcribe(..., per_chunk_language=True, chunks=regions)`:
    - Groups the VAD regions into chunks ≤ 30 s (split at region boundaries; a region > 30 s is split evenly).
    - Per chunk: `model.detect_language(chunk)` when present; else `model.transcribe(chunk[:30*16000])` info. Then `transcribe(chunk, language=lang)`, with times offset by the chunk start.
    - Each segment gets `language`. `languages` = consecutive same-language chunks merged into `{language, start, end, probability}` runs; `language` = the language with the most speech seconds.
  - `translate_segments(asr, mono16k, segments) -> list[Segment]`: for each non-`en` segment, `asr.transcribe(slice, task="translate", language=seg.language, word_timestamps=False)` and join the text into `segment.translation`. The segment timestamps are unchanged.
  - The `language` stage adds `"code_switching": bool`: true when ≥ 2 languages each have ≥ 2 s.
  - `transcribe_audio`:
    - `multilingual=True` → the option `multilingual` → per-chunk mode.
    - `translate_to_en=True` → the `translation` stage, whose segments replace the transcript segments in the result, plus the warning `TRANSLATION_TARGET_LIMITED` ("only English translation is supported locally").

- [ ] **Step 1: Write failing tests**

```python
def test_language_runs_merge(app, tmp_path, speech_en):
    x, sr = sf.read(speech_en, dtype="float32"); p = tmp_path / "two.wav"
    sf.write(p, np.concatenate([x, np.zeros(sr), x]), sr); app.settings.allowed_roots.append(tmp_path)
    r = transcribe_audio(app, TranscribeRequest(source=fp(p), multilingual=True, translate_to_en=True))
    assert [l["language"] for l in r["result"]["languages"]] == ["en", "es"]
    assert all(s.get("translation") for s in r["result"]["segments"] if s["language"] == "es")
    assert any(w["code"] == "TRANSLATION_TARGET_LIMITED" for w in r["warnings"])
```

  `tests/integration/test_whisper_multilingual.py` (`@pytest.mark.slow`):
  - Concatenate `speech_en.wav` + 1 s silence + `speech_es.ogg` with ffmpeg.
  - Assert the language runs include `en` then `es`, every `es` segment has a non-empty `translation`, and translated segments keep their start/end.
- [ ] **Step 2: Run** `uv run pytest tests/unit/test_multilingual.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement** it.
- [ ] **Step 4: Run** the unit tests and the slow test. Expected: PASS.
- [ ] **Step 5: Commit** `feat: per-chunk language ID, code-switching, English translation`.

### Task 23: Integrity indicators

**Files:**
- Create: `src/audio_mcp/dsp/integrity.py`
- Modify: `pipeline/stages.py` (stage `integrity`, deps `metadata`)
- Test: `tests/unit/test_integrity.py`

**Interfaces:**
- Produces:
  - `INTEGRITY_LIMITATIONS`, verbatim:

    > "These are signal-level observations that can also arise from normal recording, compression, or editing workflows. They are not evidence that a recording is authentic, manipulated, or synthetic, and no such determination is made."

  - `integrity_indicators(native, sr) -> dict` with `{"observations", "limitations", "synthetic_speech_detection": {"status": "not_available", "reason": "no local synthetic-speech model configured"}}`.
  - Observation = `{type, start, end, magnitude, unit, description}`. Computed on the channel mean:
    - `ZERO_RUN`: ≥ 5 ms of exact zeros where the 100 ms on each side has RMS > −45 dBFS.
    - `SPECTRAL_DISCONTINUITY`: STFT 1024/256 log-magnitude flux; robust z (median/MAD·1.4826) > 6.
    - `NOISE_FLOOR_JUMP`: the 10th-percentile frame RMS (dBFS) of adjacent 1 s windows differs by > 10 dB.
    - `BANDLIMIT_CHANGE`: the 99% rolloff of adjacent 1 s windows changes > 40%, with both windows above −50 dBFS.
    - `DC_STEP`: the window-mean change > 0.02.
  - Same-type observations within 50 ms merge.

- [ ] **Step 1: Write failing tests**

```python
def test_splice(synth):
    o = integrity_indicators(*load(synth["spliced"]))
    types = {(i["type"], round(i["start"], 1)) for i in o["observations"]}
    assert ("ZERO_RUN", 1.0) in types and any(t == "NOISE_FLOOR_JUMP" and abs(s - 2.0) <= 1.0 for t, s in types)
    assert "not evidence" in o["limitations"] and o["synthetic_speech_detection"]["status"] == "not_available"

def test_clean_sine_quiet(synth):
    assert integrity_indicators(*load(synth["sine"]))["observations"] == []
```

- [ ] **Step 2: Run** `uv run pytest tests/unit/test_integrity.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement** it. `analyze_audio` maps `integrity` → `analysis.integrity`.
- [ ] **Step 4: Run** the same command. Expected: PASS.
- [ ] **Step 5: Commit** `feat: integrity indicators with fixed limitations statement`.

**Phase 3 gate:** fast tests green. The e2e `EXPECTED_TOOLS` holds 10 tools.

---

# Phase 4 — Production readiness

### Task 24: Jobs (async, progress, cancel, timeout) + get_job

**Files:**
- Create: `src/audio_mcp/jobs.py`
- Modify: `pipeline/runner.py`, `context.py` (`jobs = JobManager(settings.max_concurrent_jobs)`), `tools/handlers.py`, `tools/schemas.py`, `server.py` (13th tool `get_job`; lifespan shutdown calls `jobs.shutdown()`), e2e `EXPECTED_TOOLS`
- Test: `tests/unit/test_jobs.py`

**Interfaces:**
- Produces:
  - `@dataclass JobInfo`: `job_id, tool, status: Literal["queued","running","done","failed","cancelled"], progress: float, created_at, updated_at, response: dict | None`.
  - `class JobManager(max_concurrent)`, backed by a `ThreadPoolExecutor`:
    - `submit(tool, fn: Callable[[threading.Event, Callable[[int,int],None]], dict]) -> str`. Ids `job_<12 hex>`.
    - `get(job_id) -> JobInfo`: `JOB_NOT_FOUND`.
    - `cancel(job_id) -> JobInfo`: queued → `cancelled` immediately; running → sets the event.
    - `prune(older_than: float) -> int`, `shutdown()`.
    - A job raising `AudioMCPError(CANCELLED)` ends `cancelled`; other exceptions end `failed` with `response = error_response(...)`.
  - `maybe_async(app, tool, req, run: Callable[[Event | None, Callable | None], dict]) -> dict`, in handlers:
    - Async when `req.async_mode == "always"`, or `"auto"` and the source probes to > `async_threshold_seconds`. The probe goes via `store.ingest`, so the response includes `audio_id`.
    - Returns `build_response(result=None, job_id=..., audio_id=...)`. Every audio-taking handler uses it.
  - Runner timeout: each stage runs via `ThreadPoolExecutor(1).submit(...).result(timeout=settings.stage_timeout_seconds)`. On `TimeoutError` → error `{"code": "TIMEOUT", "stage", "retryable": True}`; the worker thread is abandoned (documented limitation).
  - `GetJobRequest`: `job_id: str = Field(pattern=r"^job_[0-9a-f]{12}$"), action: Literal["status","cancel"] = "status"`.
  - `get_job` → `build_response(result={"job_status","progress","tool","response"?}, job_id=..., status=("processing" if not finished else "success"))`.

- [ ] **Step 1: Write failing tests**

```python
def test_async_roundtrip(app, speech_en):
    r = transcribe_audio(app, TranscribeRequest(source=fp(speech_en), async_mode="always"))
    assert r["status"] == "processing"; jid = r["job_id"]
    wait_until(lambda: get_job(app, GetJobRequest(job_id=jid))["result"]["job_status"] == "done")
    assert get_job(app, GetJobRequest(job_id=jid))["result"]["response"]["status"] == "success"

def test_cancel(app, speech_en, monkeypatch):
    monkeypatch.setattr(app.backends.asr, "transcribe", slow_transcribe(2.0))
    jid = analyze_audio(app, AnalyzeRequest(source=fp(speech_en), tasks=["transcription", "acoustic_features"], async_mode="always"))["job_id"]
    get_job(app, GetJobRequest(job_id=jid, action="cancel"))
    wait_until(lambda: get_job(app, GetJobRequest(job_id=jid))["result"]["job_status"] == "cancelled")

def test_stage_timeout(app, speech_en, monkeypatch):
    app.settings.stage_timeout_seconds = 0.2; monkeypatch.setattr(app.backends.asr, "transcribe", slow_transcribe(1.0))
    r = transcribe_audio(app, TranscribeRequest(source=fp(speech_en)))
    assert r["errors"][0]["code"] == "TIMEOUT" and r["errors"][0]["retryable"] is True

def test_unknown_job(app):
    assert get_job(app, GetJobRequest(job_id="job_000000000000"))["error"]["code"] == "JOB_NOT_FOUND"

@pytest.mark.long
def test_long_file_goes_async(app, tmp_path):
    app.settings.allowed_roots.append(tmp_path)
    assert extract_audio_features(app, FeaturesRequest(source=fp(make_long(tmp_path))))["status"] == "processing"
```

  `slow_transcribe(s)` (conftest) returns a function that sleeps `s` seconds, then delegates to `FakeASR`.
  - In `test_cancel`, `transcription` is placed before `acoustic_features`. Cancellation is checked between stages, so the job is cancelled after ASR returns.
- [ ] **Step 2: Run** `uv run pytest tests/unit/test_jobs.py -v -m "not long"`. Expected: FAIL.
- [ ] **Step 3: Implement** it.
- [ ] **Step 4: Run** the same command (PASS), then `uv run pytest tests/unit/test_jobs.py -m long -v` (PASS).
- [ ] **Step 5: Commit** `feat: async jobs with progress, cancel, stage timeouts, get_job tool`.

### Task 25: Chunked streaming sessions + analyze_audio_stream

**Files:**
- Create: `src/audio_mcp/stream/__init__.py`, `src/audio_mcp/stream/sessions.py`
- Modify: `context.py` (`streams = StreamSessions(app)`), `tools/*`, `server.py`, e2e `EXPECTED_TOOLS`
- Test: `tests/unit/test_stream.py`

**Interfaces:**
- Produces: `class StreamSessions(app)`.
  - `start(chunk_format: Literal["pcm","file"], sample_rate_hz=None, channels=None, sample_format: Literal["s16le","f32le"] | None = None) -> dict` → `{"session_id": "str_<12hex>"}`.
    - `pcm` without all three params → `INVALID_INPUT`.
    - The session holds a temp dir under `resolved_temp_dir`, a `native.f32` file (interleaved append), `native_sr`, channels, chunk boundaries, provisional segments/events, and `last_activity`.
    - A `file` session takes `native_sr`/channels from the first chunk; later chunks are resampled and down/up-mixed to match.
  - `push(session_id, data_b64=None, path=None) -> dict`:
    - Exactly one of `data_b64`/`path` (else `INVALID_INPUT`). A path goes through `check_path_allowed`.
    - Decoding: pcm via `np.frombuffer` (s16 → /32768), reshaped by channels; file via `probe` + `decode`. Then append.
    - The cumulative duration must not exceed `max_duration_seconds` (`DURATION_TOO_LONG`).
    - Runs `extract_features(["level"])` on the chunk, plus tagger events on the chunk at 16 kHz with times offset, plus ASR on the last ≤ 30 s at 16 kHz (when available). New segments with `start` ≥ the previous window end are stored as provisional.
    - Returns `{"chunk_index", "chunk_start", "chunk_end", "provisional": True, "features", "events", "transcript_window": {"start","end","text","segments"}}`.
  - `status(session_id) -> {"chunks", "duration_seconds", "state"}`.
  - `finish(session_id, tasks: list[str] | Literal["auto"] = "auto") -> dict`:
    - Writes `<dir>/stream.flac`, then `store.ingest_path(path, "stream-<id>.flac")`.
    - Runs the analysis the way `analyze_audio` does, sharing the helper `analysis_payload(outcome)`.
    - Reconciliation: `{"transcript_word_difference_rate": word_error_rate(provisional_text, final_text), "events_confirmed", "events_added", "events_dropped"}`, matching events by label + IoU ≥ 0.3.
    - Deletes the session dir. The response has `audio_id` and `result = {"analysis", "reconciliation"}`.
  - `cancel(session_id)`. Unknown/finished session → `INVALID_INPUT` "unknown session".
  - `expire(idle_seconds=3600) -> int`.
- `StreamRequest`: `action: Literal["start","push","status","finish","cancel"]`, `session_id: str | None`, `chunk_format: Literal["pcm","file"] | None`, `sample_rate_hz: int | None = Field(None, ge=8000, le=192000)`, `channels: int | None = Field(None, ge=1, le=8)`, `sample_format`, `data: str | None`, `path: str | None`, `tasks: list[TaskName] | Literal["auto"] = "auto"`.
- The tool description says: "Chunked sessions (raw PCM or self-contained audio files per chunk). Live socket streaming and split encoded streams are not supported."

- [ ] **Step 1: Write failing tests** (`stream(app, **kw) = analyze_audio_stream(app, StreamRequest(**kw))`)

```python
def test_pcm_session(app, speech_en):
    x, sr = sf.read(speech_en, dtype="int16")
    sid = stream(app, action="start", chunk_format="pcm", sample_rate_hz=sr, channels=1, sample_format="s16le")["result"]["session_id"]
    for part in np.array_split(x, 4):
        assert stream(app, action="push", session_id=sid, data=base64.b64encode(part.tobytes()).decode())["result"]["provisional"] is True
    f = stream(app, action="finish", session_id=sid, tasks=["transcription", "sound_event_detection"])
    assert f["status"] == "success" and "reconciliation" in f["result"] and f["audio_id"].startswith("aud_")
    assert abs(f["result"]["analysis"]["audio_metadata"]["duration_seconds"] - len(x) / sr) < 0.05

def test_file_chunks(app, synth):
    sid = stream(app, action="start", chunk_format="file")["result"]["session_id"]
    for k in ["sine", "alarm"]: stream(app, action="push", session_id=sid, path=str(synth[k]))
    assert stream(app, action="status", session_id=sid)["result"]["chunks"] == 2

def test_pcm_requires_params(app):
    assert stream(app, action="start", chunk_format="pcm")["error"]["code"] == "INVALID_INPUT"
```

- [ ] **Step 2: Run** `uv run pytest tests/unit/test_stream.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement** it.
- [ ] **Step 4: Run** the same command. Expected: PASS.
- [ ] **Step 5: Commit** `feat: chunked streaming sessions with provisional results and reconciliation`.

### Task 26: Retention, temp cleanup, redaction, log hygiene

**Files:**
- Create: `src/audio_mcp/store/retention.py`, `src/audio_mcp/export/__init__.py`, `src/audio_mcp/export/redact.py`
- Modify: `server.py` (lifespan: `cleanup_temp` + `sweep` at start; an hourly background task calling `sweep`; on shutdown `sweep(final=True)` and `cleanup_temp`), `tools/handlers.py` (`@tool_handler` applies `redact_analysis` when `settings.redact_outputs`)
- Test: `tests/unit/test_retention.py`, `tests/unit/test_redact.py`, `tests/unit/test_logging.py`

**Interfaces:**
- Produces:
  - `sweep(app, *, now=None, final=False) -> dict` with `{"audio_deleted", "results_deleted", "sessions_expired", "jobs_pruned", "library_rows_deleted"}`:
    - `retention="ttl"`: delete audio dirs (`meta.json.ingested_at`) and result dirs older than `retention_ttl_hours`.
    - `"none"`: delete all audio and results when `final`.
    - `"keep"`: delete nothing.
    - Library rows are deleted only when `library_retention == "ttl"`, using the same TTL.
    - It always calls `streams.expire()` and prunes jobs older than the TTL.
  - `cleanup_temp(settings) -> int`.
  - `redact_text(text, extra_patterns=()) -> tuple[str, int]`. Applied in order:
    1. email `[\w.+-]+@[\w-]+(\.[\w-]+)+` → `[EMAIL]`
    2. card: `(?:\d[ -]?){13,19}` passing Luhn → `[CARD]`
    3. phone `\+?\d[\d\s().-]{7,}\d` → `[PHONE]`
    4. extras → `[REDACTED]`
  - `redact_analysis(obj, extra_patterns) -> tuple[Any, int]`: recursive over dicts and lists; redacts the string values of keys `text`, `excerpt`, `translation`, `answer`, `statement`, `description`. When the count > 0, the handler adds the warning `REDACTED` with the count.

- [ ] **Step 1: Write failing tests**

```python
def test_ttl_sweep(app, speech_en):
    aid = extract_audio_features(app, FeaturesRequest(source=fp(speech_en)))["audio_id"]
    assert sweep(app, now=time.time() + 25 * 3600)["audio_deleted"] == 1
    with pytest.raises(AudioMCPError): app.store.get(aid)

def test_library_survives_sweep(app, speech_en):
    search_audio_library(app, SearchLibraryRequest(action="index", paths=[str(speech_en)]))
    sweep(app, now=time.time() + 25 * 3600)
    assert search_audio_library(app, SearchLibraryRequest(action="list"))["result"]["recordings"]

def test_redact():
    t, n = redact_text("mail a.b@x.org or call +1 (555) 123-4567, card 4111 1111 1111 1111")
    assert t == "mail [EMAIL] or call [PHONE], card [CARD]" and n == 3

def test_logs_hide_transcript(caplog, settings):
    setup_logging(settings); logging.getLogger("audio_mcp").info("x", extra={"text": "secret words"})
    assert "secret words" not in caplog.text
```

- [ ] **Step 2: Run** `uv run pytest tests/unit/test_retention.py tests/unit/test_redact.py tests/unit/test_logging.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement** it.
- [ ] **Step 4: Run** the same command. Expected: PASS.
- [ ] **Step 5: Commit** `feat: retention sweeps, temp cleanup, redaction, log hygiene`.

### Task 27: export_audio_analysis

**Files:**
- Create: `src/audio_mcp/export/exporters.py`
- Modify: `tools/*`, `server.py`, e2e `EXPECTED_TOOLS`
- Test: `tests/unit/test_export.py`

**Interfaces:**
- Produces:
  - `EXPORT_FORMATS = ("json","markdown","srt","vtt","csv","rttm")`.
  - `export_analysis(audio_id, analysis: dict, fmt, out_dir, *, redact: bool, extra_patterns) -> dict` with `{"path","format","size_bytes","sha256","redactions"}`. `analysis` is the `analyze_audio` payload shape; the file is named `<audio_id>-<YYYYmmddTHHMMSS>.<ext>` (`md` for markdown).
    - `json`: pretty-printed payload.
    - `srt`: numbered cues, `HH:MM:SS,mmm --> HH:MM:SS,mmm`, text prefixed `[speaker_x] ` when known.
    - `vtt`: `WEBVTT` + blank line; cues `HH:MM:SS.mmm --> …`; `<v speaker_x>` prefix.
    - `csv` (stdlib `csv`): header `type,id,start,end,speaker_id,label,text,confidence`; segment and event rows sorted by start.
    - `rttm`: `SPEAKER {audio_id} 1 {start:.3f} {dur:.3f} <NA> <NA> {speaker_id} <NA> <NA>` per turn from `analysis["speakers"]["turns"]`. No turns → `AudioMCPError(INVALID_INPUT, "no speaker turns; run with diarization")`.
    - `markdown`, in order: metadata table, summary, transcript (`**[M:SS.s] speaker_x:** text`), events table, quality issues, warnings.
  - `ExportRequest`: `audio_id: str = Field(pattern=r"^aud_[0-9a-f]{16}$"), format: Literal[EXPORT_FORMATS], tasks: list[TaskName] | Literal["auto"] = "auto", redact: bool = False`.
  - The handler builds the analysis (cache hits are expected) with the needed tasks, then writes to `resolved_export_dir`:
    - srt/vtt: `transcription` + `speaker_diarization` if available
    - rttm: `speaker_diarization`
    - json/markdown/csv: `tasks` (auto → the planner)

- [ ] **Step 1: Write failing tests**

```python
A = {"transcription": {"segments": [{"id": "seg_0001", "start": 1.2, "end": 4.8, "text": "We need to finish.", "speaker_id": "speaker_1"}]}}

def test_srt_vtt(tmp_path):
    s = Path(export_analysis("aud_0123456789abcdef", A, "srt", tmp_path, redact=False, extra_patterns=[])["path"]).read_text()
    assert "00:00:01,200 --> 00:00:04,800" in s and "[speaker_1] We need to finish." in s
    v = Path(export_analysis("aud_0123456789abcdef", A, "vtt", tmp_path, redact=False, extra_patterns=[])["path"]).read_text()
    assert v.startswith("WEBVTT") and "<v speaker_1>" in v

def test_rttm_line(tmp_path):
    a = {"speakers": {"turns": [{"speaker_id": "speaker_1", "start": 0.5, "end": 2.0}]}}
    t = Path(export_analysis("aud_0123456789abcdef", a, "rttm", tmp_path, redact=False, extra_patterns=[])["path"]).read_text()
    assert t.strip() == "SPEAKER aud_0123456789abcdef 1 0.500 1.500 <NA> <NA> speaker_1 <NA> <NA>"

def test_export_tool_hash(app, speech_en):
    aid = extract_audio_features(app, FeaturesRequest(source=fp(speech_en)))["audio_id"]
    r = export_audio_analysis(app, ExportRequest(audio_id=aid, format="json"))["result"]
    assert hashlib.sha256(Path(r["path"]).read_bytes()).hexdigest() == r["sha256"]
```

- [ ] **Step 2: Run** `uv run pytest tests/unit/test_export.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement** it.
- [ ] **Step 4: Run** the same command. Expected: PASS.
- [ ] **Step 5: Commit** `feat: export to json/markdown/srt/vtt/csv/rttm with optional redaction`.

### Task 28: Optional LLM backend for ask/summarize/compare

**Files:**
- Create: `src/audio_mcp/backends/llm.py` (move `NoLLM` here from `fake.py`), `src/audio_mcp/reasoning/llm_grounded.py`
- Modify: `backends/base.py`, `tools/handlers.py`
- Test: `tests/unit/test_llm.py`

**Interfaces:**
- Produces:
  - `is_loopback(url) -> bool`: the host is `localhost`, or every `getaddrinfo` address is loopback.
  - `NoLLM`: `available() -> (False, "llm_provider is none")`, `is_remote = False`.
  - `HttpLLM(settings)`:
    - `complete(system, prompt) -> str` via `httpx.Client(timeout=llm_timeout_seconds)`.
    - `anthropic`: POST `https://api.anthropic.com/v1/messages` with headers `x-api-key: os.environ[settings.llm_api_key_env]`, `anthropic-version: 2023-06-01`, `content-type: application/json`; body `{"model": llm_model, "max_tokens": 1024, "system": system, "messages": [{"role": "user", "content": prompt}]}`. Returns `data["content"][0]["text"]`.
    - `openai_compatible` / `ollama`: POST `{llm_base_url}/chat/completions` with `{"model", "messages": [{"role":"system",…},{"role":"user",…}], "temperature": 0}`, plus `Authorization: Bearer` if the key env is set. Returns `choices[0].message.content`.
    - `is_remote = not is_loopback(endpoint)`. `info = ModelInfo(stage="llm", name=provider, version=llm_model, local=not is_remote)`.
    - An HTTP error or timeout → `AudioMCPError(STAGE_FAILED, retryable=True)`.
  - `build_backends`: `llm = HttpLLM` when `llm_provider != "none"`, else `NoLLM`.
  - `build_evidence_pack(results, query, text) -> list[Evidence]`: the top 20 segments by the Task 18 scoring, plus events with confidence ≥ 0.5, capped at 40, in time order.
  - `llm_answer(question, pack, llm) -> dict`:
    - System prompt: "Answer only from the evidence. Cite evidence ids. If the evidence is insufficient, say so. Reply with JSON only."
    - User prompt: the evidence as JSON lines `{id,start,end,source,text|label}`, the question, and the schema `{"answer": str, "claims": [{"text", "evidence_ids": [...], "kind": "detected|interpretation"}], "insufficient": bool}`.
    - Parse the first `{…}` block. Drop claims citing ids not in the pack.
    - `support`: `insufficient` if flagged or no claims survive; `direct` if any surviving claim is `detected`; else `inferred`.
    - Returns the Task 18 fields plus `claims`, `method: "llm"`, and evidence = the cited pack items.
    - Parse failure → the caller falls back to the rule answer with the warning `LLM_OUTPUT_INVALID`.
  - `llm_summarize(results, pack, llm, length, focus) -> dict`: same pattern; JSON `{"sentences": [{"text","evidence_ids"}], "topics": [str]}`; sentences without valid ids are dropped; `method: "llm"`; `key_moments` reuse the extractive ones.
  - Handlers:
    - `use_llm=True` with `NoLLM` → `error_response(LLM_DISABLED)`.
    - `use_llm=None` with `NoLLM` → rule path.
    - On any LLM call: `processing.models_used` += `llm.info`, and `processing.data_sent_off_machine |= llm.is_remote`.
    - `compare_audio`: when an LLM is configured, append LLM interpretations with `"source": "llm"`, `"kind": "interpretation"`, fed only the `measured_differences` JSON.

- [ ] **Step 1: Write failing tests**

```python
def test_loopback():
    assert is_loopback("http://localhost:11434/v1") and is_loopback("http://127.0.0.1:8000") and not is_loopback("https://api.anthropic.com")

class StubLLM:
    is_remote = True; info = ModelInfo(stage="llm", name="stub", version="1", local=False)
    def __init__(self, out): self.out = out
    def available(self): return True, None
    def load(self): pass
    def complete(self, s, p): return self.out

def test_drops_uncited_claims():
    pack = [Evidence(id="seg_0001", start=1, end=2, source="transcription", text="report by friday")]
    out = llm_answer("deadline?", pack, StubLLM('{"answer":"Friday","claims":[{"text":"Friday","evidence_ids":["seg_0001"],"kind":"detected"},'
                                                '{"text":"Boss angry","evidence_ids":["seg_9"],"kind":"interpretation"}],"insufficient":false}'))
    assert [c["text"] for c in out["claims"]] == ["Friday"] and out["support"] == "direct"

def test_remote_flag_and_disabled(app, speech_en):
    assert ask_audio(app, AskRequest(source=fp(speech_en), question="deadline?", use_llm=True))["error"]["code"] == "LLM_DISABLED"
    app.backends.llm = StubLLM('{"answer":"x","claims":[],"insufficient":true}')
    r = ask_audio(app, AskRequest(source=fp(speech_en), question="deadline?", use_llm=True))
    assert r["processing"]["data_sent_off_machine"] is True and r["result"]["support"] == "insufficient"

def test_invalid_output_falls_back(app, speech_en):
    app.backends.llm = StubLLM("not json")
    r = ask_audio(app, AskRequest(source=fp(speech_en), question="When does the speaker mention the report?", use_llm=True))
    assert any(w["code"] == "LLM_OUTPUT_INVALID" for w in r["warnings"]) and r["result"]["support"] == "direct"
```

- [ ] **Step 2: Run** `uv run pytest tests/unit/test_llm.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement** it.
- [ ] **Step 4: Run** the same command. Expected: PASS.
- [ ] **Step 5: Commit** `feat: optional grounded LLM backend with off-machine reporting`.

### Task 29: Event metric, evaluation and benchmark scripts

**Files:**
- Modify: `src/audio_mcp/eval/metrics.py` (add `event_prf`)
- Create: `scripts/evaluate.py`, `scripts/benchmark.py`
- Test: `tests/unit/test_metrics.py` (extend), `tests/unit/test_scripts.py`

**Interfaces:**
- Produces:
  - `event_prf(ref, hyp, *, iou=0.5) -> {"precision","recall","f1","tp","fp","fn"}`: greedy matching by descending IoU, same label case-insensitive. Precision/recall/f1 are 0 when the denominator is 0.
  - `scripts/evaluate.py`:
    - Subcommands:
      - `wer --ref-dir D --audio-dir A`: pairs `stem.txt` ↔ `stem.<audio ext>`
      - `events --labels CSV --audio-dir A`: CSV `file,start,end,label`
      - `der --rttm-dir R --audio-dir A`
    - Options `--profile real|fake` (default real) and `--out report.json`.
    - Runs the handlers in-process. Prints JSON `{"metric", "per_file": [...], "aggregate": {...}}`; aggregate WER = the total-edits-weighted mean.
  - `scripts/benchmark.py --file F [--tasks …] [--repeat 3] [--profile real|fake]`:
    - Per stage: mean/min wall seconds and `rtf` = seconds / audio duration.
    - `peak_rss_mb` from psutil, sampled every 50 ms by a thread.
    - Clears the cache between repeats. Prints JSON.

- [ ] **Step 1: Write failing tests**

```python
def test_event_prf():
    ref = [{"label": "Dog", "start": 0, "end": 1}, {"label": "Car", "start": 2, "end": 3}]
    hyp = [{"label": "dog", "start": 0.1, "end": 1.0}, {"label": "Car", "start": 5, "end": 6}]
    assert event_prf(ref, hyp) == {"precision": .5, "recall": .5, "f1": .5, "tp": 1, "fp": 1, "fn": 1}

def test_benchmark_runs():  # tests/unit/test_scripts.py
    out = subprocess.run([sys.executable, "scripts/benchmark.py", "--file", str(FIXTURES_DIR / "speech_en.wav"),
                          "--tasks", "acoustic_features", "--repeat", "1", "--profile", "fake"], capture_output=True, text=True, check=True, env=bench_env())
    data = json.loads(out.stdout); assert data["stages"] and "rtf" in data["stages"][0] and data["peak_rss_mb"] > 0

def test_evaluate_wer_fake(tmp_path):
    ...  # copy speech_en.wav + speech_en.txt into tmp dir; run `evaluate.py wer --profile fake`; assert 0 <= aggregate.wer <= 1
```

  `bench_env()` sets `AUDIO_MCP_DATA_DIR` to a temp dir and `AUDIO_MCP_ALLOWED_ROOTS` to the fixtures dir.
- [ ] **Step 2: Run** `uv run pytest tests/unit/test_metrics.py tests/unit/test_scripts.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement** it.
- [ ] **Step 4: Run** the same command. Expected: PASS.
- [ ] **Step 5: Commit** `feat: event P/R metric, evaluation and benchmark scripts`.

### Task 30: Documentation, tool docs, full e2e acceptance

**Files:**
- Create: `README.md`, `docs/configuration.md`, `docs/capabilities.md`, `docs/evaluation.md`, `docs/tools.md` (generated), `docs/examples/<tool>.json`, `scripts/gen_tool_docs.py`
- Modify: `tests/e2e/test_stdio.py`
- Test: `tests/e2e/test_stdio.py`, `tests/unit/test_docs.py`

**Interfaces:**
- `scripts/gen_tool_docs.py`:
  - Starts the server over stdio (fake profile) and calls `list_tools`.
  - Writes `docs/tools.md`: per tool, the description, the input JSON Schema, and the example request/success/error from `docs/examples/<tool>.json`.
  - `--check` exits 1 when the output differs.
  - `--write-examples` regenerates the example files by calling each tool on the fixtures, with absolute paths replaced by `./samples/<name>`.
- README sections, in order:
  1. What it is
  2. Requirements: Python 3.12, uv, ffmpeg per OS (`winget install Gyan.FFmpeg` / `brew install ffmpeg` / `sudo apt-get install ffmpeg`)
  3. Install: `uv sync --extra models`; optional `--extra pyannote` + `HF_TOKEN`
  4. First-run downloads: whisper small ≈ 460 MB, PANNs CNN14 ≈ 330 MB, ECAPA ≈ 80 MB, MiniLM ≈ 90 MB; silero bundled
  5. Connect to Claude Code: `claude mcp add audio-intel -- uv --directory <path> run audio-mcp`
  6. Connect to Claude Desktop: a `claude_desktop_config.json` snippet with `"command": "uv"`, `"args": ["--directory", "<path>", "run", "audio-mcp"]`, `"env": {"AUDIO_MCP_ALLOWED_ROOTS": "[\"<dir>\"]"}`
  7. The 13 tools in one table
  8. Example workflows: meeting; machine baseline; environmental; library search; streaming
  9. Privacy: only an enabled non-loopback LLM sends data (text only); model downloads on first run; retention defaults
  10. Formats, limits and defaults
  11. Troubleshooting:
      - ffmpeg not found
      - ECAPA symlink errors on Windows
      - PANNs label CSV / wget
      - CPU speed and model size
      - `MODEL_UNAVAILABLE` meanings
      - nothing may print to stdout
  12. Licensing:
      - faster-whisper MIT; Whisper weights MIT
      - PANNs code MIT; CNN14 weights CC-BY 4.0
      - SpeechBrain and the ECAPA model Apache-2.0
      - all-MiniLM-L6-v2 Apache-2.0; silero-vad MIT
      - pyannote MIT + gated model terms
      - AudioSet labels CC-BY 4.0
      - fixtures: `tests/fixtures/audio/SOURCES.md`
- `docs/capabilities.md` has the headings `## Implemented`, `## Partial`, `## Future`.
  - Partial:
    - overlap detection (pyannote only)
    - translation (English target only)
    - SNR (estimate)
    - echo (heuristic)
    - tempo (heuristic)
    - synthetic-speech detection (not available; indicators only)
    - stage timeouts (thread abandoned, not killed)
  - Future:
    - live socket streaming
    - split encoded-stream chunks
    - emotion inference
    - ANN index (FAISS/sqlite-vec)
    - non-English translation targets
    - GPU tuning
- `docs/evaluation.md`:
  - The metric definitions (WER, event P/R at IoU 0.5, DER with a 0.25 s collar), plus script usage.
  - Datasets: LibriSpeech test-clean (WER), ESC-50 (events), AMI (DER), with links.
  - Expected CPU real-time factors measured on the dev machine by `benchmark.py`.
  - Limitations: TTS fixtures are not natural speech; fake-profile tests verify plumbing, not accuracy.
- `docs/configuration.md`: every `Settings` field with type, default, env var and meaning.

- [ ] **Step 1: Write failing tests**

```python
EXPECTED_TOOLS = {"analyze_audio", "transcribe_audio", "detect_audio_events", "inspect_audio_quality", "extract_audio_features",
    "ask_audio", "get_audio_timeline", "compare_audio", "search_audio_library", "summarize_audio", "analyze_audio_stream",
    "export_audio_analysis", "get_job"}

async def test_all_tools_invocable(stdio_session, samples_dir):
    assert {t.name for t in (await stdio_session.list_tools()).tools} == EXPECTED_TOOLS
    for name, args in acceptance_calls(samples_dir):  # one valid call per tool; chains audio_id / job_id / session_id from prior outputs
        out = json.loads((await stdio_session.call_tool(name, args)).content[0].text)
        assert out["status"] in {"success", "partial", "processing"}, (name, out)
        assert out["processing"]["schema_version"] == "1.0"

def test_docs_in_sync():
    assert subprocess.run([sys.executable, "scripts/gen_tool_docs.py", "--check"]).returncode == 0
    caps = Path("docs/capabilities.md").read_text(encoding="utf-8")
    assert all(h in caps for h in ["## Implemented", "## Partial", "## Future"])
```

  `samples_dir` copies `speech_en.wav`, `dialogue_2spk.wav` and the synthetic `alarm`/`sine` into a tmp dir passed as `AUDIO_MCP_ALLOWED_ROOTS`. `acceptance_calls` is a generator in the test file that reads the previous outputs through a closure dict.
- [ ] **Step 2: Run** `uv run pytest tests/e2e tests/unit/test_docs.py -v`. Expected: FAIL.
- [ ] **Step 3: Write** the docs, the generator, and `--write-examples` output.
- [ ] **Step 4: Run** `uv run pytest -m "not slow and not long"`, then `uv run pytest -m slow`, then `uv run pytest -m long`. Expected: all PASS.
- [ ] **Step 5: Manual acceptance in Claude Code.** Register using the README command, then:
  1. `analyze_audio` on `tests/fixtures/audio/dialogue_2spk.wav` with `tasks: "auto", goal: "meeting"`.
  2. `ask_audio` with "When does someone mention the budget?" and the returned `audio_id`.

  Expected: 2 speakers, an answer citing a segment with timestamps, and `data_sent_off_machine: false`.
- [ ] **Step 6: Commit** `docs: README, tool reference, capabilities, evaluation; full e2e acceptance`.

**Phase 4 gate = brief acceptance criteria:** CI green on all three OSes, every §9 acceptance criterion in the spec met, and `docs/capabilities.md` matching the code.
