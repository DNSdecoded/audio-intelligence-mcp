# Build notes: Audio Intelligence MCP Server

Branch `feat/audio-intelligence-mcp` (off `main`), built 2026-10-17/18 from
`plans/2026-10-17-audio-intelligence-mcp.md` against `specs/2026-10-17-audio-intelligence-mcp-design.md`.

## How it was built

- Tasks 1-12: subagent implementer per task, independent reviewer per task (Tasks 1-5); controller self-review from
  Task 6 after the auto-mode classifier blocked reviewer dispatches.
- Tasks 13-30: implemented inline by the controller (implementer dispatch also blocked), self-reviewed.
- Final whole-branch review (Opus): 1 security finding + 9 Important; one fix wave (13 commits) fixed all; scoped
  re-review confirmed, no new Critical/Important.
- Final state: fast suite 287 passed / 1 skipped, slow (real models) 7 passed, long 1 passed (before fix wave).
- Real-model reference: WER 0.000 (`speech_en`), 0.133 (`dialogue_2spk`); DER 0.218 (`dialogue_2spk`).
- Not done: manual `claude mcp add` (edits user config; command in README).

## Rulings (decision — reason — cost if wrong)

Process
- Branch in place, no worktree — new repo — none.
- Test helpers added to `tests/conftest.py` by first user — keep helpers near first use — trivial.
- Missing RED capture accepted for new modules (Tasks 1, 6, 8-12) — expected RED was "module missing" — none.
- Self-review from Task 6; inline implementation from Task 13 — subagent dispatches blocked — less independent
  review, offset by final Opus review.

Architecture
- Unbuilt adapters wired as `UnavailableBackend(name, "not implemented yet")` until their task — none.
- `Stage.soft_deps`: timeline/summary/ask run even if transcription/events fail — spec graceful degradation —
  small runner rework.
- Language chunks split on gaps >= 0.8 s, max 30 s — deterministic, sane — coarser code-switch granularity.
- `analysis_payload` shared by analyze/stream/export — none.
- `mcp>=1.9,<2` — mcp 2.x renamed FastMCP — later migration.
- Diarization/speaker_assign only when requested; LATE_STAGES (timeline, uncertainty, summary) ordered last via
  `order_stages()` — avoids forced diarization — callers must order stages.
- Runner passes `options["_stages"]` so the uncertainty cache key reflects which stages ran — none.

DSP / models
- Decode: no `-xerror`; keep partial decodes with `decode_warnings`; CORRUPT_AUDIO if < 50% of probed duration
  decodes — graceful degradation — heavily truncated files rejected.
- Decoder subprocess timeout -> `TIMEOUT` (retryable) — none.
- HIGH_NOISE needs VAD-based SNR or spectral flatness >= 0.3 — steady tones were flagged — noisy tonal audio
  under-reported.
- Diarization windows < 0.8 s join nearest cluster centroid — short ECAPA embeddings unreliable — speaker with only
  very short utterances merged into another.
- `Word.speaker_id` added — per-word assignment — none.
- Slow diarization test uses Silero VAD — matches real pipeline — none.
- Timeline level-jump transitions from 1 s window dBFS — none.
- SPECTRAL_DISCONTINUITY needs robust z > 6 AND log-flux > 1.0 (MAD floored at 5% of median) — tones tripped on
  numerical noise — subtle splices in steady material missed.

Reasoning
- "Before X" includes nodes still active at X's onset — none.
- ask_audio runs quality only for "unusual", diarization only for "speakers"/"change" — cost — integrity
  observations only if cached.
- Speech summaries append up to 3 salient events (conf >= 0.5, not Speech/Silence/Noise) — spec §6.8 — up to
  3 sentences over the length budget.

Library / jobs / privacy
- `Library.list_recordings` (not `list`) — builtin shadowing — none.
- Library `source_available` = size + first/last-MiB hash — speed — middle-only edits of > 2 MiB files undetected.
- Async jobs inherit cancel/progress via `JOB_CONTEXT` contextvar; async handlers ingest first, rerun on audio_id —
  none.
- Hourly retention sweep in a daemon thread — task groups across lifespan yield are fragile — none.
- `redact_outputs` redacts every successful response (and exports) — none.
- Lifespan points `sys.stdout` at stderr (panns_inference prints on load) — protects JSON-RPC — native fd-1 writes
  not covered.
- `docs/configuration.md` generated with `docs/tools.md`; drift checked by `tests/unit/test_docs.py` — none.

LLM
- Anthropic via official SDK (optional `llm` extra, >=1.15); OpenAI-compatible/Ollama via httpx — one optional dep.
- Server-side refusal fallbacks (`fallbacks="default"`, beta `server-side-fallback-2026-07-01`) for
  claude-fable-5-1 / claude-opus-5-5 / claude-opus-5 / claude-sonnet-5-5; final refusal -> rule-based answer +
  `LLM_FAILED` warning — fallback model may answer instead.
- `use_llm=None` uses a configured LLM, `False` never; `data_sent_off_machine` / `models_used` set whenever a call
  is made — none.

## Parked residuals (known limits, documented in `docs/capabilities.md` where relevant)

- Playlist guard is a denylist (hls/concat/dash/image2/"playlist"); other file-opening demuxers (imf,
  avisynth/vapoursynth on some builds) not blocked. Next hardening: allowlist of audio container formats.
- `get_job` error envelope for a failed job keeps details in `result.response`; top-level `errors` empty.
- Multi-instance edges: a live instance's temp dir idle > 24 h can be removed; an audio_id shared by two instances is
  deleted by the first instance's `retention="none"` final sweep; crashed-instance audio under "none" never swept;
  a huge pid in a proc dir name aborts startup cleanup.
- Abandoned timed-out stage threads briefly exceed the concurrency cap.
- URL fetch (off by default) has a DNS-rebinding window (host re-resolved at connect).

## Deferred minors (not fixed)

- Settings: no numeric bounds; `Path.home()` evaluated at import.
- Envelope: explicit `status` not Literal-validated; `errors` accepts non-dicts; `"plan": null` and
  `"result": null` emitted.
- Sources: FILE_NOT_FOUND before root check leaks existence outside roots (plan-mandated order); base64 write
  OSError not wrapped; no Windows case-insensitivity test; no test asserting `trust_env=False`.
- Fixtures: `speech_es.txt` is a machine transcript (approximate); settings fixture builds synth for every test.
- Decode: stderr not surfaced in errors; warning path, sr/ch <= 0 and non-finite duration untested.
- Store: `put_variant` non-atomic; Windows `rmtree(ignore_errors)` may leave locked files.
- Features: onset heuristic tuned on the alarm fixture; YIN per-frame Python loop; up to 3 STFT passes.
- PANNs: input resampled to 32 kHz held in memory (~0.5 GB/h).
- Runner: VAD_FALLBACK warning text embeds raw exception text; abandoned timeout thread may append warnings.
- FastMCP validates params before handlers, so schema errors return as MCP tool errors, not the envelope.
- EnergyVAD fallback ignores speech below ~-40 dBFS.
