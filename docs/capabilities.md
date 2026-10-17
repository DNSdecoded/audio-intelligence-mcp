# Capabilities

What works today, what works with stated limits, and what is not built. If something is not listed under
Implemented or Partial, the server does not do it.

## Implemented

- MCP server over stdio (FastMCP), 13 tools with strict input schemas and one response envelope
- Sources: file path inside `allowed_roots`, base64 upload, `audio_id` reuse; any ffmpeg-decodable format
- Content-addressed storage (`audio_id` = SHA-256 prefix) and per-stage result cache (same file via any source
  reuses results)
- Transcription with segment and word timestamps (faster-whisper), language identification
- Code-switching: per-chunk language ID (chunks split on ≥ 0.8 s gaps, ≤ 30 s), language runs
- Speaker diarization: ECAPA embeddings + agglomerative clustering, speaker-to-segment/word assignment
- Sound-event detection: PANNs CNN14 (527 AudioSet classes) on 1 s windows, merged intervals, categories
- Acoustic features with units: level, clipping, DC, silence intervals, spectral shape, band energy, YIN pitch,
  onset rate; bounded memory for long files
- Quality diagnosis with severities and recommendations; preprocessing variants (DC removal, mono, 80 Hz
  high-pass, spectral-gating denoise, silence trim, 16 kHz resample, loudness normalize); original never modified
- Before/after transcription confidence comparison for preprocessing (measured only)
- Adaptive planning (`tasks: "auto"`) from speech ratio, duration, goal text and budget, with reasons
- Uncertainty report: low-confidence segments/words, ASR–VAD and tagger–VAD conflicts, quality impact,
  `thorough` second-pass disagreement rate
- Temporal event graph with `precedes` / `during` / `overlaps` / `interrupts` edges and a text rendering
- Rule-based, evidence-grounded Q&A (mention timing, before/after/during events, presence, unusual sounds,
  change over time, speaker count) with `direct` / `inferred` / `insufficient` support
- Extractive summaries, topics and key moments linked to evidence ids
- Speech characteristics: speaking and articulation rate, pauses, pitch range (no emotion inference)
- Comparison: feature/band/event/transcript deltas, embedding similarity, baseline z-score anomalies localized
  per second, hedged rule-based interpretations kept separate from measurements
- Persistent library: SQLite FTS5 keyword search + text-embedding cosine + query-by-example audio, filters,
  stale-source detection
- Integrity observations: zero-runs, spectral discontinuities, noise-floor jumps, bandwidth changes, DC steps,
  with a fixed limitations statement
- Async jobs (auto for long audio), progress, cancellation, per-stage timeouts, `get_job`
- Chunked streaming sessions with provisional results and final reconciliation
- Exports: json, markdown, srt, vtt, csv, rttm
- Optional LLM (Anthropic SDK, OpenAI-compatible, Ollama) for answers, summaries and interpretations, limited to
  an evidence pack, uncited claims dropped, off-machine use reported
- Retention sweeps, temp cleanup, redaction (email/phone/Luhn-checked cards/custom), log hygiene
- Evaluation scripts (WER, event P/R/F1, DER) and per-stage benchmark (latency, RTF, peak RSS)

## Partial

- **URL sources** — off by default; blocks private/loopback/link-local/CGNAT/metadata addresses, but the
  hostname is re-resolved at connect time (DNS-rebinding window).
- **Overlap detection** — only with the optional pyannote backend; the default ECAPA path reports
  `OVERLAP_UNSUPPORTED`.
- **Diarization accuracy** — windows shorter than 0.8 s join the nearest speaker cluster, so a speaker with only
  very short utterances can be merged into another.
- **Translation** — English target only (Whisper translate task, per segment, timestamps preserved).
- **SNR** — an estimate (VAD-based when speech is present, percentile-based otherwise).
- **Echo, tempo, onset rate** — heuristics, labelled as such.
- **Synthetic-speech detection** — not available; integrity output gives signal indicators only.
- **Stage timeouts** — a timed-out stage is abandoned, not killed (Python threads cannot be killed).
- **Energy VAD fallback** — used only when Silero is unavailable; ignores speech quieter than about −40 dBFS.
- **Library scale** — vector search is brute force (bounded memory); fine to roughly a million segments.
- **Stdout hygiene** — Python-level prints are routed to stderr; output written directly to file descriptor 1
  by native code is not intercepted.

## Future

- Live socket/WebRTC streaming and split encoded-stream chunks (WebM/Opus, MP3 frames across chunks)
- Emotion or tone inference
- Approximate-nearest-neighbour index (FAISS / sqlite-vec) for very large libraries
- Non-English translation targets
- GPU-specific tuning and batching
- Multi-tenant authentication and remote (HTTP) transport hardening
