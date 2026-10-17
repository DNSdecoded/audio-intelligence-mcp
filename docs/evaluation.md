# Evaluation and benchmarks

## Metrics

| Metric | Definition | Script |
|---|---|---|
| WER | Word-level Levenshtein distance / reference words; lowercased, punctuation stripped; aggregate weighted by reference words | `scripts/evaluate.py wer` |
| Event P / R / F1 | Greedy one-to-one matching of same-label events (case-insensitive) with IoU ≥ 0.5 (`--iou`) | `scripts/evaluate.py events` |
| DER | Frame-based (10 ms) missed + false alarm + confusion over reference speech, optimal speaker mapping, ±0.25 s collar (`--collar`) | `scripts/evaluate.py der` |
| Latency / RTF / memory | Per-stage wall time, real-time factor (seconds / audio seconds), peak RSS sampled every 50 ms | `scripts/benchmark.py` |

The audio folder must be inside `allowed_roots`:

```bash
export AUDIO_MCP_ALLOWED_ROOTS='["/data/eval"]'
uv run python scripts/evaluate.py wer    --ref-dir /data/eval/librispeech --audio-dir /data/eval/librispeech --out wer.json
uv run python scripts/evaluate.py events --labels /data/eval/esc50/labels.csv --audio-dir /data/eval/esc50/audio
uv run python scripts/evaluate.py der    --rttm-dir /data/eval/ami/rttm --audio-dir /data/eval/ami/audio
uv run python scripts/benchmark.py --file /data/eval/meeting.wav --tasks transcription speaker_diarization --repeat 3
```

`--profile fake` runs the same plumbing with deterministic stand-in backends (no downloads; scores are meaningless).

## Public datasets (download yourself)

- **LibriSpeech test-clean** (WER) — https://www.openslr.org/12 — convert `*.trans.txt` to one `stem.txt` per file.
- **ESC-50** (sound events) — https://github.com/karolpiczak/ESC-50 — 5 s single-label clips; build
  `labels.csv` (`file,start,end,label`) mapping ESC-50 classes to AudioSet display names.
- **AMI meeting corpus** (DER) — https://groups.inf.ed.ac.uk/ami/corpus/ — use the official RTTM references.

## Reference numbers on the development machine

Windows 11, CPU only (no GPU), 16 GB RAM, default settings (`asr_model = small`, int8), measured on the
bundled fixtures. Benchmark means include first-call model loading, so they overstate steady-state cost.

| Fixture | Metric | Result |
|---|---|---|
| `speech_en.wav` (4.8 s, TTS) | WER | 0.000 |
| `dialogue_2spk.wav` (9.7 s, two TTS voices) | WER | 0.133 |
| `dialogue_2spk.wav` | DER (collar 0.25 s) | 0.218 (all missed speech at turn edges, no speaker confusion) |

Per-stage benchmark, `dialogue_2spk.wav`, 2 repeats, peak RSS 1.5 GB:

| Stage | Mean seconds | RTF |
|---|---|---|
| metadata | 0.16 | 0.02 |
| vad (Silero) | 1.54 | 0.16 |
| features | 0.20 | 0.02 |
| transcription (Whisper small) | 8.02 | 0.82 |
| events (PANNs) | 3.33 | 0.34 |
| diarization (ECAPA) | 1.54 | 0.16 |
| timeline | 0.09 | 0.01 |
| summary (incl. MiniLM load) | 5.43 | 0.56 |

Expect transcription near real time on a modern laptop CPU with `small`; `base`/`tiny` are 2-4x faster, and a
CUDA GPU is an order of magnitude faster.

## Limitations

- The bundled speech fixtures are synthetic (Windows SAPI TTS) and very short; they verify the pipeline, not
  real-world accuracy. The Spanish clip's reference transcript is approximate (machine-made).
- Fake-profile tests (the default CI suite) check plumbing, contracts and edge cases, not model quality.
- Event P/R depends on mapping your labels to AudioSet names; unmapped labels count as misses.
- Confidence values are model probabilities, not calibrated likelihoods; no calibration study has been done.
- DER excludes overlap scoring on the default ECAPA backend (it does not detect overlap).
