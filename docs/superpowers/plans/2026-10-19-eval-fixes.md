# Evaluation fixes — plan

Source: first full evaluation run on 2026-10-19 (`docs/eval/results/*.json`). Each fix below is tuned on a
**development** set that is disjoint from the evaluation set; the evaluation sets are then re-run unchanged and
both before/after numbers are published.

| Dev set (tuning only) | Eval set (reported) |
|---|---|
| `quality_dev/`: 10 other LibriSpeech speakers (`[2::4]`), seed 99 | `quality/` (unchanged) |
| AMI dev `ES2011a`, `TS3004a` | AMI test `EN2002a`, `ES2004a`, `IS1009a`, `TS3003a` |
| none (no parameters tuned; library defaults) | FLEURS hi/te/ar, LibriSpeech clean (regression) |

## Fixes

| # | Finding (eval) | Fix | Files |
|---|---|---|---|
| F1 | CLIPPING recall 0/10: only samples at ±0.999 count | Also count flat-topped runs (≥ 3 equal samples within 1e-4 of the file's own peak, peak ≥ 0.25) | `dsp/features.py` |
| F2 | ECHO 51 FP / 90: onset-envelope autocorrelation fires on syllable rhythm (~200 ms) | Real cepstrum of the DC-removed signal; robust z-score of the 30–300 ms quefrency peak | `dsp/quality.py` |
| F3 | HIGH_NOISE 42 FP / 90: noise floor taken from VAD-leaky gaps or the 10th-percentile frame of continuous speech; DC counted as noise | Remove DC first; noise = median power of non-speech frames ≥ 50 ms from speech; < 0.5 s of non-speech → no estimate (`snr_method: "insufficient_nonspeech"`) instead of a guess | `dsp/features.py` |
| F4 | BANDWIDTH_LIMITED 25 FP / 90: 99 % roll-off pulled down by hum/DC and dark voices | Ratio of 4 kHz–Nyquist energy to 300 Hz–4 kHz energy; flag below the dev-tuned dB threshold | `dsp/quality.py` |
| F5 | LOW_LEVEL fires when long silence dilutes RMS; quiet speech counts as SILENCE | Active level = RMS over speech regions (or frames above −60 dBFS); silence frames relative to active level (−30 dB, floor −60 dBFS) | `dsp/features.py`, `dsp/quality.py` |
| F6 | Whisper repetition loops (Hindi/Telugu WER > 100 %, 20-min runs) | Use faster-whisper's temperature fallback (0.0…1.0) instead of a fixed 0.0; `condition_on_previous_text=False` | `backends/whisper.py` |
| F7 | `speaker_diarization` runs full ASR (speaker_assign needs words): AMI ~1.15× real time | `analyze_audio` adds `speaker_assign` only when transcription is also requested | `tools/handlers.py` |
| F8 | AMI missed speech 7–28 % beyond overlap: VAD/segmentation drops speech | Expose Silero parameters (`vad_threshold`, `vad_min_silence_ms`, `vad_speech_pad_ms`); tune on AMI dev; confirm LibriSpeech WER unchanged | `backends/silero.py`, `config.py` |

Not fixed (documented): overlap (needs pyannote), Whisper-small script confusion (Hindi→Urdu, Telugu→Devanagari),
ESC-50 precision (multi-label AudioSet output), diarization RTF ~0.57.

## Dev-set tuning (done 2026-10-19)

- `quality_dev`: thresholds ECHO z > 20 (clean ≤ 8.0, echo ≥ 80.8), HF ratio < −27 dB (clean ≥ −24.2,
  low-pass ≤ −29.1), active level < −42 dBFS (clean ≥ −39.1, low-level ≤ −48.2), silence > 0.6 (clean ≤ 0.44,
  silence ≥ 0.75), SNR fallback "speech_dips" = mean / 5th-percentile frame power (clean ≥ 26.9 dB, 5 dB noise
  ≈ 6.4 dB); HIGH_NOISE suppressed when HUM is flagged. Dev exact-match 12 % → 95.6 % (remaining: 4 clipped
  files also show a real DC offset from asymmetric clipping).
- AMI dev (ES2011a, TS3004a) speech-detection error (missed + false alarm over reference speech):
  Silero defaults 0.5/100 ms/30 ms 20.9 %; chosen 0.35/300 ms/150 ms 14.6 % (missed 8.9 %, FA 5.7 %);
  0.2/300/150 was 13.9 % but FA 8.1 % (more non-speech sent to Whisper).
- Found while tuning: the result cache is keyed by stage config, not code, so upgraded code reused stale results.
  Added version keys (`dsp: 2`, quality `rules: 2`, ASR `decode: 2`, VAD params) to the cache config.

## Verification

1. Unit tests for F1–F5 thresholds on synthetic signals; fast suite green.
2. Dev sets: tune, record chosen values and dev scores here.
3. Re-run eval sets: quality, AMI (both collars), FLEURS hi/te/ar, LibriSpeech clean + noise (regression), simulated DER.
4. Publish `docs/eval/results/` (before → `before/`), update `docs/evaluation.md`.

## Outcome (eval sets, re-run 2026-10-19)

Quality exact match 12.2 → 87.8 %; AMI DER 42.2 → 38.2 % (collar 0.25: 36.2 → 30.4 %) and 83 → 42 min;
simulated DER 17.5 → 13.3 %; WER at 0 dB SNR 23.6 → 14.5 %; FLEURS Hindi 68.0 → 58.8 %; LibriSpeech test-clean
unchanged (4.1 %), test-other 6.7 → 6.1 %. Not improved: Telugu (wrong script), Arabic, ESC-50 (not touched).
Full tables: `docs/eval/RESULTS.md`.
