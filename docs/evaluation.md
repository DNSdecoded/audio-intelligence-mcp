# Evaluation and benchmarks

**Results: [docs/eval/RESULTS.md](eval/RESULTS.md)** (generated from the raw per-file JSON in
[docs/eval/results/](eval/results/)). This page explains what was measured, on which data, how to reproduce it,
and what the numbers do and do not show.

## Metrics

| Metric | Definition | Script |
|---|---|---|
| WER | Word-level Levenshtein distance / reference words; lowercased, punctuation and symbols stripped by Unicode category (combining marks kept); aggregate weighted by reference words | `scripts/evaluate.py wer` |
| CER | Same normalization, characters with spaces removed (for unsegmented scripts such as Mandarin) | `scripts/evaluate.py wer --unit char` |
| Language detection | Share of files whose detected language equals `--language` | `scripts/evaluate.py wer --language xx` |
| Event P / R / F1, top-1 | Greedy one-to-one matching of same-label events (case-insensitive; a reference `A\|B` accepts either) with IoU ≥ `--iou`; top-1 = highest-confidence event's label is a reference label | `scripts/evaluate.py events` |
| DER | Frame-based (10 ms) missed + false alarm + confusion over reference speech, optimal speaker mapping, collar `--collar` | `scripts/evaluate.py der` |
| Quality P / R | Per issue code, over files labelled with the injected defects; exact match = detected set equals expected set | `scripts/evaluate.py quality` |
| Latency / RTF / memory | Per-stage wall time, real-time factor (seconds / audio seconds), peak RSS sampled every 50 ms | `scripts/benchmark.py` |

## Datasets and why

| Set | Source (license) | What it checks |
|---|---|---|
| LibriSpeech test-clean / test-other, first 2 utterances per speaker (80 / 66) | [openslr.org/12](https://www.openslr.org/12) (CC BY 4.0) | The standard ASR benchmark, with published Whisper-small numbers (3.4 % / 7.6 % WER) to check the pipeline adds no loss |
| FLEURS test, first 25 sentences in en, es, fr, de, ar, hi, te, cmn | [google/fleurs](https://huggingface.co/datasets/google/fleurs) (CC BY 4.0) | Multilingual transcription and language detection on parallel sentences |
| AMI test meetings EN2002a, ES2004a, IS1009a, TS3003a (Mix-Headset), pyannote "only_words" RTTM | [AMI corpus](https://groups.inf.ed.ac.uk/ami/corpus/) (CC BY 4.0), [AMI-diarization-setup](https://github.com/pyannote/AMI-diarization-setup) | Diarization on real meetings with crosstalk and overlap |
| ESC-50 fold 1, 4 clips × 50 classes | [ESC-50](https://github.com/karolpiczak/ESC-50) (CC BY-NC 3.0; clips are not redistributed here) | Sound-event labels for the classes the spec names (alarms, doors, animals, machinery…); disjoint from AudioSet, which PANNs was trained on |
| LibriSpeech + ESC-50 ambient noise at 20/10/5/0 dB SNR (30 utt.) | derived | WER as noise increases |
| LibriSpeech transcoded to 12 containers/codecs/rates (3 utt.) | derived | The "any ffmpeg format" claim |
| LibriSpeech with 8 injected defects + clean (90 files) | derived | Quality diagnosis has exact ground truth per issue (no public dataset labels defect types) |
| Simulated 2–3 speaker LibriSpeech dialogues (12) | derived | Secondary diarization check without overlap |

## Reproduce

```bash
# 1. download: LibriSpeech test-clean (+ test-other), ESC-50 master zip, FLEURS <lang>/test.tsv + audio/test.tar.gz,
#    AMI <meeting>.Mix-Headset.wav + pyannote only_words RTTM (same stem)
# 2. build the sets (deterministic: sorted selection, fixed seeds)
uv run python scripts/prepare_eval_data.py --librispeech RAW/LibriSpeech/test-clean --esc50 RAW/ESC-50-master \
    --librispeech-other RAW/LibriSpeech/test-other --fleurs RAW/fleurs --ami RAW/ami --out SETS
# 3. score (the audio must be inside allowed_roots)
export AUDIO_MCP_ALLOWED_ROOTS='["SETS"]'
uv run python scripts/evaluate.py wer --ref-dir SETS/librispeech --audio-dir SETS/librispeech --language en --out docs/eval/results/wer_librispeech_clean.json
uv run python scripts/evaluate.py wer --unit char --ref-dir SETS/fleurs/cmn_hans_cn --audio-dir SETS/fleurs/cmn_hans_cn --language zh --out ...
uv run python scripts/evaluate.py events --labels SETS/esc50/labels_strict.csv --audio-dir SETS/esc50/audio --iou 0 --out ...
uv run python scripts/evaluate.py der --rttm-dir SETS/ami --audio-dir SETS/ami --collar 0 --out ...
uv run python scripts/evaluate.py quality --labels SETS/quality/labels.csv --audio-dir SETS/quality --out ...
uv run python scripts/benchmark.py --file SETS/long/long_10min.flac --tasks transcription sound_event_detection speaker_diarization --repeat 1
# 4. render the tables
uv run python scripts/eval_report.py
```

`--profile fake` runs the same plumbing with deterministic stand-in backends (no downloads; scores are meaningless).

## Method: tuned on dev, reported on eval

The first run exposed weak quality checks, Whisper repetition loops, missed speech in meetings and a diarization
path that ran full transcription. Every fix was tuned on **development data disjoint from the reported sets**
(10 other LibriSpeech speakers with their own seed for quality; AMI dev meetings ES2011a and TS3004a for the VAD),
then the unchanged evaluation sets were re-run. Both runs are kept (`results/before/` and `results/`). Details and
every tuned value: [the fix plan](superpowers/plans/2026-10-19-eval-fixes.md).

## Findings

- **Transcription matches the model's published accuracy.** LibriSpeech WER is in line with Whisper-small's
  published 3.4 % / 7.6 % (our normalizer is simpler than Whisper's, which adds a little). All 12 formats decode;
  their WER differences are single words.
- **Noise:** accuracy degrades smoothly down to 5 dB SNR (4.3 → 8.4 % WER) and is 14.5 % at 0 dB; at 0 dB the
  first run was 23.6 %, mostly repetition loops that the decoding fix removed. Language detection holds.
- **Run-to-run:** decoding changes moved some rows by ±1–1.5 points in either direction (German 11.5 → 12.9 %,
  French 14.6 → 13.7 %); with 25 utterances per language that is within noise.
- **Multilingual:** high-resource languages are usable; Arabic, Hindi and especially Telugu are not with
  `asr_model = "small"`. The fixes removed the repetition loops (2 → 0 looping files in both Hindi and Telugu), but
  Whisper-small often writes Telugu in Devanagari and Hindi in Urdu script. Use a larger `asr_model` and pass
  `language` for these.
- **Diarization:** on AMI most error is missed speech, much of it overlapped speech the default ECAPA backend
  cannot label (overlap is 5–25 % of each meeting). Speaker confusion is low except in the 4-speaker EN2002a.
  For reference, pyannote 3.1 reports about 18.8 % DER on AMI; install the `pyannote` extra for overlap-aware
  diarization. Diarization-only requests now skip transcription (AMI: 83 → 42 min for the 4 meetings).
- **Sound events:** clip recall is reasonable zero-shot; precision is low by construction because PANNs emits
  several hierarchical AudioSet labels per sound (e.g. Dog, Animal, Domestic animals) and only one can match.
- **Quality diagnosis** went from 12 % to 88 % exact issue sets. Remaining errors: HIGH_NOISE on one recording that
  is noisy to begin with, and DC_OFFSET on clipped files, where asymmetric clipping really shifts the mean.

## Limitations

- Subsets are small (25–80 utterances per set, 4 meetings): treat differences under ~2 points as noise.
- One CPU machine (Windows 11, 4 cores, 16 GB RAM); speed numbers will differ elsewhere.
- Derived sets (noise, defects, formats, simulated dialogues) are synthetic by design; they measure a capability
  against exact ground truth, not real-world prevalence.
- Our text normalizer does not expand numbers or spellings, so WER includes "100" vs "one hundred" mismatches.
- Confidence values are model probabilities, not calibrated likelihoods; no calibration study has been done.
- `ask_audio` and `summarize_audio` have no public benchmark that fits their evidence-and-timestamp design; they are
  covered by tests on synthetic fixtures only.
- The bundled test fixtures are synthetic (Windows SAPI TTS) and very short; they verify the pipeline, not
  real-world accuracy.
