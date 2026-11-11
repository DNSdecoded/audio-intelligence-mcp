---
name: audio-intelligence
description: Use for any request to analyze, transcribe, summarize, search or compare audio files (.wav .mp3 .m4a .flac .ogg .opus .webm), meeting or call recordings, podcasts, voice notes, machine or environmental sounds; also "who said", "when did they mention", subtitles (srt/vtt), speaker labels, noise or quality checks. Covers which Audio Intelligence MCP tool to call, in what order, and how to reuse results. Not for generating speech or music.
---

# Audio Intelligence MCP

The tools are exposed by the `audio-intelligence` MCP server (your client may prefix their names, for example `mcp__<name>__transcribe_audio`).

## Sources

Every `source` is one of:

- `{"type": "file_path", "path": "/abs/path.wav"}`. The folder must be inside the server's `allowed_roots`; `PATH_NOT_ALLOWED` means the server config needs that folder, not a different spelling.
- `{"type": "base64", "data": "...", "filename_hint": "a.mp3"}`. Capped at 50 MB by default.
- `{"type": "url", "url": "https://..."}`. Off unless the server enables `allow_url_fetch`.
- `{"type": "audio_id", "audio_id": "aud_0123456789abcdef"}`. Reuses audio already ingested; every response carries its `audio_id`.

## Choose the tool

| The user wants | Call |
|---|---|
| Overview of an unknown recording | `analyze_audio` with `"tasks": "auto"` and a `goal`. `budget` is `fast`, `balanced` or `thorough` |
| Transcript, speakers, English translation | `transcribe_audio` (`diarize`, `translate_to_en`, `multilingual` for code-switching, `language` to force one) |
| Names or jargon transcribed wrongly | `transcribe_audio` with `hotwords` |
| Alarms, impacts, animals, music | `detect_audio_events`. `categories` are `alarm`, `impact`, `footsteps`, `door`, `animal`, `machinery`, `music`, `environment`; `labels` are free AudioSet names |
| Is the recording usable (clipping, noise, hum, echo) | `inspect_audio_quality` (`apply_preprocessing` to also get a cleaned variant) |
| Raw measurements | `extract_audio_features` (`feature_groups`: `level`, `silence`, `spectral`, `pitch`, `rhythm`, `bands`) |
| What happened when, who spoke when | `get_audio_timeline` (`include` node types such as `speaker_turn`, `sound_event`, `silence`, `overlap`) |
| A question about content or timing | `ask_audio`. Check `support`: `direct`, `inferred` or `insufficient` |
| Summary, key moments | `summarize_audio` (`length`: `short`, `medium`, `long`; `focus` for a topic) |
| Different from normal, or A versus B | `compare_audio`: `sources` plus `baseline` for per-second anomalies, or two `sources` for a pairwise diff |
| Find recordings across many files | `search_audio_library`: `action` is `index`, `search`, `list` or `remove`; search by `query` text or by example clip with `query_audio` |
| Live capture in chunks | `analyze_audio_stream`: `start`, repeated `push`, then `finish` (or `status`, `cancel`) |
| Subtitles or a report file | `export_audio_analysis`: needs an `audio_id` from an earlier call; formats `json`, `markdown`, `srt`, `vtt`, `csv`, `rttm` |
| Progress of long audio | `get_job` (`action`: `status` or `cancel`) |

## Worked sequences

Meeting: analyze once, then ask and export from the returned id.

```
analyze_audio {"source": {"type": "file_path", "path": "/audio/standup.m4a"}, "goal": "meeting"}
ask_audio {"source": {"type": "audio_id", "audio_id": "aud_0123456789abcdef"}, "question": "When does someone mention the deadline?"}
export_audio_analysis {"audio_id": "aud_0123456789abcdef", "format": "srt"}
```

Machine baseline: compare today's recording against known-good ones.

```
compare_audio {"sources": [{"type": "file_path", "path": "/audio/today.wav"}], "baseline": [{"type": "file_path", "path": "/audio/ok_1.wav"}, {"type": "file_path", "path": "/audio/ok_2.wav"}]}
```

Environmental recording: events first, then the order of things.

```
detect_audio_events {"source": {"type": "file_path", "path": "/audio/yard.wav"}, "categories": ["animal", "machinery"]}
ask_audio {"source": {"type": "file_path", "path": "/audio/yard.wav"}, "question": "What happens immediately before the alarm sounds?"}
```

Library: index once, then search by text or by clip.

```
search_audio_library {"action": "index", "paths": ["/audio/calls"]}
search_audio_library {"action": "search", "query": "refund request", "filters": {"language": "en"}}
```

Long audio runs as a job. When a response has `status: "processing"` and a `job_id`:

```
get_job {"job_id": "job_0123456789ab"}
```

## Reading results

- Success envelopes carry `status`, `audio_id`, `result` (`analysis` for `analyze_audio`), `warnings`, `errors` and `processing`. Bad arguments come back as `{"status": "error", "error": {"code", "message", "retryable"}}` instead.
- Read `warnings` and `processing.stages_skipped` first. A skipped stage means that model was unavailable, not that nothing was found.
- Cached stages are reused only when the options match, and stored audio expires after `retention_ttl_hours` (24 by default). If `AUDIO_NOT_FOUND` comes back for an old `audio_id`, send the file again.
- `LOW_RESOURCE_LANGUAGE` (Arabic, Hindi, Telugu on a small model): tell the user the server needs `asr_model = "large-v3"` and pass `language`. `OVERLAP_UNSUPPORTED`: the default diarizer cannot see talk-over; say so instead of implying there was none.
- `use_llm` on `ask_audio` and `summarize_audio`: leave it unset to follow the server config. `true` fails with `LLM_DISABLED` when no LLM is configured; you cannot enable one from a call. `processing.data_sent_off_machine` reports when text left the machine.
- Confidence values are uncalibrated. Integrity output is signal observations, never proof of authenticity. The server does not infer emotion.
