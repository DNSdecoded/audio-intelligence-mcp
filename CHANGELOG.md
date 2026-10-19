# Changelog

## 0.1.1 — 2026-10-19

Bug-fix release. Upgrading is recommended, especially on Windows and for anyone who sets `temp_dir`.

### Fixed
- **Data loss:** a blank `temp_dir` (e.g. `temp_dir = ""`, or `AUDIO_MCP_TEMP_DIR=""`) resolved to the working
  directory, which startup cleanup then emptied. Blank directory settings now mean "use the default", and
  startup cleanup only removes temp directories left by dead server instances (`proc_<pid>_<tag>`), never other files.
- **Privacy:** with `redact_outputs` / `redact: true`, timeline speech nodes kept the transcript in `label`, and
  numbers spoken across several words ("555 123 4567") were not redacted in word lists. Both are now redacted.
- **Windows:** concurrent requests for the same audio could fail with `PermissionError` while another thread
  replaced `meta.json` or a cached result.
- An LLM reply that was valid JSON with the wrong structure made `summarize_audio` / `ask_audio` /
  `compare_audio` fail instead of falling back to the rule-based result.
- `get_audio_timeline(diarize=True)` could return a cached timeline without speaker labels.
- The hourly sweep deleted cached results of audio that was still in use.
- `ask_audio`: passive questions ("When is the deadline mentioned?") were not recognised as time-of-mention
  questions; "anything odd?" never triggered the unusual-sounds intent (stray control characters in a regex).
- Async `analyze_audio` with `tasks="auto"` reported 0.99 progress at the start of the job.
- Stream pushes and `compare_transcription` ran models outside the `max_concurrent_jobs` limit.

### Performance
- Sound-event tagging and audio embeddings share one PANNs pass instead of two.
- `compare_audio` with a baseline no longer runs feature, event and embedding stages it does not use.

## 0.1.0 — 2026-10-19

First release: 13 MCP tools, local models, evaluation on public datasets (see `docs/eval/RESULTS.md`).
