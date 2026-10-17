# Fixture sources

- `speech_en.wav`, `dialogue_2spk.wav`: synthesized locally with Windows SAPI voices (Microsoft David / Zira Desktop) via `tests/fixtures/make_speech.py`. References in `.txt`, `.rttm`.
- `speech_es.ogg`: 8 s excerpt of "Las aventuras de Pinoquio" by Carlo Collodi (tr. Antonio José Restrepo), LibriVox recording read in Spanish by Victor Villarraza. License: LibriVox public domain (LibriVox recordings are dedicated to the public domain).
  - Source page: https://archive.org/details/las_aventuras_de_pinoquio_2305_librivox
  - File URL: https://archive.org/download/las_aventuras_de_pinoquio_2305_librivox/aventurasdepinoquio_01_collodi_64kb.mp3
  - Cut: 60.0 s to 68.0 s of chapter 1; ffmpeg, libvorbis, 16 kHz mono.
  - `speech_es.txt` transcript is approximate: machine-assisted (faster-whisper small, Spanish), not verified word-for-word by a human; the clip starts mid-sentence.
- AudioSet label CSV (CC-BY 4.0): added in Task 8.
