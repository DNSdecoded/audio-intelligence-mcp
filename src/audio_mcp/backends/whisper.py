import math
import threading

import numpy as np

from audio_mcp.backends.base import ASRResult, group_chunks
from audio_mcp.envelope import AudioMCPError, ErrorCode, ModelInfo
from audio_mcp.types import Segment, Word

_TEMPERATURES = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)  # faster-whisper's own fallback schedule


def _r(x: float, duration: float) -> float:
    return round(min(max(float(x), 0.0), duration), 3)


class FasterWhisperASR:
    def __init__(self, settings):
        self.settings = settings
        self.info = ModelInfo(stage="transcription", name="faster-whisper", version=settings.asr_model)
        self._model = None
        self._lock = threading.Lock()

    def available(self):
        from audio_mcp.backends.base import _importable

        if not _importable("faster_whisper"):
            return False, "faster_whisper not installed (uv sync --extra models)"
        return True, None

    def load(self):
        with self._lock:
            if self._model is not None:
                return
            s = self.settings
            try:
                import ctranslate2
                from faster_whisper import WhisperModel

                cuda = s.device == "cuda" or (s.device == "auto" and ctranslate2.get_cuda_device_count() > 0)
                self._model = WhisperModel(
                    s.asr_model,
                    device="cuda" if cuda else "cpu",
                    compute_type=s.asr_compute_type,
                    download_root=str(s.resolved_models_dir / "whisper"),
                )
            except Exception as e:
                raise AudioMCPError(ErrorCode.MODEL_UNAVAILABLE, f"faster-whisper failed to load: {e}") from e

    def transcribe(
        self,
        mono16k,
        *,
        language=None,
        word_timestamps=True,
        task="transcribe",
        per_chunk_language=False,
        chunks=None,
        **decode_kwargs,
    ) -> ASRResult:
        self.load()
        x = np.ascontiguousarray(mono16k, dtype=np.float32)
        duration = len(x) / 16000
        if per_chunk_language and chunks:
            return self._per_chunk(x, duration, chunks, word_timestamps, task, decode_kwargs)
        gen, info = self._model.transcribe(
            x,
            language=language,
            task=task,
            word_timestamps=word_timestamps,
            vad_filter=False,
            beam_size=decode_kwargs.get("beam_size", 5),
            # fallback temperatures re-decode segments that fail the compression/log-prob checks (repetition loops);
            # not conditioning on earlier text stops one looping segment from seeding the next
            temperature=decode_kwargs.get("temperature", _TEMPERATURES),
            condition_on_previous_text=False,
        )
        segments, dropped = [], 0
        for s in gen:
            if s.no_speech_prob > 0.8 and s.avg_logprob < -1.0:
                dropped += 1
                continue
            words = [
                Word(
                    start=_r(w.start, duration),
                    end=_r(w.end, duration),
                    text=w.word.strip(),
                    confidence=min(1.0, max(0.0, float(w.probability))),
                    confidence_kind="model_probability",
                )
                for w in (s.words or [])
            ]
            segments.append(
                Segment(
                    id=f"seg_{len(segments) + 1:04d}",
                    start=_r(s.start, duration),
                    end=_r(s.end, duration),
                    text=s.text.strip(),
                    language=info.language,
                    words=words,
                    confidence=min(1.0, math.exp(s.avg_logprob)),
                    confidence_kind="model_probability",
                )
            )
        lang, prob = info.language, float(info.language_probability)
        return ASRResult(
            segments=segments,
            language=lang,
            language_probability=prob,
            languages=[{"language": lang, "start": 0.0, "end": round(duration, 3), "probability": prob}],
            dropped_low_speech=dropped,
        )


    def _detect(self, piece):
        if hasattr(self._model, "detect_language"):  # faster-whisper >= 1.1
            lang, prob, _ = self._model.detect_language(piece)
            return lang, float(prob)
        _, info = self._model.transcribe(piece[: 30 * 16000], beam_size=1)
        return info.language, float(info.language_probability)

    def _per_chunk(self, x, duration, chunks, word_timestamps, task, decode_kwargs) -> ASRResult:
        segments, runs, dropped, seconds = [], [], 0, {}
        for c in group_chunks(chunks):
            off = int(c.start * 16000)
            piece = x[off:int(c.end * 16000)]
            if piece.size < 1600:  # under 0.1 s: nothing to transcribe
                continue
            lang, prob = self._detect(piece)
            r = self.transcribe(piece, language=lang, word_timestamps=word_timestamps, task=task, **decode_kwargs)
            dropped += r.dropped_low_speech
            for s in r.segments:
                words = [w.model_copy(update={"start": _r(w.start + c.start, duration), "end": _r(w.end + c.start, duration)})
                         for w in s.words]
                segments.append(s.model_copy(update={
                    "id": f"seg_{len(segments) + 1:04d}", "start": _r(s.start + c.start, duration),
                    "end": _r(s.end + c.start, duration), "language": lang, "words": words}))
            seconds[lang] = seconds.get(lang, 0.0) + c.end - c.start
            if runs and runs[-1]["language"] == lang:
                runs[-1]["end"] = round(c.end, 3)
            else:
                runs.append({"language": lang, "start": round(c.start, 3), "end": round(c.end, 3),
                             "probability": round(prob, 3)})
        top = max(seconds, key=seconds.get) if seconds else None
        top_prob = next((r["probability"] for r in runs if r["language"] == top), None)
        return ASRResult(segments=segments, language=top, language_probability=top_prob, languages=runs,
                         dropped_low_speech=dropped)
