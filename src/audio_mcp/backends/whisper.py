import math
import threading

import numpy as np

from audio_mcp.backends.base import ASRResult
from audio_mcp.envelope import AudioMCPError, ErrorCode, ModelInfo
from audio_mcp.types import Segment, Word


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
        if per_chunk_language:
            raise NotImplementedError("per-chunk language detection arrives in Task 22")
        self.load()
        x = np.ascontiguousarray(mono16k, dtype=np.float32)
        duration = len(x) / 16000
        gen, info = self._model.transcribe(
            x,
            language=language,
            task=task,
            word_timestamps=word_timestamps,
            vad_filter=False,
            beam_size=decode_kwargs.get("beam_size", 5),
            temperature=decode_kwargs.get("temperature", 0.0),
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
