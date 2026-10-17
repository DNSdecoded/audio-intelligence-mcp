"""Deterministic fake backends for tests and the `fake` profile."""
import csv
import zlib
from importlib import resources

import numpy as np

from audio_mcp.backends.base import ASRResult, Backends, TagResult, group_chunks
from audio_mcp.backends.energy_vad import EnergyVAD
from audio_mcp.envelope import AudioMCPError, ErrorCode, ModelInfo
from audio_mcp.types import Segment, Word

_TEXTS = ["we need to finish the report by friday", "the alarm test is scheduled for monday morning"]


def _info(stage):
    return ModelInfo(stage=stage, name=f"fake-{stage}", version="1")


class _Fake:
    def available(self):
        return True, None

    def load(self):
        pass


def _labels() -> list[str]:
    path = resources.files("audio_mcp.backends").joinpath("audioset_labels.csv")
    with path.open(encoding="utf-8", newline="") as f:
        return [row["display_name"] for row in csv.DictReader(f)]


class FakeASR(_Fake):
    info = _info("asr")

    def __init__(self):
        self.vad = EnergyVAD()

    def transcribe(self, mono16k, *, language=None, word_timestamps=True, task="transcribe",
                   per_chunk_language=False, chunks=None, **decode_kwargs):
        regions = chunks if chunks is not None else self.vad.speech_regions(mono16k)
        if per_chunk_language:
            regions = group_chunks(regions)
        segs, langs = [], []
        for i, r in enumerate(regions):
            lang = ["en", "es"][i % 2] if per_chunk_language else (language or "en")
            text = _TEXTS[i % len(_TEXTS)]
            if task == "translate":
                text = "[en] " + text
            toks = text.split()
            step = (r.end - r.start) / len(toks)
            words = [
                Word(start=r.start + j * step, end=r.start + (j + 1) * step, text=t,
                     confidence=0.8, confidence_kind="model_probability")
                for j, t in enumerate(toks)
            ] if word_timestamps else []
            segs.append(Segment(id=f"seg_{i + 1:04d}", start=r.start, end=r.end, text=text, language=lang,
                                words=words, confidence=0.85, confidence_kind="model_probability"))
            if langs and langs[-1]["language"] == lang:
                langs[-1]["end"] = r.end
            else:
                langs.append({"language": lang, "start": r.start, "end": r.end, "probability": 0.9})
        top = segs[0].language if segs else (language or "en")
        return ASRResult(segments=segs, language=top, language_probability=0.9,
                         languages=langs if per_chunk_language else
                         [{"language": top, "start": 0.0, "end": len(mono16k) / 16000, "probability": 0.9}])


class FakeTagger(_Fake):
    info = _info("tagger")
    _WIN, _HOP, _BANDS = 1.0, 0.5, 64

    def __init__(self):
        self.labels = _labels()
        self.vad = EnergyVAD()

    def _embed_window(self, w):
        p = np.abs(np.fft.rfft(w)) ** 2
        v = np.log1p(np.array([b.sum() for b in np.array_split(p, self._BANDS)]))
        return np.pad(v, (0, 2048 - self._BANDS)).astype(np.float32)

    def tag_windows(self, mono16k):
        x = np.asarray(mono16k, dtype=np.float32)
        n = 16000
        starts = np.arange(0, max(len(x) - n, 0) + 1, n // 2)  # short file: one zero-padded window
        speech = self.vad.speech_regions(x)
        idx = {l: i for i, l in enumerate(self.labels)}
        scores = np.zeros((len(starts), len(self.labels)), dtype=np.float32)
        emb = np.zeros((len(starts), 2048), dtype=np.float32)
        freqs = np.fft.rfftfreq(n, 1 / 16000)
        for k, s in enumerate(starts):
            seg = x[s:s + n]
            w = np.pad(seg, (0, n - len(seg)))
            t0, t1 = s / 16000, s / 16000 + self._WIN
            p = np.abs(np.fft.rfft(w)) ** 2
            beep = p.sum() > 0 and p[(freqs >= 900) & (freqs <= 1100)].sum() / p.sum() > 0.5
            rms_db = 20 * np.log10(max(float(np.sqrt((w ** 2).mean())), 1e-10))
            if beep:
                scores[k, idx["Beep, bleep"]] = 0.8
            elif any(r.start < t1 and r.end > t0 for r in speech):
                scores[k, idx["Speech"]] = 0.9
            elif rms_db < -50:
                scores[k, idx["Silence"]] = 0.7
            else:
                scores[k, idx["Noise"]] = 0.5
            emb[k] = self._embed_window(w)
        return TagResult(labels=self.labels, scores=scores, window_starts=starts / 16000,
                         window_seconds=self._WIN, embeddings=emb)

    def embed(self, mono16k):
        return self._embed_window(np.asarray(mono16k, dtype=np.float32)[:16000 * 10])


class FakeSpeaker(_Fake):
    info = _info("speaker")

    def embed(self, windows):
        out = []
        for w in windows:
            p = np.abs(np.fft.rfft(np.asarray(w, dtype=np.float64))) ** 2
            v = np.array([b.mean() for b in np.array_split(np.log1p(p), 64)])
            out.append(v / (np.linalg.norm(v) or 1.0))
        return np.array(out, dtype=np.float32).reshape(len(windows), 64)


class FakeText(_Fake):
    info = _info("text")

    def embed(self, texts):
        out = np.zeros((len(texts), 256), dtype=np.float32)
        for i, t in enumerate(texts):
            for tok in t.lower().split():
                out[i, zlib.crc32(tok.encode()) % 256] += 1
        n = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.where(n == 0, 1, n)


class NoLLM:
    info = ModelInfo(stage="llm", name="none", version="0")
    is_remote = False

    def available(self):
        return False, "llm_provider is none"

    def load(self):
        raise AudioMCPError(ErrorCode.LLM_DISABLED, "llm_provider is none")

    def complete(self, system, prompt):
        raise AudioMCPError(ErrorCode.LLM_DISABLED, "llm_provider is none")


def fake_backends() -> Backends:
    vad = EnergyVAD()
    return Backends(vad=vad, vad_fallback=vad, asr=FakeASR(), tagger=FakeTagger(), speaker=FakeSpeaker(),
                    diarizer=None, text=FakeText(), llm=NoLLM())
