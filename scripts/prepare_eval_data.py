"""Build the evaluation sets in docs/evaluation.md from LibriSpeech test-clean and ESC-50 (download both yourself).

Usage:
  uv run python scripts/prepare_eval_data.py --librispeech RAW/LibriSpeech/test-clean --esc50 RAW/ESC-50-master --out OUT       [--librispeech-other RAW/LibriSpeech/test-other] [--fleurs RAW/fleurs] [--ami RAW/ami]

Writes, deterministically (fixed seed, sorted selection):
  OUT/librispeech/   80 utterances (first 2 per speaker) + stem.txt references             -> evaluate.py wer
  OUT/librispeech_other/  same selection from test-other                                   -> evaluate.py wer
  OUT/fleurs/<lang>/ 25 test utterances per language (FLEURS normalized transcription)     -> evaluate.py wer
  OUT/ami/           AMI test meetings (Mix-Headset) + pyannote "only_words" RTTM          -> evaluate.py der
  OUT/noise/snr_N/   30 utterances mixed with ESC-50 ambient noise at 20/10/5/0 dB SNR      -> evaluate.py wer
  OUT/formats/       3 utterances transcoded to 12 container/codec/rate variants           -> evaluate.py wer
  OUT/esc50/         200 clips (fold 1, first 4 per class) + labels_strict/lenient.csv      -> evaluate.py events
  OUT/diarization/   12 simulated 2-3 speaker conversations + RTTM                          -> evaluate.py der
  OUT/quality/       10 utterances x 9 injected defects + labels.csv                       -> evaluate.py quality
  OUT/quality_dev/   same defects on 10 other speakers, own seed (tuning only)              -> evaluate.py quality
  OUT/long/          one ~10 min recording of concatenated utterances                      -> benchmark.py
"""
import argparse
import csv
import subprocess
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfilt

from audio_mcp.backends.panns import _labels

SR = 16000
RNG = np.random.default_rng(1234)

# ESC-50 class -> AudioSet display names. The first name is the strict label; the rest are accepted in the
# lenient set (sound-specific child or parent classes PANNs commonly outputs for that sound).
ESC50_TO_AUDIOSET = {
    "dog": ["Dog", "Bark", "Bow-wow", "Canidae, dogs, wolves", "Animal", "Domestic animals, pets"],
    "rooster": ["Crowing, cock-a-doodle-doo", "Chicken, rooster", "Fowl", "Bird", "Animal"],
    "pig": ["Pig", "Oink", "Livestock, farm animals, working animals", "Animal"],
    "cow": ["Cattle, bovinae", "Moo", "Livestock, farm animals, working animals", "Animal"],
    "frog": ["Frog", "Croak", "Animal"],
    "cat": ["Cat", "Meow", "Caterwaul", "Purr", "Domestic animals, pets", "Animal"],
    "hen": ["Chicken, rooster", "Cluck", "Fowl", "Bird", "Animal"],
    "insects": ["Insect", "Bee, wasp, etc.", "Fly, housefly", "Mosquito", "Buzz", "Animal"],
    "sheep": ["Sheep", "Bleat", "Goat", "Livestock, farm animals, working animals", "Animal"],
    "crow": ["Crow", "Caw", "Bird", "Bird vocalization, bird call, bird song", "Animal"],
    "rain": ["Rain", "Raindrop", "Rain on surface"],
    "sea_waves": ["Waves, surf", "Ocean", "Water"],
    "crackling_fire": ["Crackle", "Fire"],
    "crickets": ["Cricket", "Insect", "Animal"],
    "chirping_birds": ["Bird vocalization, bird call, bird song", "Chirp, tweet", "Bird", "Animal"],
    "water_drops": ["Drip", "Water", "Liquid", "Raindrop"],
    "wind": ["Wind", "Wind noise (microphone)", "Rustling leaves"],
    "pouring_water": ["Pour", "Water", "Liquid", "Trickle, dribble", "Fill (with liquid)"],
    "toilet_flush": ["Toilet flush", "Water", "Liquid"],
    "thunderstorm": ["Thunderstorm", "Thunder", "Rain"],
    "crying_baby": ["Baby cry, infant cry", "Crying, sobbing"],
    "sneezing": ["Sneeze"],
    "clapping": ["Clapping", "Applause", "Hands"],
    "breathing": ["Breathing", "Gasp", "Pant", "Wheeze", "Snoring"],
    "coughing": ["Cough", "Throat clearing"],
    "footsteps": ["Walk, footsteps", "Run", "Shuffle"],
    "laughing": ["Laughter", "Giggle", "Chuckle, chortle", "Belly laugh", "Snicker", "Baby laughter"],
    "brushing_teeth": ["Toothbrush", "Electric toothbrush", "Scrape", "Rub"],
    "snoring": ["Snoring", "Breathing", "Snort"],
    "drinking_sipping": ["Gargling", "Liquid", "Slosh", "Chewing, mastication"],
    "door_wood_knock": ["Knock", "Door", "Tap", "Wood"],
    "mouse_click": ["Clicking", "Tick"],
    "keyboard_typing": ["Typing", "Computer keyboard", "Typewriter"],
    "door_wood_creaks": ["Creak", "Squeak", "Door"],
    "can_opening": ["Burst, pop", "Hiss", "Liquid"],
    "washing_machine": ["Mechanisms", "Hum", "Vacuum cleaner", "Liquid"],
    "vacuum_cleaner": ["Vacuum cleaner", "Mechanisms"],
    "clock_alarm": ["Alarm clock", "Alarm", "Beep, bleep", "Buzzer", "Ringtone"],
    "clock_tick": ["Tick-tock", "Tick", "Clock"],
    "glass_breaking": ["Shatter", "Glass", "Breaking", "Chink, clink"],
    "helicopter": ["Helicopter", "Aircraft"],
    "chainsaw": ["Chainsaw", "Power tool", "Sawing", "Tools"],
    "siren": ["Siren", "Civil defense siren", "Police car (siren)", "Ambulance (siren)",
              "Fire engine, fire truck (siren)", "Emergency vehicle"],
    "car_horn": ["Vehicle horn, car horn, honking", "Toot", "Air horn, truck horn", "Car", "Vehicle"],
    "engine": ["Engine", "Idling", "Medium engine (mid frequency)", "Heavy engine (low frequency)",
               "Light engine (high frequency)", "Accelerating, revving, vroom", "Vehicle", "Motor vehicle (road)"],
    "train": ["Train", "Rail transport", "Railroad car, train wagon", "Train wheels squealing", "Train horn",
              "Vehicle"],
    "church_bells": ["Church bell", "Bell", "Change ringing (campanology)"],
    "airplane": ["Fixed-wing aircraft, airplane", "Aircraft", "Aircraft engine", "Jet engine",
                 "Propeller, airscrew", "Vehicle"],
    "fireworks": ["Fireworks", "Firecracker", "Explosion", "Burst, pop"],
    "hand_saw": ["Sawing", "Tools", "Wood"],
}
NOISE_CLASSES = ["rain", "wind", "sea_waves", "crackling_fire", "washing_machine", "vacuum_cleaner"]
FORMATS = {  # name: (extension, ffmpeg output args)
    "wav16k": (".wav", ["-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le"]),
    "wav48kstereo": (".wav", ["-ar", "48000", "-ac", "2", "-c:a", "pcm_f32le"]),
    "wav8kmulaw": (".wav", ["-ar", "8000", "-ac", "1", "-c:a", "pcm_mulaw"]),
    "flac": (".flac", ["-c:a", "flac"]),
    "mp3": (".mp3", ["-c:a", "libmp3lame", "-b:a", "64k"]),
    "ogg": (".ogg", ["-c:a", "libvorbis", "-q:a", "3"]),
    "opus": (".opus", ["-c:a", "libopus", "-b:a", "24k"]),
    "m4a": (".m4a", ["-c:a", "aac", "-b:a", "64k"]),
    "aac": (".aac", ["-c:a", "aac", "-b:a", "64k"]),
    "webm": (".webm", ["-c:a", "libopus", "-b:a", "32k"]),
    "aiff": (".aiff", ["-c:a", "pcm_s16be"]),
    "wma": (".wma", ["-c:a", "wmav2", "-b:a", "64k"]),
}


def librispeech(root: Path) -> dict[str, list[tuple[Path, str]]]:
    """speaker -> sorted [(flac, transcript)]"""
    out = {}
    for spk in sorted(root.iterdir(), key=lambda p: int(p.name)):
        utts = []
        for trans in sorted(spk.glob("*/*.trans.txt")):
            for line in trans.read_text(encoding="utf-8").splitlines():
                uid, text = line.split(" ", 1)
                utts.append((trans.parent / f"{uid}.flac", text.lower()))
        out[spk.name] = sorted(utts)
    return out


def load(p: Path) -> np.ndarray:
    x, sr = sf.read(p, dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    return librosa.resample(x, orig_sr=sr, target_sr=SR) if sr != SR else x


def rms(x):
    return float(np.sqrt(np.mean(x ** 2)) + 1e-12)


def write(path: Path, x, sr=SR):
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, np.clip(x, -1.0, 1.0), sr, subtype="PCM_16")


def esc50_meta(root: Path) -> list[dict]:
    with open(root / "meta" / "esc50.csv", encoding="utf-8", newline="") as fh:
        return sorted(csv.DictReader(fh), key=lambda r: r["filename"])


def build_librispeech(spk, out, name="librispeech"):
    d = out / name
    d.mkdir(parents=True, exist_ok=True)
    for utts in spk.values():
        for flac, text in utts[:2]:
            (d / flac.name).write_bytes(flac.read_bytes())
            (d / f"{flac.stem}.txt").write_text(text, encoding="utf-8")


def build_noise(spk, esc_root, meta, out):
    noises = [load(esc_root / "audio" / r["filename"]) for c in NOISE_CLASSES
              for r in [r for r in meta if r["category"] == c][:5]]
    utts = [u[0] for u in spk.values()][:30]
    for snr in (20, 10, 5, 0):
        d = out / "noise" / f"snr_{snr}"
        for k, (flac, text) in enumerate(utts):
            x = load(flac)
            n = np.resize(noises[k % len(noises)], len(x))
            n *= rms(x) / rms(n) / 10 ** (snr / 20)
            write(d / f"{flac.stem}.flac", x + n)
            (d / f"{flac.stem}.txt").write_text(text, encoding="utf-8")


def build_formats(spk, out, ffmpeg):
    d = out / "formats"
    d.mkdir(parents=True, exist_ok=True)
    for flac, text in [u[0] for u in spk.values()][30:33]:
        for name, (ext, args) in FORMATS.items():
            stem = f"{flac.stem}__{name}"
            subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-i", str(flac), *args, str(d / f"{stem}{ext}")],
                           check=True)
            (d / f"{stem}.txt").write_text(text, encoding="utf-8")


def build_esc50(esc_root, meta, out):
    missing = {n for names in ESC50_TO_AUDIOSET.values() for n in names} - set(_labels())
    assert not missing, f"not AudioSet display names: {sorted(missing)}"
    d = out / "esc50"
    (d / "audio").mkdir(parents=True, exist_ok=True)
    rows = []
    for cat, names in ESC50_TO_AUDIOSET.items():
        for r in [r for r in meta if r["category"] == cat and r["fold"] == "1"][:4]:
            (d / "audio" / r["filename"]).write_bytes((esc_root / "audio" / r["filename"]).read_bytes())
            rows.append((r["filename"], names))
    for kind, label in (("strict", lambda n: n[0]), ("lenient", lambda n: "|".join(n))):
        with open(d / f"labels_{kind}.csv", "w", encoding="utf-8", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["file", "start", "end", "label"])
            w.writerows((f, 0.0, 5.0, label(n)) for f, n in rows)


def build_diarization(spk, out):
    d = out / "diarization"
    d.mkdir(parents=True, exist_ok=True)
    speakers = list(spk)
    cursor = {s: 2 for s in speakers}  # skip the 2 utterances used in librispeech/
    for c in range(12):
        chosen = list(RNG.choice(speakers, size=3 if c >= 8 else 2, replace=False))
        audio, rttm, t, prev = [], [], 0.0, None
        for _ in range(10):
            s = RNG.choice([x for x in chosen if x != prev])
            flac = spk[s][cursor[s] % len(spk[s])][0]
            cursor[s] += 1
            x, _ = librosa.effects.trim(load(flac), top_db=30)
            gap = np.zeros(int(RNG.uniform(0.3, 1.0) * SR), dtype=np.float32)
            audio += [gap, x]
            t += len(gap) / SR
            rttm.append(f"SPEAKER conv_{c:02d} 1 {t:.3f} {len(x) / SR:.3f} <NA> <NA> spk{s} <NA> <NA>")
            t += len(x) / SR
            prev = s
        write(d / f"conv_{c:02d}.flac", np.concatenate(audio))
        (d / f"conv_{c:02d}.rttm").write_text("\n".join(rttm) + "\n", encoding="utf-8")


def _defects(x, rng=RNG):
    t = np.arange(len(x)) / SR
    lp = butter(8, 3400, btype="low", fs=SR, output="sos")
    d = int(0.12 * SR)
    echo = x.copy()
    echo[d:] += 0.6 * x[:-d]
    noise = rng.standard_normal(len(x)).astype(np.float32)
    return {
        "clean": (x, []),
        "clipping": (np.clip(x / np.abs(x).max() * 6, -0.99, 0.99), ["CLIPPING"]),
        "low_level": (x * (10 ** (-50 / 20) / rms(x)), ["LOW_LEVEL"]),
        "hum": (x + 0.05 * np.sin(2 * np.pi * 50 * t) + 0.02 * np.sin(2 * np.pi * 150 * t), ["HUM"]),
        "dc_offset": (x + 0.03, ["DC_OFFSET"]),
        "bandlimit": (sosfilt(lp, x), ["BANDWIDTH_LIMITED"]),
        "noise": (x + noise * rms(x) / rms(noise) / 10 ** (5 / 20), ["HIGH_NOISE"]),
        "silence": (np.concatenate([x, np.zeros(3 * len(x), dtype=np.float32)]), ["SILENCE"]),
        "echo": (echo * 0.6, ["ECHO"]),
    }


def build_quality(spk, out, name="quality", offset=0, rng=RNG):
    """offset 0 = evaluation set; offset 2 (other speakers, own seed) = development set for tuning."""
    d = out / name
    rows = []
    for flac, _ in [u[0] for u in spk.values()][offset::4]:  # 10 speakers
        x = load(flac) * 0.5  # headroom so additive defects do not clip
        for cond, (y, issues) in _defects(x, rng).items():
            f = f"{flac.stem}__{cond}.wav"
            write(d / f, np.asarray(y, dtype=np.float32))
            rows.append((f, "|".join(issues)))
    with open(d / "labels.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["file", "issues"])
        w.writerows(rows)


def build_long(spk, out, seconds=600):
    parts, total = [], 0
    for utts in spk.values():
        for flac, _ in utts[2:]:
            x = load(flac)
            parts += [x, np.zeros(SR // 2, dtype=np.float32)]
            total += len(x) + SR // 2
            if total >= seconds * SR:
                write(out / "long" / "long_10min.flac", np.concatenate(parts))
                return


def build_fleurs(root: Path, out, per_lang=25):
    """root/<lang>/test.tsv + root/<lang>/test.tar.gz (from huggingface.co/datasets/google/fleurs)."""
    import tarfile
    for lang_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        rows = {}
        for line in (lang_dir / "test.tsv").read_text(encoding="utf-8").splitlines():
            f = line.split("	")
            rows.setdefault(f[0], (f[1], f[3]))  # id -> (wav name, normalized transcription); first take per id
        chosen = dict(sorted(rows.items(), key=lambda kv: int(kv[0]))[:per_lang])
        want = {wav: (uid, text) for uid, (wav, text) in chosen.items()}
        d = out / "fleurs" / lang_dir.name
        d.mkdir(parents=True, exist_ok=True)
        with tarfile.open(lang_dir / "test.tar.gz") as tar:
            for m in tar:
                name = Path(m.name).name
                if m.isfile() and name in want:
                    uid, text = want[name]
                    (d / f"{uid}.wav").write_bytes(tar.extractfile(m).read())
                    (d / f"{uid}.txt").write_text(text, encoding="utf-8")


def build_ami(root: Path, out):
    d = out / "ami"
    d.mkdir(parents=True, exist_ok=True)
    for rttm in sorted(root.glob("*.rttm")):
        (d / rttm.name).write_bytes(rttm.read_bytes())
        (d / f"{rttm.stem}.wav").write_bytes((root / f"{rttm.stem}.wav").read_bytes())


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--librispeech", required=True, type=Path, help="LibriSpeech/test-clean directory")
    ap.add_argument("--esc50", required=True, type=Path, help="ESC-50-master directory")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--ffmpeg", default="ffmpeg")
    ap.add_argument("--librispeech-other", type=Path, help="LibriSpeech/test-other directory")
    ap.add_argument("--fleurs", type=Path, help="directory of <lang>/test.tsv + <lang>/test.tar.gz")
    ap.add_argument("--ami", type=Path, help="directory of <meeting>.wav (Mix-Headset) + <meeting>.rttm")
    a = ap.parse_args(argv)
    spk, meta = librispeech(a.librispeech), esc50_meta(a.esc50)
    build_esc50(a.esc50, meta, a.out)
    build_librispeech(spk, a.out)
    build_noise(spk, a.esc50, meta, a.out)
    build_formats(spk, a.out, a.ffmpeg)
    build_diarization(spk, a.out)
    build_quality(spk, a.out)
    build_long(spk, a.out)
    build_quality(spk, a.out, "quality_dev", offset=2, rng=np.random.default_rng(99))
    if a.librispeech_other:
        build_librispeech(librispeech(a.librispeech_other), a.out, "librispeech_other")
    if a.fleurs:
        build_fleurs(a.fleurs, a.out)
    if a.ami:
        build_ami(a.ami, a.out)
    print(f"wrote evaluation sets to {a.out}")


if __name__ == "__main__":
    main()
