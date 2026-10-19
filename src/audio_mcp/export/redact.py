"""Optional redaction of emails, phone numbers, card numbers and user patterns in text outputs."""
import re

_EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
_CARD = re.compile(r"(?<!\d)(?:\d[ -]?){12,18}\d(?!\d)")
_PHONE = re.compile(r"\+?\d[\d\s().-]{7,}\d")
_TEXT_KEYS = {"text", "excerpt", "translation", "answer", "statement", "description", "label"}  # label: timeline speech nodes carry the transcript


def _luhn(digits: str) -> bool:
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = int(ch) * (2 if i % 2 else 1)
        total += d - 9 if d > 9 else d
    return total % 10 == 0


def _rules(extra_patterns):
    yield _EMAIL, "[EMAIL]", lambda m: True
    yield _CARD, "[CARD]", lambda m: _luhn(re.sub(r"\D", "", m.group(0)))
    yield _PHONE, "[PHONE]", lambda m: True
    for p in extra_patterns:
        yield re.compile(p, re.IGNORECASE), "[REDACTED]", lambda m: True


def redact_text(text: str, extra_patterns=()) -> tuple[str, int]:
    count = 0

    def sub(rx, repl, s, keep):
        def f(m):
            nonlocal count
            if not keep(m):
                return m.group(0)
            count += 1
            return repl
        return rx.sub(f, s)

    for rx, repl, keep in _rules(extra_patterns):
        text = sub(rx, repl, text, keep)
    return text, count


def _redact_words(words: list, extra_patterns) -> int:
    """Per-word text would never match a number spoken as several words ("555 123 4567"): match on the joined
    sequence, then replace every word that overlaps a match. Mutates the (already copied) word dicts."""
    spans, pos = [], 0
    for w in words:
        t = w.get("text") if isinstance(w, dict) else None
        if isinstance(t, str):
            lead = len(t) - len(t.lstrip())
            spans.append((w, pos + lead, pos + len(t.rstrip())))
            pos += len(t.rstrip()) + 1
    joined = " ".join(w["text"].strip() for w, _, _ in spans)
    count = 0
    for rx, repl, keep in _rules(extra_patterns):
        for m in rx.finditer(joined):
            if not keep(m):
                continue
            count += 1
            for w, a, b in spans:
                if a < m.end() and b > m.start():
                    w["text"] = repl
    return count


def redact_analysis(obj, extra_patterns=()):
    """Return a redacted copy of a JSON-like object and the number of replacements."""
    if isinstance(obj, dict):
        out, n = {}, 0
        for k, v in obj.items():
            if k in _TEXT_KEYS and isinstance(v, str):
                out[k], c = redact_text(v, extra_patterns)
            else:
                out[k], c = redact_analysis(v, extra_patterns)
            n += c
        if isinstance(out.get("words"), list):
            n += _redact_words(out["words"], extra_patterns)
        return out, n
    if isinstance(obj, list):
        items = [redact_analysis(v, extra_patterns) for v in obj]
        return [i for i, _ in items], sum(c for _, c in items)
    return obj, 0
