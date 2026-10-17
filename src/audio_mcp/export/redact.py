"""Optional redaction of emails, phone numbers, card numbers and user patterns in text outputs."""
import re

_EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
_CARD = re.compile(r"(?<!\d)(?:\d[ -]?){12,18}\d(?!\d)")
_PHONE = re.compile(r"\+?\d[\d\s().-]{7,}\d")
_TEXT_KEYS = {"text", "excerpt", "translation", "answer", "statement", "description"}


def _luhn(digits: str) -> bool:
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = int(ch) * (2 if i % 2 else 1)
        total += d - 9 if d > 9 else d
    return total % 10 == 0


def redact_text(text: str, extra_patterns=()) -> tuple[str, int]:
    count = 0

    def sub(rx, repl, s, keep=lambda m: True):
        def f(m):
            nonlocal count
            if not keep(m):
                return m.group(0)
            count += 1
            return repl
        return rx.sub(f, s)

    text = sub(_EMAIL, "[EMAIL]", text)
    text = sub(_CARD, "[CARD]", text, lambda m: _luhn(re.sub(r"\D", "", m.group(0))))
    text = sub(_PHONE, "[PHONE]", text)
    for p in extra_patterns:
        text = sub(re.compile(p, re.IGNORECASE), "[REDACTED]", text)
    return text, count


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
        return out, n
    if isinstance(obj, list):
        items = [redact_analysis(v, extra_patterns) for v in obj]
        return [i for i, _ in items], sum(c for _, c in items)
    return obj, 0
