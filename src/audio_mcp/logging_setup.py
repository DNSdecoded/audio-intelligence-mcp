import json
import logging
import sys

REDACT = ("text", "transcript", "question", "answer")
_NOISY = ("httpx", "httpcore", "urllib3", "huggingface_hub", "speechbrain", "mcp")  # per-request INFO chatter


class JsonFormatter(logging.Formatter):
    def format(self, record):
        d = {"ts": self.formatTime(record), "level": record.levelname, "logger": record.name, "msg": record.getMessage()}
        for k in REDACT:
            if hasattr(record, k):
                d[k] = getattr(record, k)
        if record.exc_info:
            d["exc"] = self.formatException(record.exc_info)
        return json.dumps(d, default=str)


class RedactFilter(logging.Filter):
    def filter(self, record):
        for k in REDACT:
            if hasattr(record, k):
                setattr(record, k, "[redacted]")
        return True


def setup_logging(settings) -> None:
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
    handlers = [logging.StreamHandler(sys.stderr)]  # never stdout: it carries the MCP protocol
    if settings.log_file:
        handlers.append(logging.FileHandler(settings.log_file, encoding="utf-8"))
    for h in handlers:
        h.setFormatter(JsonFormatter())
        if not settings.log_sensitive:
            h.addFilter(RedactFilter())
        root.addHandler(h)
    root.setLevel(settings.log_level.upper())
    for name in _NOISY:
        logging.getLogger(name).setLevel(logging.WARNING)
