import logging

from audio_mcp.logging_setup import setup_logging


def test_logs_hide_transcript(capsys, settings):
    setup_logging(settings)
    logging.getLogger("audio_mcp").info("x", extra={"text": "secret words"})
    err = capsys.readouterr()
    assert "secret words" not in err.err and "[redacted]" in err.err and err.out == ""


def test_log_sensitive_opt_in(capsys, settings):
    settings.log_sensitive = True
    setup_logging(settings)
    logging.getLogger("audio_mcp").info("x", extra={"text": "secret words"})
    assert "secret words" in capsys.readouterr().err


def test_third_party_loggers_quiet(settings):  # M3
    setup_logging(settings)
    for name in ("httpx", "httpcore", "urllib3", "huggingface_hub", "speechbrain", "mcp"):
        assert logging.getLogger(name).getEffectiveLevel() == logging.WARNING
    assert logging.getLogger("audio_mcp").isEnabledFor(logging.INFO)
