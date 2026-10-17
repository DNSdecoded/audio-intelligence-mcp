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
