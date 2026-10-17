from audio_mcp.envelope import (
    AudioMCPError,
    ErrorCode,
    build_response,
    error_response,
    warning,
)


def test_status_derivation():
    assert build_response(result={"a": 1})["status"] == "success"
    assert build_response(result={"a": 1}, errors=[{"code": "STAGE_FAILED"}])["status"] == "partial"
    assert build_response(result=None, errors=[{"code": "STAGE_FAILED"}])["status"] == "error"
    r = build_response(result=None, job_id="job_1")
    assert r["status"] == "processing" and r["job_id"] == "job_1"


def test_payload_key_and_schema_version():
    r = build_response(result={"x": 1}, payload_key="analysis", audio_id="aud_0123456789abcdef")
    assert "analysis" in r and "result" not in r and r["processing"]["schema_version"] == "1.0"
    assert r["warnings"] == [] and r["errors"] == []


def test_error_response_shape():
    r = error_response(AudioMCPError(ErrorCode.UNSUPPORTED_AUDIO_FORMAT, "nope"))
    assert r == {"status": "error", "error": {"code": "UNSUPPORTED_AUDIO_FORMAT", "message": "nope", "retryable": False}}


def test_error_code_has_17_section_5_1_codes_and_compares_as_str():
    assert len(ErrorCode) == 17
    assert ErrorCode.STAGE_FAILED == "STAGE_FAILED"


def test_audio_mcp_error_stores_enum_for_str_or_enum_input():
    assert AudioMCPError("STAGE_FAILED", "x").code is ErrorCode.STAGE_FAILED
    assert AudioMCPError(ErrorCode.STAGE_FAILED, "x").code is ErrorCode.STAGE_FAILED


def test_audio_mcp_error_to_dict_includes_stage_when_set():
    err = AudioMCPError(ErrorCode.TIMEOUT, "slow", retryable=True, stage="diarization")
    assert err.to_dict() == {"code": "TIMEOUT", "message": "slow", "retryable": True, "stage": "diarization"}


def test_build_response_omits_audio_and_job_id_when_none():
    r = build_response(result={"a": 1})
    assert "audio_id" not in r and "job_id" not in r


def test_build_response_passes_warnings_and_errors_through():
    w = warning("LOW_CONFIDENCE", "low")
    assert w == {"code": "LOW_CONFIDENCE", "message": "low"}
    r = build_response(result={"a": 1}, warnings=[w])
    assert r["warnings"] == [w]


def test_build_response_serialises_processing_info():
    from audio_mcp.envelope import ModelInfo, ProcessingInfo

    info = ProcessingInfo(
        stages_run=["decode"],
        stages_skipped=[{"stage": "diarization", "reason": "no model"}],
        models_used=[ModelInfo(stage="transcription", name="faster-whisper", version="small")],
    )
    p = build_response(result={}, processing=info)["processing"]
    assert p["stages_run"] == ["decode"]
    assert p["models_used"][0] == {"stage": "transcription", "name": "faster-whisper", "version": "small", "local": True}
    assert p["data_sent_off_machine"] is False
