from enum import StrEnum
from typing import Any

from pydantic import BaseModel


class ErrorCode(StrEnum):
    INVALID_INPUT = "INVALID_INPUT"
    PATH_NOT_ALLOWED = "PATH_NOT_ALLOWED"
    FILE_NOT_FOUND = "FILE_NOT_FOUND"
    UNSUPPORTED_AUDIO_FORMAT = "UNSUPPORTED_AUDIO_FORMAT"
    CORRUPT_AUDIO = "CORRUPT_AUDIO"
    EMPTY_AUDIO = "EMPTY_AUDIO"
    FILE_TOO_LARGE = "FILE_TOO_LARGE"
    DURATION_TOO_LONG = "DURATION_TOO_LONG"
    URL_FETCH_DISABLED = "URL_FETCH_DISABLED"
    URL_FETCH_FAILED = "URL_FETCH_FAILED"
    AUDIO_NOT_FOUND = "AUDIO_NOT_FOUND"
    JOB_NOT_FOUND = "JOB_NOT_FOUND"
    MODEL_UNAVAILABLE = "MODEL_UNAVAILABLE"
    STAGE_FAILED = "STAGE_FAILED"
    TIMEOUT = "TIMEOUT"
    CANCELLED = "CANCELLED"
    LLM_DISABLED = "LLM_DISABLED"


class AudioMCPError(Exception):
    def __init__(
        self,
        code: ErrorCode | str,
        message: str,
        *,
        retryable: bool = False,
        stage: str | None = None,
    ):
        super().__init__(message)
        self.code = ErrorCode(code)
        self.message = message
        self.retryable = retryable
        self.stage = stage

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"code": self.code.value, "message": self.message, "retryable": self.retryable}
        if self.stage is not None:
            d["stage"] = self.stage
        return d


class ModelInfo(BaseModel):
    stage: str
    name: str
    version: str
    local: bool = True


class ProcessingInfo(BaseModel):
    duration_seconds: float = 0
    stages_run: list[str] = []
    stages_skipped: list[dict] = []
    cache_hits: list[str] = []
    models_used: list[ModelInfo] = []
    data_sent_off_machine: bool = False
    schema_version: str = "1.0"
    plan: list[dict] | None = None


def build_response(
    *,
    result: dict | None,
    audio_id: str | None = None,
    payload_key: str = "result",
    warnings=(),
    errors=(),
    processing: ProcessingInfo | None = None,
    job_id: str | None = None,
    status: str | None = None,
) -> dict[str, Any]:
    if status is None:
        if job_id:
            status = "processing"
        elif errors:
            status = "partial" if result is not None else "error"
        else:
            status = "success"
    resp: dict[str, Any] = {"status": status}
    if audio_id is not None:
        resp["audio_id"] = audio_id
    if job_id is not None:
        resp["job_id"] = job_id
    resp[payload_key] = result
    resp["warnings"] = list(warnings)
    resp["errors"] = list(errors)
    resp["processing"] = (processing or ProcessingInfo()).model_dump()
    return resp


def error_response(err: AudioMCPError) -> dict[str, Any]:
    return {"status": "error", "error": err.to_dict()}


def warning(code: str, message: str) -> dict[str, str]:
    return {"code": code, "message": message}
