import base64

import pytest

from audio_mcp.envelope import AudioMCPError, ErrorCode
from audio_mcp.ingest.sources import (
    Base64Source,
    FilePathSource,
    UrlSource,
    check_path_allowed,
    materialize_source,
)


def test_path_inside_root(tmp_path):
    f = tmp_path / "a b ü.wav"
    f.write_bytes(b"x")
    assert check_path_allowed(str(f), [tmp_path]) == f.resolve()


def test_dotdot_escape(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    (tmp_path / "secret.wav").write_bytes(b"x")
    with pytest.raises(AudioMCPError) as e:
        check_path_allowed(str(root / ".." / "secret.wav"), [root])
    assert e.value.code == ErrorCode.PATH_NOT_ALLOWED


def test_symlink_escape(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    target = tmp_path / "out.wav"
    target.write_bytes(b"x")
    link = root / "link.wav"
    try:
        link.symlink_to(target)
    except OSError:
        pytest.skip("symlinks not permitted")
    with pytest.raises(AudioMCPError) as e:
        check_path_allowed(str(link), [root])
    assert e.value.code == ErrorCode.PATH_NOT_ALLOWED


def test_sibling_prefix_not_inside(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    (tmp_path / "root2").mkdir()
    f = tmp_path / "root2" / "a.wav"
    f.write_bytes(b"x")
    with pytest.raises(AudioMCPError) as e:
        check_path_allowed(str(f), [root])
    assert e.value.code == ErrorCode.PATH_NOT_ALLOWED


def test_missing_file(tmp_path):
    with pytest.raises(AudioMCPError) as e:
        check_path_allowed(str(tmp_path / "nope.wav"), [tmp_path])
    assert e.value.code == ErrorCode.FILE_NOT_FOUND


def test_directory_rejected(tmp_path):
    with pytest.raises(AudioMCPError) as e:
        check_path_allowed(str(tmp_path), [tmp_path])
    assert e.value.code == ErrorCode.INVALID_INPUT


def test_file_too_large(settings, tmp_path):
    f = tmp_path / "a.wav"
    f.write_bytes(b"x" * 2048)
    settings.max_file_mb = 0
    with pytest.raises(AudioMCPError) as e:
        materialize_source(FilePathSource(type="file_path", path=str(f)), settings)
    assert e.value.code == ErrorCode.FILE_TOO_LARGE


def test_file_ok(settings, tmp_path):
    f = tmp_path / "a.wav"
    f.write_bytes(b"x")
    m = materialize_source(FilePathSource(type="file_path", path=str(f)), settings)
    assert m.path == f.resolve() and not m.is_temp and m.original_name == "a.wav"


def test_base64_too_large(settings):
    settings.max_upload_mb = 1
    with pytest.raises(AudioMCPError) as e:
        materialize_source(Base64Source(type="base64", data="A" * (2 * 1024 * 1024)), settings)
    assert e.value.code == ErrorCode.FILE_TOO_LARGE


def test_base64_invalid(settings):
    with pytest.raises(AudioMCPError) as e:
        materialize_source(Base64Source(type="base64", data="!!notbase64!!"), settings)
    assert e.value.code == ErrorCode.INVALID_INPUT


def test_base64_bad_hint_is_neutralised(settings):
    m = materialize_source(
        Base64Source(
            type="base64", data=base64.b64encode(b"abc").decode(), filename_hint="../../evil.sh;rm"
        ),
        settings,
    )
    assert m.path.parent == settings.resolved_temp_dir and m.path.suffix == ".bin" and m.is_temp


def test_base64_good_hint(settings):
    m = materialize_source(
        Base64Source(type="base64", data=base64.b64encode(b"abc").decode(), filename_hint="x.mp3"),
        settings,
    )
    assert m.path.suffix == ".mp3" and m.path.read_bytes() == b"abc"


def test_url_disabled(settings):
    with pytest.raises(AudioMCPError) as e:
        materialize_source(UrlSource(type="url", url="https://x.org/a.mp3"), settings)
    assert e.value.code == ErrorCode.URL_FETCH_DISABLED


def test_url_bad_scheme(settings):
    settings.allow_url_fetch = True
    with pytest.raises(AudioMCPError) as e:
        materialize_source(UrlSource(type="url", url="ftp://x.org/a.mp3"), settings)
    assert e.value.code == ErrorCode.INVALID_INPUT


@pytest.mark.parametrize("host", ["127.0.0.1", "10.0.0.1", "169.254.169.254", "[::1]"])
def test_url_blocked_addresses(settings, host):
    settings.allow_url_fetch = True
    with pytest.raises(AudioMCPError) as e:
        materialize_source(UrlSource(type="url", url=f"http://{host}/a.wav"), settings)
    assert e.value.code == ErrorCode.URL_FETCH_FAILED
    assert "blocked address" in e.value.message
