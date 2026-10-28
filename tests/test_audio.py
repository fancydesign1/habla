import pytest
from src.audio import UlawToLinear16Converter, Mp3ToUlawConverter


def test_ulaw_converter_produces_bytes():
    c = UlawToLinear16Converter()
    silence_ulaw = b'\x7f' * 160  # 20ms of silence at 8kHz
    c.write(silence_ulaw)
    import time; time.sleep(0.1)
    out = c.read()
    c.close()
    assert isinstance(out, bytes)
    assert len(out) > 0


def test_ulaw_converter_write_after_close_skips_gracefully():
    c = UlawToLinear16Converter()
    c.close()
    c.write(b'\x7f' * 160)  # must not raise


def test_mp3_converter_restart_replaces_process():
    c = Mp3ToUlawConverter()
    pid1 = c._process.pid
    c.restart()
    pid2 = c._process.pid
    c.close()
    assert pid1 != pid2
