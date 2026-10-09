import json

import pytest

from forecasting.utils.logging import configure_logging, get_logger


def test_json_logs(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging("INFO", json_logs=True)
    get_logger("test").info("hello", run_id="abc")
    record = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert record["event"] == "hello"
    assert record["run_id"] == "abc"
    assert record["level"] == "info"


def test_level_filtering(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging("WARNING", json_logs=True)
    get_logger("test").info("hidden")
    assert "hidden" not in capsys.readouterr().out


def test_rejects_unknown_level() -> None:
    with pytest.raises(ValueError, match="unknown log level"):
        configure_logging("LOUD")
