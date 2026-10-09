from pathlib import Path

import pytest
from pydantic import ValidationError

from forecasting.utils.config import Secrets, deep_merge, load_config

CONFIGS = Path(__file__).resolve().parents[2] / "configs"


def test_loads_base_and_inventory() -> None:
    config = load_config(CONFIGS / "base.yaml", CONFIGS / "inventory.yaml")
    assert config.env == "dev"
    assert config.forecast.horizon_days == 28
    assert config.inventory.service_levels == (0.90, 0.95, 0.98)


def test_overlay_overrides_base() -> None:
    config = load_config(CONFIGS / "base.yaml", CONFIGS / "inventory.yaml", CONFIGS / "prod.yaml")
    assert config.env == "prod"


def test_deep_merge_is_recursive_and_pure() -> None:
    base = {"a": {"b": 1, "c": 2}}
    merged = deep_merge(base, {"a": {"c": 3}})
    assert merged == {"a": {"b": 1, "c": 3}}
    assert base == {"a": {"b": 1, "c": 2}}


def test_rejects_unknown_keys(tmp_path: Path) -> None:
    extra = tmp_path / "extra.yaml"
    extra.write_text("unexpected: true\n", encoding="utf-8")
    with pytest.raises(ValidationError):
        load_config(CONFIGS / "base.yaml", CONFIGS / "inventory.yaml", extra)


@pytest.mark.parametrize("bad", ["[0.0, 0.5]", "[1.0]", "[]"])
def test_rejects_invalid_quantiles(tmp_path: Path, bad: str) -> None:
    override = tmp_path / "q.yaml"
    override.write_text(f"forecast:\n  quantiles: {bad}\n", encoding="utf-8")
    with pytest.raises(ValidationError):
        load_config(CONFIGS / "base.yaml", CONFIGS / "inventory.yaml", override)


def test_rejects_non_mapping_yaml(tmp_path: Path) -> None:
    bad = tmp_path / "list.yaml"
    bad.write_text("- a\n- b\n", encoding="utf-8")
    with pytest.raises(ValueError, match="mapping"):
        load_config(bad)


def test_requires_a_path() -> None:
    with pytest.raises(ValueError, match="at least one"):
        load_config()


def test_secrets_read_from_env_and_stay_masked(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DF_DATABASE_URL", "postgresql://u:p@h/db")
    secrets = Secrets(_env_file=None)
    assert secrets.database_url is not None
    assert secrets.database_url.get_secret_value() == "postgresql://u:p@h/db"
    assert "p@h" not in repr(secrets)
