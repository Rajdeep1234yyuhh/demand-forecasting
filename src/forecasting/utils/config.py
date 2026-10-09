"""Configuration loading.

Non-secret parameters live in versioned YAML files and are merged in order,
so environment overlays (e.g. ``prod.yaml``) override the base. Secrets come
only from environment variables prefixed with ``DF_``.
"""

from __future__ import annotations

from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


State = Literal["CA", "TX", "WI"]


class DataConfig(_StrictModel):
    root: Path = Path("data")
    raw_prefix: str
    processed_prefix: str
    ingest_state: State | None = None
    train_subset_state: State | None = None
    max_missing_price_ratio: float = Field(default=0.01, ge=0, le=1)

    @property
    def raw_dir(self) -> Path:
        return self.root / self.raw_prefix

    @property
    def processed_dir(self) -> Path:
        return self.root / self.processed_prefix


class ForecastConfig(_StrictModel):
    horizon_days: int = Field(gt=0)
    quantiles: tuple[float, ...]
    backtest_folds: int = Field(gt=0)

    @field_validator("quantiles")
    @classmethod
    def _validate_quantiles(cls, value: tuple[float, ...]) -> tuple[float, ...]:
        if not value:
            raise ValueError("at least one quantile is required")
        if any(not 0 < q < 1 for q in value):
            raise ValueError("quantiles must be strictly between 0 and 1")
        return tuple(sorted(set(value)))


class InventoryConfig(_StrictModel):
    service_levels: tuple[float, ...]
    lead_time_days: int = Field(gt=0)
    holding_cost_rate: float = Field(gt=0)
    stockout_cost_multiplier: float = Field(gt=0)

    @field_validator("service_levels")
    @classmethod
    def _validate_service_levels(cls, value: tuple[float, ...]) -> tuple[float, ...]:
        if any(not 0 < s < 1 for s in value):
            raise ValueError("service levels must be strictly between 0 and 1")
        return tuple(sorted(set(value)))


class AppConfig(_StrictModel):
    env: Literal["dev", "test", "prod"]
    data: DataConfig
    forecast: ForecastConfig
    inventory: InventoryConfig


class Secrets(BaseSettings):
    """Credentials and endpoints, read from ``DF_*`` environment variables."""

    model_config = SettingsConfigDict(env_prefix="DF_", env_file=".env", extra="ignore")

    database_url: SecretStr | None = None
    object_storage_endpoint: str | None = None
    object_storage_access_key: SecretStr | None = None
    object_storage_secret_key: SecretStr | None = None
    mlflow_tracking_uri: str | None = None


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into ``base`` without mutating either."""
    merged: dict[str, Any] = dict(base)
    for key, value in override.items():
        current = merged.get(key)
        if isinstance(current, Mapping) and isinstance(value, Mapping):
            merged[key] = deep_merge(current, value)
        else:
            merged[key] = value
    return merged


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        content = yaml.safe_load(handle) or {}
    if not isinstance(content, dict):
        raise ValueError(f"{path} must contain a YAML mapping at the top level")
    return content


def load_config(*paths: Path | str) -> AppConfig:
    """Load and validate configuration from one or more YAML files, merged in order."""
    if not paths:
        raise ValueError("at least one config path is required")
    merged: dict[str, Any] = {}
    for path in paths:
        merged = deep_merge(merged, _read_yaml(Path(path)))
    return AppConfig.model_validate(merged)


@lru_cache(maxsize=1)
def get_secrets() -> Secrets:
    """Return process-wide secrets, read once."""
    return Secrets()
