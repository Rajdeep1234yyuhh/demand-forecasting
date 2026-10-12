from pathlib import Path

import pytest

from forecasting.data.dataset import Dataset
from forecasting.models.registry import MODEL_NAMES, make_model
from forecasting.utils.config import AppConfig, load_config

CONFIGS = Path(__file__).resolve().parents[2] / "configs"


@pytest.fixture
def config() -> AppConfig:
    return load_config(CONFIGS / "base.yaml", CONFIGS / "inventory.yaml")


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_builds_every_registered_model(name: str, config: AppConfig, dataset: Dataset) -> None:
    assert make_model(name, config, dataset).name == name


def test_rejects_unknown_model(config: AppConfig, dataset: Dataset) -> None:
    with pytest.raises(ValueError, match="unknown model 'prophet'"):
        make_model("prophet", config, dataset)
