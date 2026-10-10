"""Dataset manifest: provenance and a deterministic data version for every ingestion run."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict

_CHUNK_BYTES = 1 << 20


class SourceFile(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    sha256: str
    size_bytes: int


class OutputTable(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    rows: int


class Manifest(BaseModel):
    model_config = ConfigDict(frozen=True)

    dataset: str
    data_version: str
    transform_version: str
    code_version: str
    created_at: datetime
    ingest_state: str | None
    missing_price_ratio: float
    sources: tuple[SourceFile, ...]
    outputs: tuple[OutputTable, ...]


def describe_file(path: Path) -> SourceFile:
    """Hash a file in chunks so large inputs never load fully into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_CHUNK_BYTES):
            digest.update(chunk)
    return SourceFile(name=path.name, sha256=digest.hexdigest(), size_bytes=path.stat().st_size)


def compute_data_version(
    sources: Iterable[SourceFile], *, transform_version: str, ingest_state: str | None
) -> str:
    """Return a short, stable ID: same inputs, transform, and filter always give the same ID."""
    digest = hashlib.sha256()
    for source in sorted(sources, key=lambda s: s.name):
        digest.update(f"{source.name}:{source.sha256}\n".encode())
    digest.update(f"transform:{transform_version}\nstate:{ingest_state or 'ALL'}\n".encode())
    return digest.hexdigest()[:12]
