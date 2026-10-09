import hashlib
from pathlib import Path

from forecasting.data.manifest import SourceFile, compute_data_version, describe_file


def test_describe_file_hashes_content(tmp_path: Path) -> None:
    path = tmp_path / "f.csv"
    path.write_bytes(b"a,b\n1,2\n")
    described = describe_file(path)
    assert described.sha256 == hashlib.sha256(b"a,b\n1,2\n").hexdigest()
    assert described.size_bytes == 8


def test_data_version_is_order_independent_and_sensitive() -> None:
    a = SourceFile(name="a.csv", sha256="1" * 64, size_bytes=1)
    b = SourceFile(name="b.csv", sha256="2" * 64, size_bytes=1)
    base = compute_data_version([a, b], transform_version="1", ingest_state=None)
    assert base == compute_data_version([b, a], transform_version="1", ingest_state=None)
    assert base != compute_data_version([a, b], transform_version="2", ingest_state=None)
    assert base != compute_data_version([a, b], transform_version="1", ingest_state="CA")
    assert len(base) == 12
