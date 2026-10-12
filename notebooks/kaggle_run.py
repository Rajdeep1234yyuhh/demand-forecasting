# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Real-data run on Kaggle
#
# Runs the whole pipeline on the real M5 data in a Kaggle notebook: install, ingest, EDA,
# a backtest of every model, and the results charts. It records how long each part took and
# its peak memory, then packs everything into one zip in the Output tab.
#
# **Before you run it**
#
# - Accept the M5 rules: https://www.kaggle.com/competitions/m5-forecasting-accuracy/rules
# - **Add Input** in the side panel: the *M5 Forecasting - Accuracy* competition data, and
#   your private dataset holding the zip from `make kaggle-bundle`.
# - **Session options**: accelerator *None* (nothing here uses a GPU), and internet *on*
#   (needs a phone-verified account) to install the locked dependencies.
# - Start it with **Save Version**, then **Save & Run All (Commit)**. It runs in the
#   background, and the zip appears in that version's **Output** tab.
#
# This notebook runs on Kaggle's own Python and uses only the standard library. The project
# runs in its own Python 3.12 environment, installed from `uv.lock` with the same package
# versions as local development and CI.

# %%
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from datetime import datetime, timezone
from fnmatch import fnmatch
from pathlib import Path
from typing import NamedTuple

# Every store ("ALL"), or one state: "CA", "TX" or "WI".
SCOPE = "ALL"
# uv is pinned, so every run builds the environment the same way.
UV_VERSION = "0.11.32"

KAGGLE_INPUT = Path("/kaggle/input")
KAGGLE_OUTPUT = Path("/kaggle/working")
# Outside /kaggle/working, so the environment and data are not saved as notebook output.
WORK_DIR = Path(tempfile.gettempdir()) / "demand-forecasting"
LOG_DIR = KAGGLE_OUTPUT / "logs"
RUN_INFO = KAGGLE_OUTPUT / "run_info.json"

M5_FILES = ("calendar.csv", "sell_prices.csv", "sales_train_evaluation.csv")
PROJECT_MARKER = 'name = "demand-forecasting"'
SKIPPED_TOP_LEVEL = {".venv", "data", "reports", "mlruns", "dist"}
SEARCH_DEPTH = 4
GIB = 1024**3
# Terminal color codes: kept in the notebook output, stripped from the log files.
ANSI_CODES = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
# A session with less memory than this may run out before the run finishes.
SUGGESTED_MEMORY_GB = {"ALL": 16, "CA": 8, "TX": 8, "WI": 8}
# Settings of Kaggle's own Python that must not leak into the project's environment.
HOST_ONLY_VARIABLES = {
    "JUPYTER_PATH",
    "MPLBACKEND",
    "PYTHONHOME",
    "PYTHONPATH",
    "PYTHONSTARTUP",
    "VIRTUAL_ENV",
}

if SCOPE not in SUGGESTED_MEMORY_GB:
    raise ValueError(f"SCOPE must be one of {sorted(SUGGESTED_MEMORY_GB)}, got {SCOPE!r}")


# %% [markdown]
# ## Helpers


# %%
def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def cpu_limit() -> int:
    """CPUs this session can use: its CPU affinity, capped by a cgroup v2 quota if one is set."""
    cpus = len(os.sched_getaffinity(0))
    quota_file = Path("/sys/fs/cgroup/cpu.max")
    if quota_file.is_file():
        quota, period = quota_file.read_text(encoding="utf-8").split()[:2]
        if quota != "max":
            cpus = min(cpus, max(1, int(quota) // int(period)))
    return cpus


def memory_limit_gb() -> float:
    """Memory this session can use: physical memory, capped by a cgroup limit if one is set."""
    limit = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    for limit_file in (
        Path("/sys/fs/cgroup/memory.max"),  # cgroup v2
        Path("/sys/fs/cgroup/memory/memory.limit_in_bytes"),  # cgroup v1
    ):
        if limit_file.is_file():
            value = limit_file.read_text(encoding="utf-8").strip()
            if value.isdigit():
                limit = min(limit, int(value))
    return limit / GIB


def find_files(root: Path, pattern: str) -> list[Path]:
    """Files whose name matches ``pattern``, at most SEARCH_DEPTH folders below ``root``."""
    found: list[Path] = []
    for folder, subfolders, files in os.walk(root):
        if len(Path(folder).relative_to(root).parts) >= SEARCH_DEPTH:
            subfolders.clear()
        found += [Path(folder) / name for name in files if fnmatch(name, pattern)]
    return sorted(found)


def locate_m5() -> Path:
    """The attached folder holding the M5 files the pipeline reads."""
    folders = [
        path.parent
        for path in find_files(KAGGLE_INPUT, M5_FILES[-1])
        if all((path.parent / name).is_file() for name in M5_FILES)
    ]
    if not folders:
        raise FileNotFoundError(
            f"No folder under {KAGGLE_INPUT} has {', '.join(M5_FILES)}. Accept the M5 rules, "
            "then add the 'M5 Forecasting - Accuracy' competition with Add Input."
        )
    if len(folders) > 1:
        raise RuntimeError(f"M5 is attached more than once: {folders}. Keep one.")
    return folders[0]


def is_project(pyproject: str) -> bool:
    return PROJECT_MARKER in pyproject


def zipped_project(archive: Path) -> str | None:
    """Folder of the project inside ``archive`` ("" for its root), or None if it isn't there."""
    with zipfile.ZipFile(archive) as bundle:
        for name in bundle.namelist():
            folder, _, file = name.rpartition("/")
            if (
                file == "pyproject.toml"
                and "/" not in folder
                and is_project(bundle.read(name).decode("utf-8", errors="replace"))
            ):
                return folder
    return None


def locate_code() -> Path:
    """The project from the attached code dataset. Extracts the zip if Kaggle kept it zipped."""
    folders = [
        path.parent
        for path in find_files(KAGGLE_INPUT, "pyproject.toml")
        if is_project(path.read_text(encoding="utf-8", errors="replace"))
    ]
    if len(folders) > 1:
        raise RuntimeError(f"The project is attached more than once: {folders}. Keep one.")
    if folders:
        return folders[0]

    archives = {path: zipped_project(path) for path in find_files(KAGGLE_INPUT, "*.zip")}
    found = [(path, folder) for path, folder in archives.items() if folder is not None]
    if not found:
        raise FileNotFoundError(
            f"No copy of the project under {KAGGLE_INPUT}. Run `make kaggle-bundle`, upload "
            "the zip as a private Kaggle dataset, and add that dataset with Add Input."
        )
    if len(found) > 1:
        raise RuntimeError(f"The project is attached more than once: {found}. Keep one.")
    [(archive, folder)] = found
    extracted = WORK_DIR.with_name("code-bundle")
    shutil.rmtree(extracted, ignore_errors=True)
    with zipfile.ZipFile(archive) as bundle:
        bundle.extractall(extracted)
    return extracted / folder


def read_build_info(code: Path) -> dict[str, str]:
    """Commit the code bundle was built from, written by `make kaggle-bundle`."""
    path = code / "BUILD_INFO.json"
    info: dict[str, str] = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    return info


def prepare_workspace(code: Path, m5: Path) -> None:
    """Copy the code somewhere writable and link the M5 files to where the pipeline reads."""
    shutil.rmtree(WORK_DIR, ignore_errors=True)

    def skip(folder: str, names: list[str]) -> set[str]:
        top_level = Path(folder) == code
        return {
            name
            for name in names
            if name == "__pycache__" or (top_level and name in SKIPPED_TOP_LEVEL)
        }

    shutil.copytree(code, WORK_DIR, ignore=skip)
    # Kaggle inputs are read-only, and copies keep their permissions.
    for path in [WORK_DIR, *WORK_DIR.rglob("*")]:
        path.chmod(path.stat().st_mode | 0o200)
    raw = WORK_DIR / "data" / "raw"
    raw.mkdir(parents=True)
    (raw / "m5").symlink_to(m5, target_is_directory=True)


CPUS = cpu_limit()
ENV = {key: value for key, value in os.environ.items() if key not in HOST_ONLY_VARIABLES}
ENV.update(
    {
        # LightGBM's OpenMP counts the machine's cores, not the session's CPU quota.
        "OMP_NUM_THREADS": str(CPUS),
        # The project environment is the active one. Without this, `python -m uv` would
        # point uv at Kaggle's Python whenever that Python is itself a virtual environment.
        "VIRTUAL_ENV": str(WORK_DIR / ".venv"),
        # Use the project environment's Jupyter kernel, never a kernel Kaggle registered.
        "JUPYTER_PATH": str(WORK_DIR / ".venv" / "share" / "jupyter"),
        # Stream logs as they happen instead of in blocks.
        "PYTHONUNBUFFERED": "1",
    }
)
UV = (sys.executable, "-m", "uv")
PROJECT = (*UV, "run", "--frozen", "--no-dev", "--group", "notebook")


class Step(NamedTuple):
    name: str
    seconds: float
    peak_memory_gb: float
    exit_code: int


RUN: dict[str, object] = {"scope": SCOPE, "started_at": utc_now(), "uv_version": UV_VERSION}
STEPS: list[Step] = []


def write_run_info() -> None:
    KAGGLE_OUTPUT.mkdir(parents=True, exist_ok=True)
    steps = [step._asdict() for step in STEPS]
    RUN_INFO.write_text(json.dumps({**RUN, "steps": steps}, indent=2), encoding="utf-8")


def run_step(name: str, *command: str, hint: str = "") -> None:
    """Run one part of the pipeline, streaming its log and recording time and peak memory.

    Peak memory is the largest resident size of any single process the part started.
    """
    print(f"\n=== {name}: {' '.join(command)}", flush=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"{len(STEPS) + 1:02d}-{name.replace(' ', '-')}.log"
    started = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log:
        # Fixed argument lists, no shell.
        process = subprocess.Popen(  # noqa: S603
            command,
            cwd=WORK_DIR if WORK_DIR.is_dir() else None,
            env=ENV,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        if process.stdout is not None:
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(ANSI_CODES.sub("", line))
            process.stdout.close()
        # Unlike wait(), wait4 also reports the resources the process tree used.
        _, status, usage = os.wait4(process.pid, 0)
        process.returncode = os.waitstatus_to_exitcode(status)
    step = Step(
        name=name,
        seconds=round(time.perf_counter() - started, 1),
        peak_memory_gb=round(usage.ru_maxrss * 1024 / GIB, 2),  # Linux reports kibibytes
        exit_code=process.returncode,
    )
    STEPS.append(step)
    write_run_info()
    print(
        f"=== {name}: {step.seconds / 60:.1f} min, peak memory {step.peak_memory_gb:.2f} GB",
        flush=True,
    )
    if process.returncode != 0:
        raise RuntimeError(
            f"{name} failed with exit code {process.returncode}; log in {log_path}. {hint}".strip()
        )


def package_results() -> Path:
    """Zip the executed notebooks and the report, laid out like the repository."""
    manifest = WORK_DIR / "data" / "processed" / "m5" / "manifest.json"
    version = json.loads(manifest.read_text(encoding="utf-8"))["data_version"]
    report = sorted((WORK_DIR / "reports" / "backtest" / version).iterdir())[-1]
    RUN.update(
        data_version=version,
        report=report.relative_to(WORK_DIR).as_posix(),
        finished_at=utc_now(),
    )
    write_run_info()
    shutil.copy(manifest, report / "manifest.json")
    shutil.copy(RUN_INFO, report / "run_info.json")
    shutil.copy(report / "summary.md", KAGGLE_OUTPUT / "summary.md")

    archive = KAGGLE_OUTPUT / f"m5-run-{SCOPE.lower()}-{version}.zip"
    notebooks = [WORK_DIR / "notebooks" / name for name in ("01_eda.ipynb", "02_results.ipynb")]
    files = [*notebooks, *sorted(path for path in report.rglob("*") if path.is_file())]
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for path in files:
            bundle.write(path, path.relative_to(WORK_DIR).as_posix())
    return archive


def print_steps() -> None:
    print(f"\n{'part':<12}{'minutes':>9}{'peak memory':>14}")
    for step in STEPS:
        print(f"{step.name:<12}{step.seconds / 60:>9.1f}{step.peak_memory_gb:>11.2f} GB")


# %% [markdown]
# ## Check the session and find the inputs

# %%
m5_dir = locate_m5()
code_dir = locate_code()
memory_gb = memory_limit_gb()
RUN.update(
    code_commit=read_build_info(code_dir).get("commit"),
    machine={
        "cpus": CPUS,
        "memory_gb": round(memory_gb, 1),
        "free_disk_gb": round(shutil.disk_usage(WORK_DIR.parent).free / GIB, 1),
        "python": platform.python_version(),
        "platform": platform.platform(),
    },
)
write_run_info()
print(f"M5 data: {m5_dir}")
print(f"Code:    {code_dir} (commit {RUN['code_commit'] or 'unknown'})")
print(f"Session: {CPUS} CPUs, {memory_gb:.1f} GB memory; scope {SCOPE}")
if memory_gb < SUGGESTED_MEMORY_GB[SCOPE]:
    print(
        f"Warning: scope {SCOPE} may need {SUGGESTED_MEMORY_GB[SCOPE]} GB of memory "
        f"and this session has {memory_gb:.1f} GB."
    )

# %% [markdown]
# ## Install
#
# Copies the code to a writable folder, links the M5 files into `data/raw/m5`, and installs
# the locked environment. uv downloads Python 3.12 if Kaggle's Python is another version.

# %%
prepare_workspace(code_dir, m5_dir)
NO_INTERNET = "If it is a network error, turn the internet on in Session options."
run_step(
    "install uv",
    sys.executable,
    "-m",
    "pip",
    "install",
    "--quiet",
    "--disable-pip-version-check",
    f"uv=={UV_VERSION}",
    hint=NO_INTERNET,
)
run_step("install", *UV, "sync", "--frozen", "--no-dev", "--group", "notebook", hint=NO_INTERNET)

# %% [markdown]
# ## Ingest
#
# Validates the raw files and publishes Parquet with a `data_version`, as `make ingest` does.

# %%
run_step("ingest", *PROJECT, "df-ingest")

# %% [markdown]
# ## EDA
#
# Executes `notebooks/01_eda.py` into `notebooks/01_eda.ipynb`, as `make eda` does.

# %%
run_step("eda", *PROJECT, "jupytext", "--to", "ipynb", "--execute", "notebooks/01_eda.py")

# %% [markdown]
# ## Backtest
#
# Every model over three 28-day folds: accuracy, quantile calibration, and inventory cost.
# LightGBM takes most of the time.

# %%
run_step("backtest", *PROJECT, "df-backtest", "--state", SCOPE)

# %% [markdown]
# ## Results charts
#
# Executes `notebooks/02_results.py` into `notebooks/02_results.ipynb`, as `make results` does.

# %%
run_step("results", *PROJECT, "jupytext", "--to", "ipynb", "--execute", "notebooks/02_results.py")

# %% [markdown]
# ## Package
#
# The zip is laid out like the repository. Unzip it at the repository root: commit the two
# executed notebooks, and keep the report locally (`reports/` is not committed). The report
# folder also holds `manifest.json` (the data version) and `run_info.json` (session, commit,
# and the time and memory of each part). `summary.md` and `run_info.json` are also saved
# on their own so they can be read in the Output tab.

# %%
archive = package_results()
print_steps()
print(f"\nDownload {archive.name} from this version's Output tab.\n")
print((KAGGLE_OUTPUT / "summary.md").read_text(encoding="utf-8"))
