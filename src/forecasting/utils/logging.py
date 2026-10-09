"""Structured logging: JSON in deployed environments, readable console output locally."""

from __future__ import annotations

import logging
import sys
from typing import cast

import structlog
from structlog.typing import FilteringBoundLogger, Processor


def configure_logging(level: str = "INFO", *, json_logs: bool = True) -> None:
    """Configure structlog and the stdlib root logger. Safe to call more than once."""
    numeric_level = logging.getLevelNamesMapping().get(level.upper())
    if numeric_level is None:
        raise ValueError(f"unknown log level: {level}")

    logging.basicConfig(format="%(message)s", stream=sys.stdout, level=numeric_level, force=True)

    processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.JSONRenderer() if json_logs else structlog.dev.ConsoleRenderer(),
    ]
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(numeric_level),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=False,
    )


def get_logger(name: str) -> FilteringBoundLogger:
    """Return a logger bound to ``name``."""
    return cast(FilteringBoundLogger, structlog.get_logger(name))
