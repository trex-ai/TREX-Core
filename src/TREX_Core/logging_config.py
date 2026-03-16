# logging_config.py
import logging
import sys

import structlog
from structlog.typing import Processor


def setup_logging(log_level: int = logging.INFO, json_output: bool = False) -> None:
    """Call this once at application startup."""

    # Processors shared by structlog and foreign stdlib loggers.
    # `filter_by_level` is intentionally excluded here because `ProcessorFormatter`
    # may invoke `foreign_pre_chain` without a bound logger for non-structlog
    # records from third-party libraries.
    shared_processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]
    structlog_processors: list[Processor] = [
        structlog.stdlib.filter_by_level,
        *shared_processors,
    ]

    # Choose the final renderer
    if json_output:
        renderer: Processor = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer()

    # Configure the stdlib handler with structlog's ProcessorFormatter
    formatter = structlog.stdlib.ProcessorFormatter(
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
        foreign_pre_chain=shared_processors,  # for non-structlog loggers
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    handler.setLevel(log_level)
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(log_level)

    # Configure structlog itself
    structlog.configure(
        processors=[
            *structlog_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
