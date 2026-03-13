import logging

from TREX_Core.logging_config import setup_logging


def test_setup_logging_configures_root_logger() -> None:
    root_logger = logging.getLogger()

    setup_logging(log_level=logging.DEBUG, json_output=False)

    assert root_logger.level == logging.DEBUG
    assert len(root_logger.handlers) == 1
    assert root_logger.handlers[0].level == logging.DEBUG
