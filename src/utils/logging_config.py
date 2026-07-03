"""
Centralized logging configuration for the pipeline.
Call setup_logging() exactly once, from whichever script is the actual
entry point - never from inside extract/transform modules themselves.
"""

import logging


def setup_logging(level: int = logging.INFO) -> None:
    """Configure the root logger for the whole pipeline.

    Safe to call multiple times - subsequent calls are no-ops thanks to
    the `force` flag, but in practice it should only be called once,
    from the entry point script.

    Args:
        level: Minimum severity level to log.
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(message)s",
        force=True,
    )