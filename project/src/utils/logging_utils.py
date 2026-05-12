import logging


def setup_logging(level: str = "INFO") -> None:
    """Configure basic console logging for scripts and service."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
