import logging
import sys
from pathlib import Path

def setup_logger(name: str = "FakeNewsSystem", log_file: str = None, level=logging.INFO) -> logging.Logger:
    """Configures and returns a structured logger for the application."""
    if log_file is None:
        log_path = Path(__file__).resolve().parent.parent / "logs" / "system.log"
    else:
        log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] [%(name)s:%(filename)s:%(lineno)d] - %(message)s"
        )

        # Console handler
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        # File handler
        fh = logging.FileHandler(str(log_path), encoding="utf-8")
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger

logger = setup_logger()
