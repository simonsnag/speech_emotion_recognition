import os
import sys
from loguru import logger

LOGS_DIR = os.path.join(os.path.dirname(__file__), "logs")
logger.add(
    sys.stderr,
    format="<level>{level}</level>: {message}",
    level="INFO",
    colorize=True,
)
logger.add(
    f"{LOGS_DIR}/backend.log",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} - {message}",
    level="DEBUG",
    rotation="1 MB",
    retention="1 week",
    compression="zip",
)


def get_logger(name):
    return logger.bind(name=name)
