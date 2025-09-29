import logging
from pathlib import Path

def setup_logger(log_file: str = None, level=logging.INFO):
    logger = logging.getLogger("PlantAI")
    logger.setLevel(level)
    formatter = logging.Formatter("[%(asctime)s][%(levelname)s] %(message)s")

    # console
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # file
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, mode="a")
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    return logger

