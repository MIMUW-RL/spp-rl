import logging
import sys
from datetime import datetime
from pathlib import Path


def init_logger():
    """
    Run it once in __init__ to initialize root logger
    """
    file = sys.argv[0]
    path = Path(file)
    name = path.name.split(".py")[0]
    directory = path.parent / "logs"
    directory.mkdir(parents=True, exist_ok=True)
    log_file = directory / (
        name + "_" + datetime.now().strftime("%Y%m%dT%H%M%S.%f")[:-3] + ".log"
    )
    logging.basicConfig(
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
        level=logging.INFO,
        format="%(asctime)s.%(msecs)03d %(processName)s %(levelname)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
