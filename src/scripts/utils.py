"""General purpose utilities.

Author: Caioflp

"""
import logging
import os
from datetime import datetime
from pathlib import Path
from functools import wraps
from typing import Callable


def setup_logger() -> None:
    """ Performs basic logging configuration.
    """
    logging.basicConfig(
        filename="run.log",
        format="%(asctime)s %(name)-12s %(levelname)-8s %(message)s",
        level=logging.DEBUG,
    )
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger("src").addHandler(console)


# Decorator with arguments syntax is weird
# https://stackoverflow.com/questions/5929107/decorators-with-parameters
def experiment(path: str = None) -> Callable:
    """Marks a function as the main function of an experiment.

    Creates a separate directory for it to be run.

    """
    output_dir = Path("./outputs")
    now = datetime.now()
    if path is None:
        inner_output_dir = output_dir / now.strftime("%Y-%m-%d")
    else:
        inner_output_dir = output_dir / path.lower().replace(" ", "_")
    inner_output_dir.mkdir(parents=True, exist_ok=True)
    run_dir = inner_output_dir / ("run_" + str(len(os.listdir(inner_output_dir))))
    run_dir.mkdir(parents=True, exist_ok=False)

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            os.chdir(run_dir)
            setup_logger()
            return func(*args, **kwargs)
        return wrapper
    return decorator

