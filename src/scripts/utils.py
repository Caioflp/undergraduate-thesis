"""General purpose utilities.

Author: Caioflp

"""
import os
from datetime import datetime
from pathlib import Path

from functools import wraps
from typing import Callable


# Decorator with arguments syntax is weird
# https://stackoverflow.com/questions/5929107/decorators-with-parameters
def experiment(path: str = None):
    """Marks a function as the main function of an experiment.

    Creates a separate directory for it to be run.

    """
    output_dir = Path("./outputs")
    now = datetime.now()
    if path is None:
        inner_output_dir = output_dir / now.strftime("%Y-%m-%d")
    else:
        inner_output_dir = output_dir / path.lower().replace(" ", "_")
    run_dir = inner_output_dir / now.strftime("%H-%M-%S")
    run_dir.mkdir(parents=True, exist_ok=False)

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            os.chdir(run_dir)
            return func(*args, **kwargs)
        return wrapper
    return decorator
