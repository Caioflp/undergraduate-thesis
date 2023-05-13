"""General purpose utilities.

Author: Caioflp

"""
import os
from datetime import datetime
from pathlib import Path

from functools import wraps
from typing import Callable


def run_on_output_dir(func: Callable) -> Callable:
    """Runs function in its own newly created directory.

    """
    now = datetime.now()
    output_dir = Path("./outputs")
    day_dir = output_dir / now.strftime("%Y-%m-%d")
    run_dir = day_dir / now.strftime("%H-%M-%S")
    run_dir.mkdir(parents=True, exist_ok=False)

    @wraps(func)
    def wrapper(*args, **kwargs):
        os.chdir(run_dir)
        return func(*args, **kwargs)

    return wrapper
