import numpy as np
from pathlib import Path

yuri_experiment_path = Path("/Users/caio/Documents/benchmark-01-29-2024/benchmark-with-strong-instrument")
my_experiment_path = Path("./outputs/benchmark-with-strong-instrument").resolve()

for response in ["step", "abs", "linear", "sin"]:
    my_mse_path = my_experiment_path / response / "mse_arrays.npz"
    yuri_mse_path = yuri_experiment_path / response / "mse_arrays.npz"
    my_mse = np.load(my_mse_path)
    print([key for key in my_mse.keys()])
    yuri_mse = np.load(yuri_mse_path)
    print([key for key in yuri_mse.keys()])
    joint_mse = {**yuri_mse, **my_mse}
    print(joint_mse.keys())
    joint_mse.pop("SAGD-IV")
    np.savez(my_mse_path, **joint_mse)

