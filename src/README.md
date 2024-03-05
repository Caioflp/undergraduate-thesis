## Running the experiments:

From this folder, run
```sh
  pip install -r requirements.txt
  PYTHONPATH="${PYTHONPATH}:../" python scripts/benchmark_continuous_response.py
  PYTHONPATH="${PYTHONPATH}:../" python scripts/benchmark_binary_response.py
```
Results will be stored under `src/outputs/`.