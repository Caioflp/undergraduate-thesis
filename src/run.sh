#!/usr/bin/env bash
pip install -r requirements.txt
PYTHONPATH="${PYTHONPATH}:../" python scripts/benchmark_continuous_response.py
PYTHONPATH="${PYTHONPATH}:../" python scripts/benchmark_binary_response.py