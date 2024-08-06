#!/usr/bin/env bash
pip install -r requirements.txt
PYTHONPATH="${PYTHONPATH}:../" python scripts/benchmark_continuous_response_high_dim.py