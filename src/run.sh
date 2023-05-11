#! /usr/bin/bash
# This script is used to perform multiruns using hydra
# Author: Caioflp

python main.py --multirun dataset.n_samples=200 dataset.response=sin,step,abs,linear
