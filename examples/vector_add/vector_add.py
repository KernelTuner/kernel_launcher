#!/usr/bin/env python
"""Test project for device target creation"""

import os
import sys
from collections import OrderedDict
import json

import numpy as np

from kernel_tuner import tune_kernel
from kernel_tuner.integration import store_results, create_device_targets

def tune(size):

    a = np.random.randn(size).astype(np.float32)
    b = np.random.randn(size).astype(np.float32)
    c = np.zeros_like(b)
    n = np.int32(size)

    args = [c, a, b, n]

    tune_params = dict()
    tune_params["block_size_x"] = [32*i for i in range(1,33)]

    metrics = OrderedDict()
    metrics["GFLOP/s"] = lambda x: (size/1e9)/(x['time']/1e3)

    #tune the kernel
    results, env = tune_kernel("vector_add", "vector_add.cu", size, args, tune_params, metrics=metrics, compiler_options=["-I" + os.getcwd()], lang="cupy")

    #store the best performing configurations
    store_results("vector_add_results.json", "vector_add", "vector_add.cu", tune_params, size, results, env, top=3, objective="GFLOP/s")

    #create a header file with GPU targets
    #create_device_targets("vector_add.h", "vector_add_results.json", objective="GFLOP/s")



if __name__ == "__main__":

    if len(sys.argv) > 1:
        size = int(sys.argv[1])
    else:
        size = int(800000000)

    tune(size)
