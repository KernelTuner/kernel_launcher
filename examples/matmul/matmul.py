#!/usr/bin/env python
""" Simple matrix multiplication example

This file demonstrates how to use Kernel Tuner to tune a relatively
simple matrix multiplication kernel.

"""
import json
import numpy
import kernel_tuner
from kernel_tuner.integration import store_results
from collections import OrderedDict

def tune():
    problem_size = (4096, 4096)
    size = numpy.prod(problem_size)

    A = numpy.random.randn(*problem_size).astype(numpy.float32)
    B = numpy.random.randn(*problem_size).astype(numpy.float32)
    C = numpy.zeros_like(A)

    args = [C, A, B]
    tune_params = OrderedDict()
    tune_params["block_size_x"] = [16*2**i for i in range(3)]
    tune_params["block_size_y"] = [2**i for i in range(2,5)]

    tune_params["tile_size_x"] = [2**i for i in range(3)]
    tune_params["tile_size_y"] = [2**i for i in range(3)]

    grid_div_x = ["block_size_x", "tile_size_x"]
    grid_div_y = ["block_size_y", "tile_size_y"]

    restrict = ["block_size_x==block_size_y*tile_size_y"]

    answer = [numpy.dot(A,B), None, None]

    metrics = OrderedDict()
    metrics["GFLOP/s"] = lambda p : (2*4096**3/1e9) / (p["time"] / 1e3)

    res, env = kernel_tuner.tune_kernel("matmul_kernel", "matmul.cu",
        problem_size, args, tune_params,
        grid_div_y=grid_div_y, grid_div_x=grid_div_x,
        restrictions=restrict, verbose=True, iterations=32, metrics=metrics)

    store_results("matmul_results.json", "matmul_kernel", "matmul.cu", tune_params, problem_size, res, env, top=3)


if __name__ == "__main__":
    tune()

