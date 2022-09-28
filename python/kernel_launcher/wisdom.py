import itertools
import json
import logging
import math
import numpy as np
import os.path
import socket
import subprocess

from datetime import datetime
from .reader import TuningProblem

logger = logging.getLogger(__name__)

WISDOM_VERSION = "1.0"
WISDOM_OBJECTIVE = "time"


def write_wisdom_for_problem(path: str, problem: TuningProblem, results: list,
                             env: dict, **kwargs):
    """Write the results of ``TuningProblem.tune`` to a wisdom file. This function calls ``write_wisdom``

    :param path: Directory were wisdom files are stored. Alternatively, this can be a file name ending with ``.wisdom``.
    :param problem: The ``TuningProblem``.
    :param results: The results returned by ``kernel_tuner.tune_kernel``.
    :param env: The environment returned by ``kernel_tuner.tune_kernel``.
    :param kwargs: Additional keyword arguments passed to `write_wisdom`.
    """
    return write_wisdom(path, problem.key, problem.space.params, problem.problem_size, results, env, **kwargs)


def write_wisdom(path: str, key: str, params: dict, problem_size: list, results: list, env: dict, *,
                 max_results=5, merge_existing_results=False):
    """Write the results of ``kernel_tuner.tune_kernel`` to a wisdom file.

    :param path: Directory were wisdom files are stored. Alternatively, this can be a file name ending with ``.wisdom``.
    :param key: The tuning key used inside the wisdom file
    :param params: The tunable parameters as passed to ``kernel_tuner.tune_kernel``
    :param problem_size: The problem size as passed to ``kernel_tuner.tune_kernel``
    :param results: The results returned by ``kernel_tuner.tune_kernel``
    :param env: The environment returned by ``kernel_tuner.tune_kernel``
    :param max_results: Only the top ``max_results`` results are written in the wisdom file.
    :param merge_existing_results: If ``True``, existing results in the wisdom file for the same problem size and
                                   environment are merged with the provided ``results``.

    """
    device_name = env["device_name"]
    config2result = dict()

    # Path to the wisdom file
    filename = _wisdom_file(path, key)

    # If wisdom file exists, read the lines in the file
    if os.path.exists(filename):
        with open(filename, "r") as handle:
            line = next(handle)
            lines = [line]
            param_keys = _check_header(line, key, params)

            for line, record in _parse_wisdom(handle):
                # Skip lines that have a matching problem_size and device_name
                if not record or \
                        record["problem_size"] != problem_size or \
                        record["environment"].get("device_name") != device_name:
                    lines.append(line)
                elif merge_existing_results:
                    index = tuple(record["config"])
                    config2result[index] = record
    else:
        logger.info(f"creating wisdom file: {filename}")
        param_keys = list(params.keys())
        lines = [_create_header(key, param_keys)]

    environment = _build_environment(env)

    for result in results:
        # Skip invalid result (i.e., failed due to insufficient resources)
        if not _is_valid_config(result):
            continue

        config = _convert_config(result, param_keys)
        record = {
            "config": config,
            "problem_size": problem_size,
            "time": result[WISDOM_OBJECTIVE],
            "environment": environment,
        }

        index = tuple(config)
        config2result[index] = record

    # Sort results and append the top results to lines.
    sorted_results = sorted(config2result.values(), key=lambda p: p["time"])

    for result in sorted_results[:max_results]:
        logger.info(f"writing entry to {filename} for device {device_name}: {result}")
        lines.append(json.dumps(result))

    # Keep old file if it exists.
    if os.path.exists(filename):
        os.rename(filename, filename + ".backup")

    with open(filename, "w") as handle:
        handle.writelines(line + "\n" for line in lines)


def read_wisdom(path: str, key: str = None, params: dict = None, *, error_if_missing: bool = True) -> list:
    """
    Read the results of a wisdom file.

    :param path: Directory were wisdom file are stored. Alternatively, this can be file name ending with ``.wisdom``
    :param key: The tuning key. Used to validate if the wisdom file has the correct tuning key. If ``None``, this check
                is not performed
    :param params: The tunable parameters. Used to validate if the wisdom file has the correct parameters. If ``None``,
                    this check is not performed.
    :param error_if_missing: If ``True``, throws an exception if the wisdom file was not found. If ``False``, the error
                             is ignored instead and an empty list is returned.
    :return: List of dictionaries containing the wisdom records.
    """
    filename = _wisdom_file(path, key)
    results = []

    try:
        with open(filename, "r") as handle:
            param_keys = _check_header(next(handle), key, params)

            for line, record in _parse_wisdom(handle):
                # Skip lines that do not have a valid record
                if not record:
                    continue

                # Replace configuration
                record["config"] = dict(zip(param_keys, record["config"]))

                # Append to results
                results.append(record)
    except FileNotFoundError:
        if error_if_missing:
            raise

    return results


def _parse_wisdom(handle):
    for lineno, line in zip(itertools.count(2), handle):
        line = line.strip()
        data = None

        if line and not line.startswith("#") and not line.startswith("//"):
            try:
                data = json.loads(line)
            except Exception as e:
                logger.warning(f"exception occured while parsing line {lineno}: {e}")

        yield line, data


def _check_header(line: str, key: str, params: dict):
    data = json.loads(line)

    if data.get("version") != WISDOM_VERSION:
        raise RuntimeError("invalid version in wisdom file")

    if data.get("objective") != WISDOM_OBJECTIVE:
        raise RuntimeError("invalid version in wisdom file")

    if key is not None:
        if data.get("key") != key:
            print(data)
            raise RuntimeError(f"invalid key in wisdom file: {key} != {data.get('key')}")

    keys = data.get("tunable_parameters", [])
    if params is not None:
        if set(keys) != set(params.keys()):
            raise RuntimeError("invalid tunable parameters in wisdom file: " +
                               f"{list(params.keys())} != {keys}")

    return keys


def _create_header(key: str, param_keys: list) -> str:
    """Header of wisdom file (ie, the first line).
    """
    return json.dumps({
        "version": WISDOM_VERSION,
        "objective": WISDOM_OBJECTIVE,
        "tunable_parameters": list(param_keys),
        "key": key,
    })


def _is_valid_config(config):
    objective = config.get(WISDOM_OBJECTIVE)
    return isinstance(objective, (float, np.float32, np.float64))


def _convert_config(config: dict, keys: list):
    result = []

    for key in keys:
        if key not in config:
            raise ValueError(f"invalid configuration, missing key {key}: {config!r}")

        value = config[key]

        if isinstance(value, np.integer):
            value = int(value)
        elif isinstance(value, np.bool_):
            value = bool(value)
        elif isinstance(value, (np.inexact, float)):
            # It can happen that what was suppose to be an integer, accidentally gets cast to a float.
            # To combat this, here we cast the value back to an integer if its fractional part is zero.
            whole, frac = math.modf(value)

            if frac == 0.0 and int(whole) == whole:
                value = int(whole)
            else:
                value = float(value)

        result.append(value)

    return result


def _wisdom_file(path, key):
    if os.path.isdir(path) and key is not None:
        return os.path.join(path, key + ".wisdom")
    elif path.endswith(".wisdom"):
        return path
    else:
        raise ValueError(f"path must be a directory or a file ending with .wisdom: {path}")


def _build_environment(env):
    env = {
        "date": datetime.now().isoformat(),
        "device_name": env["device_name"],
        "compute_capability": int(env["compute_capability"]),
    }

    # Kernel tuner related
    try:
        import kernel_tuner
        env["kernel_tuner_version"] = kernel_tuner.__version__
    except AttributeError as e:
        logger.warning(f"ignore error: kernel_tuner.__version__ is not available: {e}")

    # CUDA related
    try:
        import pycuda.driver
        env["cuda_driver_version"] = pycuda.driver.get_driver_version()

        major, minor, patch = pycuda.driver.get_version()
        env["cuda_version"] = major * 1000 + minor * 10 + patch
    except ImportError:
        pass  # Ignore error if pycuda is not installed
    except Exception as e:
        logger.warning(f"ignore PyCUDA error: {e}")

    # Host related
    try:
        host_name = socket.gethostname()
        if host_name:
            env["host_name"] = host_name

            host_ip = socket.gethostbyname(host_name)
            env["host_address"] = host_ip
    except Exception as e:
        logger.warning(f"ignore socket error: {e}")

    # Git version
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True)

        if result.returncode == 0:
            git_hash = result.stdout.decode("ascii").strip()

            if git_hash:
                env["git"] = git_hash
    except Exception as e:
        logger.warning(f"ignore error while fetching git hash: {e}")

    return env
