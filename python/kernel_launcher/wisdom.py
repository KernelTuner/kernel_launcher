import itertools
import json
import logging
import numpy as np
import os.path
import socket
import subprocess

from datetime import datetime
from .reader import TuningProblem

logger = logging.getLogger(__name__)

WISDOM_VERSION = "1.0"
WISDOM_OBJECTIVE = "time"


def write_wisdom_for_problem(directory: str, problem: TuningProblem, results: list,
                             env: dict, **kwargs):
    return write_wisdom(directory, problem.key, problem.space.params, problem.problem_size,
                        results, env, **kwargs)


def write_wisdom(path: str, key: str, params: dict, problem_size, results, env, *, max_results=5):
    filename = _wisdom_file(path, key)
    device_name = env["device_name"]

    if not os.path.exists(filename):
        logger.info(f"creating wisdom file: {filename}")
        param_keys = list(params.keys())
        lines = [_create_header(key, param_keys)]
    else:
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

    # Remove invalid result (i.e., failed due to insufficient resources)
    results = [r for r in results if _is_valid_config(r)]

    # Get top results
    top_results = sorted(results, key=lambda p: p["time"])[:max_results]

    for result in top_results:
        logger.info(f"appending entry to {filename} for device {device_name}: {result}")
        config = [result[key] for key in param_keys]

        lines.append(json.dumps({
            "config": config,
            "problem_size": problem_size,
            "time": result[WISDOM_OBJECTIVE],
            "environment": _build_environment(env),
        }))

    os.rename(filename, filename + ".backup")
    with open(filename, "w") as handle:
        handle.writelines(line + "\n" for line in lines)


def read_wisdom(path: str, key: str = None, params: dict = None):
    filename = _wisdom_file(path, key)
    results = []

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


def _create_header(key: str, param_keys: list):
    return json.dumps({
        "version": WISDOM_VERSION,
        "objective": WISDOM_OBJECTIVE,
        "tunable_parameters": list(param_keys),
        "key": key,
    })


def _is_valid_config(config):
    time = config.get(WISDOM_OBJECTIVE)
    return isinstance(time, (float, np.float32, np.float64))

def _wisdom_file(path, key):
    if os.path.isdir(path):
        return os.path.join(path, key + ".wisdom")
    elif path.endswith(".wisdom"):
        return path
    else:
        raise ValueError(f"path must be a directory or a file ending with .wisdom: {path}")

def _build_environment(env):
    env = {
        "date": datetime.now().isoformat(),
        "device_name": env["device_name"],
        "compute_capability": int(env["compute_capability"])
    }

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
