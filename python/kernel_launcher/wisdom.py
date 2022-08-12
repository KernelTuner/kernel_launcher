from .reader import TuningProblem

WISDOM_VERSION = "1.0"
WISDOM_OBJECTIVE = "time"

def write_wisdom_for_problem(directory: str, problem: TuningProblem, results: list, env: dict, *, **kwargs):
    return write_wisdom(directory, problem.key, problem.space.params, problem.problem_size, results, env, **kwargs)


def write_wisdom(path: str, key: str, params: dict, problem_size, results, env, *, max_results=5):
    if os.path.isdir(path):
        filename = os.path.join(path, key + ".wisdom")
    elif not path.endswith(".wisdom"):
        filename = path + ".wisdom"
    else:
        filename = path

    device_name = env["device_name"]

    if not os.path.exists(filename):
        logger.info(f"creating wisdom file: {filename}")
        param_keys = list(params.keys())
        lines = [_create_header(key, param_keys)]
    else:
        with open(filename, "r") as handle:
            line = next(handle)
            lines = [line]
            params_keys = _check_header(line, params, key)

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
            "time": result["time"],
            "environment", env,
        }))

    os.rename(file_name, file_name + ".backup")
    with open(file_name, "w") as handle:
        handle.writelines(line + "\n" for line in lines)


def read_wisdom(path: str):
    results = []

    with open(path, "r") as handle:
        param_keys = _check_header(next(handle), params)

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
    for lineno, line in zip(iteratools.count(2), handle):
        line = line.strip()
        data = None

        if line and not line.startswith("#") and not line.startswith("//"):
            try:
                data = json.loads(line)
            except Exception as e:
                logger.warning(f"exception occured while parsing line {lineno}: {e}")

        yield line, data


def _check_header(line: str, params: dict, key: str):
   data = json.loads(line)

   if data.get("version") != WISDOM_VERSION:
       raise RuntimeError("invalid version in wisdom file")

   if data.get("objective") != WISDOM_OBJECTIVE:
       raise RuntimeError("invalid version in wisdom file")
    
   if data.get("key") != key:
       raise RuntimeError(f"invalid key in wisdom file: {key} != {data.get('key')}")

   keys = data.get("tunable_parameters", [])
   if set(keys) != set(params.keys()):
       raise RuntimeError("invalid tunable parameters in wisdom file: " + 
                          f"{list(params.keys())} != {keys}")

   return keys


def _create_header(key: str, param_keys: list):
    return json.dumps({
        "version": WISDOM_VERSION,
        "objective": WISDOM_OBJECTIVE,
        "tunable_parameters": list(param_keys),
        "key": self.key,
    })


def _is_valid_config(config):
    time = config.get("time")
    return isinstance(time, (float, np.float32, np.float64))
