import os
import sys
import argparse
import traceback
import socket
import re
import datetime

KERNEL_LAUNCHER_HOME = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(KERNEL_LAUNCHER_HOME, "python"))
import kernel_launcher as kl


def host_name():
    return socket.gethostname()


def device_name(device=0):
    import pycuda.driver as drv
    drv.init()
    return drv.Device(device).name()


def should_skip_kernel(args, problem):
    if args.conflict != "ignore":
        return False

    device = device_name()
    problem_size = problem.problem_size

    wisdom = kl.read_wisdom(args.output_dir, problem.key, problem.space.params, error_if_missing=False)

    for record in wisdom:
        if record.get("problem_size") == problem_size and \
                record.get("environment", dict()).get("device_name") == device:
            return True

    return False


def tune_kernel(filename, args):
    print(f"parsing file: {filename}")

    problem = kl.load_tuning_problem(filename, data_dir=args.data_dir)

    if should_skip_kernel(args, problem):
        print(f"warning: skipping kernel {problem.key} ({filename}), wisdom already available")
        return

    options = dict(
            iterations=args.iterations,
            verify=args.verify,
            atol=args.atol,
    )

    print(f"host name: {socket.gethostname()}")
    print(f"device name: {device_name()}")
    print(f"kernel name: {problem.key}")
    print(f"problem size: {problem.problem_size}")

    # Run kernel once with default configuration.
    default_config = problem.space.default_config()
    default_params = dict((k, [v]) for k, v in default_config.items())
    results, env = problem.tune(
            default_params,
            strategy="brute_force")

    if args.strategy == "block":
        block_params = dict()

        for k, v in default_config.items():
            block_params[k] = [v]

        for expr in problem.kernel.block_size:
            for var in expr.free_variables():
                block_params[var] = problem.space.params[var]

        before = datetime.datetime.now()

        strategy_options = dict()
        more_results, _ = problem.tune(
            block_params,
            strategy="brute_force",
            strategy_options=strategy_options,
            **options)
        results += more_results

        after = datetime.datetime.now()
        time_remaining = args.time_limit - (after - before).total_seconds()

        if time_remaining > 0:
            best_result = min(results, key=lambda p: p["time"])

            more_params = dict(problem.space.params)
            for expr in problem.kernel.block_size:
                for var in expr.free_variables():
                    more_params[var] = [best_result[var]]

            strategy_options = dict(
                max_fevals=1e99,
                time_limit=time_remaining,
            )
            more_results, _ = problem.tune(
                more_params,
                strategy_options=strategy_options,
                **options)

            results += more_results

    elif args.strategy == "random":
        strategy_options = dict(
            time_limit=args.time_limit,
            fraction=1
        )

        more_results, _ = problem.tune(
            strategy="random_sample",
            strategy_options=strategy_options,
            **options)
        results += more_results

    elif args.strategy == "bayes":
        strategy_options = dict(
            time_limit=args.time_limit,
            max_fevals=1e99,
        )

        more_results, _ = problem.tune(
            strategy="bayes_opt",
            strategy_options=strategy_options,
            **options)
        results += more_results

    else:
        raise ValueError(f"unknown strategy: {args.strategy}")

    best_result = min(results, key=lambda p: p["time"])

    print(f"finished tuning {problem.key}")
    print(f"best configuration: {best_result!r}")

    print("writing wisdom file")
    merge = args.conflict == "combine"
    kl.write_wisdom_for_problem(args.output_dir, problem, results, env,
                                max_results=1, merge_existing_results=merge)


def parse_time(input):
    # float
    match = re.match("^([0-9]+([.][0-9]*)?)$", input)
    if match:
        return float(match[1])

    # mm:ss
    match = re.match("^([0-9]+):([0-9]+)$", input)
    if match:
        return int(match[1]) * 60 + int(match[2])

    # hh:mm:ss
    match = re.match("^([0-9]+):([0-9]+):([0-9]+)$", input)
    if match:
        return int(match[1]) * 60 * 60 + int(match[2]) * 60 + int(match[3])

    raise ValueError(f"failed to parse time: {input}")


def main():
    parser = argparse.ArgumentParser(
            description="Tune given kernel files and store the results in wisdom files")
    parser.add_argument("--strategy", "-s", default="bayes", choices=["block", "bayes", "random"],
                        help="The strategy to use for tuning:\n"
                             " - random: try random configurations until time runs out.\n"
                             " - bayes: use Bayesian optimization to try configurations until time runs out.\n"
                             " - block: brute-force search block sizes and then optimize the remaining parameters.\n")
    parser.add_argument("--time", "-t", type=parse_time, default="15:00", dest="time_limit",
                        help="Maximum time in seconds spend on tuning each kernel.")
    parser.add_argument("--conflict", "-c", default="ignore", choices=["overwrite", "ignore", "combine"],
                        help="What to do when tuning a kernel for which wisdom is already available:\n"
                        " - ignore: skip the specific kernel.\n"
                        " - overwrite: tune the kernel and overwrite the existing wisdom.\n"
                        " - combine: tune the kernel and combine results with the existing wisdom.\n")
    parser.add_argument("--combine", "-a", dest="conflict", action="store_const", const="combine",
                        help="Alias for `--conflict combine`")
    parser.add_argument("--force", "-f", dest="conflict", action="store_const", const="overwrite",
                        help="Alias for `--conflict overwrite`")

    parser.add_argument("--output", "-o", default=".", dest="output_dir",
                        help="Directory where to store resulting wisdom files.")
    parser.add_argument("--data-dir", "-d", default=None,
                        help="Directory where data files (.bin) are located.")

    parser.add_argument("--iterations", "-i", type=int, default=5,
                        help="Number of benchmark iterations for each kernel")
    parser.add_argument("--no-verify", action="store_false", default=True, dest="verify",
                        help="Skip verification if the output of each kernel launch is correct.")
    parser.add_argument("--tolerance", "--atol", dest="atol", type=float,
                        help="Absolute tolerance used for verification as interpreted by `numpy.isclose`.")
    parser.add_argument("files", nargs="*")

    args = parser.parse_args()

    if not os.path.isdir(args.output_dir):
        print(f"error: not a valid directory: {args.output_dir}")
        return

    if not args.files:
        print("error: no files given")
        return

    for file in args.files:
        # Skip binary files if they are included.
        if file.endswith(".bin") or file.endswith(".bin.gz") or os.path.isdir(file):
            print(f"warning: skipping file {file}")
            continue

        try:
            tune_kernel(file, args)
        except Exception as e:
            print(f"error: exception occurred while tuning {file}:")
            traceback.print_exception(type(e), e, e.__traceback__)
            print()


if __name__ == "__main__":
    main()
