import cupy
import gzip
import hashlib
import json
import kernel_tuner
import logging
import numpy as np
import os.path
import re

from typing import List
from collections import OrderedDict

logger = logging.getLogger(__name__)


def _type_name_to_dtype(type_name):
    """
    Converts C type name to numpy type. For example, ``"int"`` => ``np.intc``
    """
    NUMPY_TYPES = {
        "float": np.single,
        "double": np.double,
        "char": np.byte,
        "signed char": np.byte,
        "unsigned char": np.ubyte,
        "uchar": np.ubyte,
        "int": np.intc,
        "signed int": np.intc,
        "unsigned int": np.uintc,
        "uint": np.uintc,
        "short": np.short,
        "signed short": np.short,
        "unsigned short": np.ushort,
        "ushort": np.ushort,
        "long": np.int_,
        "signed long": np.int_,
        "unsigned long": np.uint,
        "ulong": np.uint,
        "long long": np.longlong,
        "signed long long": np.longlong,
        "unsigned long long": np.ulonglong,
        "bool": np.bool_,
        "int8_t": np.int8,
        "int16_t": np.int16,
        "int32_t": np.int32,
        "int64_t": np.int64,
        "uint8_t": np.uint8,
        "uint16_t": np.uint16,
        "uint32_t": np.uint32,
        "uint64_t": np.uint64,
    }

    vector = 1
    match = re.match("([a-z]+)([1-9][0-9]*)", type_name)
    if match:
        type_name = match.group(1)
        vector = int(match.group(2))

    # Remove trailing or leading `const`
    key = re.sub(r"(^(const)?\s+)|(\s+(const?)$)", "", type_name)
    dtype = NUMPY_TYPES.get(key)

    if dtype is not None and vector > 1:
        dtype = np.dtype((dtype, vector))

    return dtype


class KernelDef:
    def __init__(self, obj):
        if obj.get("file"):
            self.file = obj["file"]
            self.source = None
        elif obj.get("source"):
            self.file = None
            self.source = obj["source"]
        else:
            raise ValueError("kernel has no source code")

        self.name = obj["name"]
        self.shared_memory = _parse_expr(obj["shared_memory"])
        self.template_args = _parse_expr_list(obj["template_args"])
        self.compiler_options = _parse_expr_list(obj["compile_flags"])

        self.block_size = _parse_expr_list(obj["block_size"])
        self.grid_size = _parse_expr_list(obj["grid_size"])

        while len(self.block_size) < 3:
            self.block_size.append(_parse_expr(1))

        while len(self.grid_size) < 3:
            self.grid_size.append(_parse_expr(1))

        self.defines = OrderedDict()
        for k, v in obj.get("defines", dict()).items():
            self.defines[k] = _parse_expr(v)

    def generate_name(self):
        name = self.name
        targs = self.template_args

        if not targs:
            return name

        vargs = []
        for targ in targs:
            if isinstance(targ, ValueExpr):
                vargs.append(str(targ.value))
            elif isinstance(targ, ParamExpr):
                vargs.append(targ.name)
            else:
                raise ValueError(f"invalid template argument expression: {targ!r}")

        return f"{name}<{','.join(vargs)}>"

    def generate_source(self, working_dir):
        if self.source is not None:
            source = self.source
        else:
            path = os.path.join(working_dir, self.file)
            with open(path) as f:
                source = f.read()

        return source


class ConfigSpace:
    def __init__(self, obj):
        self.params = OrderedDict()
        self.restrictions = []
        self.defaults = dict()

        for row in obj["parameters"]:
            self.params[row["name"]] = row["values"]

            if "default" in row:
                default_value = row["default"]
            else:
                default_value = row["values"][0]

            self.defaults[row["name"]] = default_value

        for row in obj["restrictions"]:
            self.restrictions.append(_parse_expr(row))

    def default_config(self):
        return dict(self.defaults)

    def is_valid(self, config):
        return all(e(config) for e in self.restrictions)


def _parse_scalar_argument(entry):
    dtype = _type_name_to_dtype(entry["type"])
    data = entry["data"]

    if dtype is None:
        # We cannot determine the datatype. Create a custom numpy
        # data type that is just a binary blob of N bytes
        dtype = np.dtype((np.void, len(data)))

    return np.frombuffer(bytes(data), dtype=dtype)[0]


def _parse_array_file(file_name: str, data_dir: str, expect_hash: str, dtype, validate: bool):
    file_path = os.path.join(data_dir, file_name)

    if file_name.endswith(".gz"):
        with gzip.open(file_path, "rb") as handle:
            buf = handle.read()
    else:
        with open(file_path, "rb") as handle:
            buf = handle.read()

    if expect_hash is not None and validate:
        got_hash = hashlib.new("sha1", buf, usedforsecurity=False).hexdigest()

        if got_hash != expect_hash:
            raise RuntimeError(f"invalid file hash for {file_name}, data was corrupted")

    return np.frombuffer(buf, dtype=dtype)


def _parse_array_argument(entry: dict, data_dir: str, validate_checksum: bool):
    type_name = entry["type"]
    dtype = None

    if type_name.endswith("*"):  # is a pointer
        dtype = _type_name_to_dtype(type_name[:-1])

    if dtype is None:
        logger.warning(f"unknown type \"{type_name}\", falling back to byte array")
        dtype = np.byte

    arg = _parse_array_file(entry["file"], data_dir, entry.get("file_hash"),
                            dtype, validate_checksum)

    if "reference_file" in entry:
        answer = _parse_array_file(entry["reference_file"], data_dir, entry.get("reference_hash"),
                                   dtype, validate_checksum)
    else:
        answer = None

    return arg, answer


class TuningProblem:
    def __init__(self, obj, data_dir: str, validate_checksum: bool = True):
        self.space = ConfigSpace(obj["config_space"])
        self.kernel = KernelDef(obj["kernel"])

        self.environment = obj.get("environment", dict())
        self.key = obj["key"]
        self.problem_size = obj["problem_size"]
        self.args = []
        self.answers = []

        for entry in obj["arguments"]:
            kind = entry.get("kind")

            if kind == "scalar":
                arg = _parse_scalar_argument(entry)
                answer = None
            elif kind == "array":
                arg, answer = _parse_array_argument(entry, data_dir, validate_checksum)
            else:
                raise ValueError(f"invalid argument kind {kind}")

            self.args.append(arg)
            self.answers.append(answer)

    def _tune_options(self, working_dir=None, lang="cupy", compiler_options=None, defines=None, device=0, **kwargs):
        if working_dir is None:
            working_dir = os.getcwd()

        extra_params = dict()
        block_size_names = []
        context = dict(problem_size=self.problem_size, device=cupy.cuda.Device(device))

        for axis, expr in zip("XYZ", self.kernel.block_size):
            expr = expr.resolve(**context)

            if isinstance(expr, ParamExpr):
                block_size_names.append(expr.name)
            elif isinstance(expr, ValueExpr):
                key = f"__KERNEL_TUNER_PROXY_BLOCK_SIZE_{axis}"
                extra_params[key] = expr.value
                block_size_names.append(key)
            else:
                raise ValueError(f"invalid block size expression: {expr!r}")

        grid_exprs = [e.resolve(**context) for e in self.kernel.grid_size]

        def grid_size(config):
            return [e(config) for e in grid_exprs]

        if not compiler_options:
            compiler_options = []
        else:
            compiler_options = list(compiler_options)

        for expr in self.kernel.compiler_options:
            if isinstance(expr, ValueExpr):
                compiler_options.append(str(expr.value))
            else:
                raise ValueError(f"invalid compiler options expression: {expr!r}")

        all_defines = self.kernel.defines

        if defines:
            all_defines = OrderedDict(all_defines)  # make copy

            for key, value in defines.items():
                all_defines[key] = value

        restrictions = [e.resolve(**context) for e in self.space.restrictions]

        options = dict(
                kernel_name=self.kernel.generate_name(),
                kernel_source=self.kernel.generate_source(working_dir),
                arguments=self.args,
                problem_size=grid_size,
                restrictions=lambda config: all(f(config) for f in restrictions),
                defines=all_defines,
                compiler_options=compiler_options,
                block_size_names=block_size_names,
                grid_div_x=[],
                grid_div_y=[],
                grid_div_z=[],
                lang=lang,
                device=device,
                **kwargs)

        os.chdir(working_dir)
        return extra_params, options

    def run(self, config=None, arguments=None, *, check_restrictions=True, **kwargs):
        """Run this kernel once using ``kernel_tuner.run_kernel`` and return the results

        :param config: The configuration to compile. If ``None``, the default configuration is used
        :param arguments: The arguments to use. If ``None``, the default arguments for this problem size are used
        :param check_restrictions: Check if the given configuration meets the restrictions on the search space
        :param kwargs: Additional keyword arguments passed to ``kernel_tuner.run_kernel``
        :return: Result of ``kernel_tuner.run_kernel``
        """
        if not config:
            config = self.space.default_config()

        if check_restrictions and self.space.is_valid(config):
            raise RuntimeError("configuration fails restrictions: {config!r}")

        extra_params, options = self._tune_options(**kwargs)
        config = dict(config) | extra_params

        if arguments is not None:
            options["arguments"] = arguments

        return kernel_tuner.run_kernel(params=config, **options)

    def tune(self, params=None, **kwargs):
        """Execute ``kernel_tuner.tune_kernel`` for this tuning problem.

        :param params: Tunable parameters passed to ``tune_kernel``. If ``None``, the parameters as defined by this
                       tuning problem are used. This argument is useful to overwrite the tunable parameters, for
                       instance, if the search space is large and not all parameters need to be tuned.
        :param kwargs: Additional keyword arguments passed to ``tune_kernel``
        """
        if params is None:
            params = self.space.params

        extra_params, options = self._tune_options(**kwargs)

        params = OrderedDict(params)  # Copy parameters since it can be modified below
        for k, v in extra_params.items():
            params[k] = [v]

        verify = options.pop("verify", None)
        if verify is True:
            verify = _fancy_verify
            answer = self.answers
        elif verify is False:
            verify = None
            answer = None
        else:
            answer = self.answers

        strategy = options.pop("strategy", None)
        if strategy is None:
            total_configs = np.prod([len(v) for v in params.values()])
            strategy = "brute_force" if total_configs < 100 else "bayes_opt"

        return kernel_tuner.tune_kernel(
                tune_params=params,
                strategy=strategy,
                answer=answer,
                verify=verify,
                **options)


def _fancy_verify(answers, outputs, *, atol=None):
    INTEGRAL_DTYPES = [np.int8, np.int16, np.int32, np.int64,
                       np.uint8, np.uint16, np.uint32, np.uint64]
    FLOATING_DTYPES = [np.float16, np.float32, np.float64]
    PRINT_TOP_VALUES = 25
    DEFAULT_ATOL = 1e-8

    # np.byte represents an unknown data type (it is usually an alias for np.int8 unfortunately)
    INTEGRAL_DTYPES.remove(np.byte)

    if atol is None:
        atol = DEFAULT_ATOL

    is_valid = True

    for index, (output, expected) in enumerate(zip(outputs, answers)):
        if output is None or expected is None:
            continue

        if output.dtype != expected.dtype or output.shape != expected.shape:
            raise RuntimeError(f"arrays data type or shape do not match: {output} and {expected}")

        if output.dtype in INTEGRAL_DTYPES:
            matches = output == expected
        elif output.dtype in FLOATING_DTYPES:
            matches = np.isclose(output, expected, atol=atol, equal_nan=True)
        else:
            matches = True  # unknown data type, skip

        # All match, great!
        if np.all(matches):
            continue

        # Overall result is invalid
        is_valid = False

        indices = np.where(~matches)[0]
        nerrors = len(indices)

        # Should indices be sorted?
        # indices = indices[np.argsort(errors[indices], kind="stable")][::-1]

        percentage = nerrors / len(output) * 100
        print(f"argument {index + 1} fails validation: {nerrors} incorrect values" +
              f"({percentage:.5}%)")

        errors = np.abs(output - expected)

        for index in indices[:PRINT_TOP_VALUES]:
            print(f" * at index {index}: {output[index]} != {expected[index]} " +
                  f"(error: {errors[index]})")

        if nerrors > PRINT_TOP_VALUES:
            print(f" * ({nerrors - PRINT_TOP_VALUES} more entries have been omitted)")

    return is_valid


class Expr:
    def __call__(self, config):
        return self.evaluate(config)

    def evaluate(self, config):
        raise NotImplementedError

    def visit_children(self, fun):
        raise NotImplementedError

    def resolve(self, **kwargs):
        return self.visit_children(lambda e: e.resolve(**kwargs))

    def free_variables(self):
        result = set()

        def f(expr):
            result.update(expr.free_variables())

        self.visit_children(f)
        return result


class ValueExpr(Expr):
    def __init__(self, value):
        self.value = value

    def evaluate(self, config):
        return self.value

    def __repr__(self):
        return repr(self.value)

    def visit_children(self, fun):
        return self


class ParamExpr(Expr):
    def __init__(self, name):
        self.name = name

    def evaluate(self, config):
        return config[self.name]

    def __repr__(self):
        return repr(self.name)

    def free_variables(self):
        return [self.name]

    def visit_children(self, fun):
        return self


def is_int_like(v):
    return isinstance(v, (int, np.integer, bool, np.bool_))


class BinaryExpr(Expr):
    def __init__(self, op, lhs, rhs):
        self.op = op
        self.lhs = lhs
        self.rhs = rhs

    def evaluate(self, config):
        op = self.op
        lhs = self.lhs(config)
        rhs = self.rhs(config)

        if op == "+":
            return lhs + rhs
        if op == "-":
            return lhs - rhs
        if op == "*":
            return lhs * rhs
        if op == "/":
            if is_int_like(lhs) and is_int_like(rhs):
                # To get the same behavior for ints as in C, we perform regular FP division
                # and then truncate the result. This is different from Python's `//` operator
                # which will always round down (not truncate).
                return int(lhs / rhs)
            else:
                return lhs / rhs
        if op == "%":
            if is_int_like(lhs) and is_int_like(rhs):
                # see above
                return lhs - int(lhs / rhs) * rhs
            else:
                return lhs % rhs
        if op == "==":
            return lhs == rhs
        if op == "!=":
            return lhs != rhs
        if op == "<":
            return lhs < rhs
        if op == ">":
            return lhs > rhs
        if op == "<=":
            return lhs <= rhs
        if op == ">=":
            return lhs >= rhs
        if op == "&&" or op == "and":
            return bool(lhs) and bool(rhs)
        if op == "||" or op == "or":
            return bool(lhs) or bool(rhs)

        raise ValueError(f"invalid binary operator: {self.op}")

    def visit_children(self, fun):
        lhs = fun(self.lhs)
        rhs = fun(self.rhs)
        expr = BinaryExpr(self.op, lhs, rhs)

        # If both sides are values, we can evaluate the result now
        if isinstance(lhs, ValueExpr) and isinstance(rhs, ValueExpr):
            return ValueExpr(expr.evaluate(dict()))

        return expr

    def __repr__(self):
        return f"({self.lhs!r} {self.op} {self.rhs!r})"


class UnaryExpr(Expr):
    def __init__(self, op, arg):
        self.op = op
        self.arg = arg

    def evaluate(self, config):
        op = self.op
        arg = self.arg.evaluate(config)

        if op == "+":
            return arg
        if op == "-":
            return -arg
        if op == "!" or op == "not":
            return not bool(arg)

        raise ValueError(f"invalid unary operator: {op}")

    def __repr__(self):
        return f"({self.op} {self.arg!r})"

    def visit_children(self, fun):
        return UnaryExpr(self.op, fun(self.arg))


class ProblemExpr(Expr):
    def __init__(self, axis):
        self.axis = axis

    def __repr__(self):
        return f"problem_size_{self.axis}"

    def evaluate(self, config):
        raise ValueError("expression cannot be problem dependent")

    def visit_children(self, fun):
        return self

    def resolve(self, problem_size, **kwargs):
        return ValueExpr(problem_size[self.axis])


class DeviceAttributeExpr(Expr):
    # Map cuda.h names to cupy names
    NAME_MAPPING = dict([
        ('MAX_THREADS_PER_BLOCK', 'MaxThreadsPerBlock'),
        ('MAX_BLOCK_DIM_X', 'MaxBlockDimX'),
        ('MAX_BLOCK_DIM_Y', 'MaxBlockDimY'),
        ('MAX_BLOCK_DIM_Z', 'MaxBlockDimZ'),
        ('MAX_GRID_DIM_X', 'MaxGridDimX'),
        ('MAX_GRID_DIM_Y', 'MaxGridDimY'),
        ('MAX_GRID_DIM_Z', 'MaxGridDimZ'),
        ('MAX_SHARED_MEMORY_PER_BLOCK', 'MaxSharedMemoryPerBlock'),
        ('WARP_SIZE', 'WarpSize'),
        ('MAX_REGISTERS_PER_BLOCK', 'MaxRegistersPerBlock'),
        ('MULTIPROCESSOR_COUNT', 'MultiProcessorCount'),
        ('MAX_THREADS_PER_MULTIPROCESSOR', 'MaxThreadsPerMultiProcessor'),
        ('MAX_SHARED_MEMORY_PER_MULTIPROCESSOR', 'MaxSharedMemoryPerMultiprocessor'),
        ('MAX_REGISTERS_PER_MULTIPROCESSOR', 'MaxRegistersPerMultiprocessor'),
    ])

    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return f"device_attribute({self.name!r})"

    def evaluate(self, config):
        raise ValueError("expression cannot be device dependent")

    def visit_children(self, fun):
        return self

    def resolve(self, device, **kwargs):
        internal_name = DeviceAttributeExpr.NAME_MAPPING[self.name]
        value = device.attributes[internal_name]
        return ValueExpr(value)


class SelectExpr(Expr):
    def __init__(self, condition, options):
        self.condition = condition
        self.options = options

    def __repr__(self):
        return f"select({self.condition!r}, {self.options!r})"

    def evaluate(self, config):
        index = self.condition.evaluate(config)

        if not is_int_like(index) or index < 0 or index >= len(self.options):
            raise RuntimeError("expression must yield an integer in " +
                               f"range 0..{len(self.options)}: {self}")

        return self.options[int(index)].evaluate(config)

    def visit_children(self, fun):
        return SelectExpr(fun(self.condition), [fun(a) for a in self.options])


def _parse_expr(entry) -> Expr:
    # literal int, str or float becomes ValueExpr.
    if isinstance(entry, (int, str, float)) or \
            entry is None:
        return ValueExpr(entry)

    # Otherwise it must be an operator expression
    if not isinstance(entry, dict) or "operator" not in entry:
        raise ValueError(f"invalid expression: {entry!r}")

    op = entry["operator"]
    args = _parse_expr_list(entry.get("operands", []))

    if op == "parameter":
        if len(args) != 1 and not isinstance(args[0], ValueExpr):
            raise ValueError(f"invalid expression: {entry!r}")
        return ParamExpr(args[0].value)

    elif op == "select":
        if len(args) < 1:
            raise ValueError(f"invalid expression: {entry!r}")
        return SelectExpr(args[0], args[1:])

    elif op == "problem_size":
        if len(args) != 1 or not isinstance(args[0], ValueExpr):
            raise ValueError(f"invalid expression: {entry!r}")
        return ProblemExpr(args[0].value)

    elif op == "device_attribute":
        if len(args) != 1 or not isinstance(args[0], ValueExpr):
            raise ValueError(f"invalid expression: {entry!r}")
        return DeviceAttributeExpr(args[0].value)

    elif len(args) == 1:
        return UnaryExpr(op, args[0])

    elif len(args) == 2:
        return BinaryExpr(op, args[0], args[1])

    else:
        raise ValueError(f"invalid operator: {op}")


def _parse_expr_list(entries) -> List[Expr]:
    return [_parse_expr(e) for e in entries]


def load_tuning_problem(filename: str, data_dir=None, **kwargs) -> TuningProblem:
    """
    Load a tuning problem from a file generated by Kernel Launcher.

    :param filename: Path to the tuning file.
    :param data_dir: Directory were the data files are located. If ``None``, it is assumed to be the same directory as
                     ``filename``.
    :param kwargs: Additional keyword arguments passed to ``TuningProblem``.
    :return: The tuning problem.
    """
    if data_dir is None:
        data_dir = os.path.dirname(os.path.abspath(filename))

    with open(filename, encoding="utf8") as handle:
        return TuningProblem(json.load(handle), data_dir, **kwargs)
