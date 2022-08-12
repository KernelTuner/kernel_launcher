import hashlib
import json
import logging
import numpy as np
import os.path
import re

from collections import OrderedDict

logger = logging.getLogger(__name__)

def _type_name_to_dtype(type_name):
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


def KernelDef:
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
        self.compile_flags = _parse_expr_list(obj["compile_flags"])

        self.block_size = _parse_expr_list(obj["block_size"])
        self.grid_size = _parse_expr_list(obj["grid_size"])

        while len(self.block_size) < 3:
            self.block_size.append(_parse_expr(1))

        while len(self.grid_size) < 3:
            self.grid_size.append(_parse_expr(1))

        self.defines = OrderedDict()
        for k, v in entry.get("defines", dict()).items():
            self.defines[k] = _parse_expr(v)


def ConfigDef:
    def __init__(self, obj):
        self.params = OrderedDict()
        self.restrictions = []

        for row in obj["parameters"]:
            self.params[row["name"]] = row["values"]

        for row in obj["restrictions"]:
            self.restrictions.append(_parse_expr(row))


def _parse_scalar_argument(entry):
    dtype = _type_name_to_dtype(entry["type"])
    data = entry["data"]

    if dtype is None:
        # We cannot determine the datatype. Create a custom numpy
        # data type that is just a binary blob of N bytes
        dtype = np.dtype((np.void, len(data)))

    return np.frombuffer(bytes(data), dtype=dtype)[0]


def _parse_array_file(file_name: str, data_dir: str, expect_hash: str, dtype, validate: bool):
    with open(os.path.join(data_dir, file_name), "rb") as handle:
        buf = handle.read()

    if expect_hash is not None and validate:
        got_hash = hashlib.new("sha1", buf, usedforsecurity=False).hexdigest()

    if got_hash != expect_hash:
        raise Runtimerror(f"invalid file hash for {file_name}, data was corrupted")

    return np.frombuffer(buf, dtype=dtype)


def _parse_array_argument(entry: dict, data_dir: str, validate_checksum: bool):
    type_name = entry["type"]
    dtype = None

    if type_name.endswith("*"): # is a pointer
        dtype = _type_name_to_dtype(type_name[:-1])

    if dtype is None:
        logger.warning(f"unknown type \"{type_name}\", falling back to byte array") 
        dtype = np.byte

    arg = _parse_array_file(entry["file"], data_dir, entry.get("file_hash"), dtype, validate_checksum)

    if "reference_file" in entry:
        answer = _parse_array_file(entry["reference_file"], data_dir, entry.get("reference_hash"),
                                   dtype, validate_checksum)
    else:
        answer = None

    return arg, answer


class TuningProblem:
    def __init__(self, obj, data_dir: str, validate_checksum: bool=True):
        self.space = ConfigSpace(data["config_space"])
        self.kernel = KernelDef(data["kernel"])

        self.environment = data.get("environment", dict())
        self.key = data["key"]
        self.problem_size = data["problem_size"]
        self.args = []
        self.answers = []

        for entry in data["arguments"]:
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


def load_tuning_problem(filename: str, data_dir=None, **kwargs): TuningProblem:
    if data_dir is None:
        data_dir = os.path.dirname(os.path.abspath(filename))

    with open(filename, encoding="utf8") as handle:
        return TuningProblem(json.load(handle), data_dir, **kwargs)


class Expr:
    def __call__(self, config):
        return self.evaluate(config)
    
    def evaluate(self, config):
        raise NotImplemented

    def visit_children(self, fun):
        raise NotImplemented

    def resolve_problem(self, problem_size):
        return self.visit_children(lambda e: e.resolve_problem(problem_size))


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
        return repr(self.value)

    def visit_children(self, fun):
        return self

class BinaryExpr(Expr):
    def __init__(self, op, lhs, rhs):
        self.op = op
        self.lhs = lhs
        self.rhs = rhs

    def evaluate(self, config):
        lhs = self.lhs(config)
        rhs = self.rhs(config)

        if op == "+":
            return lhs + rhs
        if op == "-":
            return lhs - rhs
        if op == "*":
            return lhs * rhs
        if op == "/":
            if isinstance(lhs, int) and isinstance(rhs, int):
                # To get the same behavior for ints as in C, we perform regular FP division
                # and then truncate the result. This is different from Python's `//` operator
                # which will always round down (not truncate).
                return int(lhs / rhs)
            else:
                return lhs / rhs
        if op == "%":
            if isinstance(lhs, int) and isinstance(rhs, int):
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
        return BinaryExpr(self.op, fun(self.lhs), fun(self.rhs))

    def __repr__(self):
        return f"({self.lhs!r} {self.op} {self.rhs!r})"


class UnaryExpr(Expr):
    def __init__(self, op, arg):
        self.op = op
        self.arg = arg

    def evaluate(self, config):
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

    def resolve_problem(self, problem_size):
        return ValueExpr(problem_size[self.axis])

    
class SelectExpr(Expr):
    def __init__(self, condition, options):
        self.condition = condition
        self.options = options

    def __repr__(self):
        return f"select({self.condition!r}, {self.options!r})"

    def evaluate(self, config):
        index = self.condition.evaluate(config):

        if isinstance(index, bool):
            index = int(index)

        if not isinstance(index, int) or index < 0 or index >= self.options:
            raise RuntimeError(f"expression must yield an integer in range 0..{len(self.options)}: {self}")
        
        return self.options[index].evaluate(config)

    def visit_children(self, fun):
        return SelectExpr(fun(self.condition), [fun(a) for a in self.options])


def _parse_expr(entry):
    if isinstance(entry, (int, str, float)) or \
            entry is None:
        return ValueExpr(entry)

    if not isinstance(entry, dict):
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

    elif len(args) == 1:
        return UnaryExpr(op, args[0])

    elif len(args) == 2:
        return BinaryExpr(op, args[0], args[1])

    else:
        raise ValueError(f"invalid operator: {op}")


def _parse_expr_list(entries):
    return [_parse_expr(e) for e in entries]
