from __future__ import annotations

import ast
import math
import re
import textwrap
from dataclasses import dataclass, field

from gapdeleg.split.template import PLACEHOLDER_RE

MAX_STEPS = 200_000
MAX_LOOP = 10_000
MAX_DEPTH = 30
MAX_POW = 64
MAX_ABS = 1e18
EPS = 1e-9

NON_NUMERIC_TYPES = {"PERSON", "ORGANIZATION", "LOCATION", "ADDRESS", "EMAIL_ADDRESS", "PHONE_NUMBER",
                     "ID_NUMBER", "ACCOUNT_NUMBER", "URL", "PERSONAL_FACT", "NAME", "MEDICAL_CONDITION"}


class StaticError(ValueError):
    pass


class RunError(ValueError):
    pass


class NotANumber:

    def __init__(self, name: str):
        self.name = name

    def __repr__(self):
        return f"NotANumber({self.name})"


def _round(x, ndigits=None):
    q = 10 ** int(ndigits or 0)
    r = math.floor(abs(x) * q + 0.5 + EPS) / q
    r = r if x >= 0 else -r
    return int(r) if not ndigits else r


def _floor(x):
    return math.floor(x + EPS)


def _ceil(x):
    return math.ceil(x - EPS)


def _int(x):
    return int(x + EPS) if x >= 0 else int(x - EPS)


def _sqrt(x):
    if x < 0:
        raise RunError("square root of a negative number")
    return math.sqrt(x)


def _minmax(f):
    def g(*args):
        vals = args[0] if len(args) == 1 and isinstance(args[0], list) else list(args)
        if not vals:
            raise RunError(f"{f.__name__}() of nothing")
        for v in vals:
            if not _num(v):
                raise RunError(f"{f.__name__}() of something that is not a number")
        return f(vals)
    return g


def _sum(xs, start=0):
    if not isinstance(xs, list) or not all(_num(x) for x in xs):
        raise RunError("sum() needs a list of numbers")
    return sum(xs, start)


def _len(xs):
    if not isinstance(xs, list):
        raise RunError("len() needs a list")
    return len(xs)


FUNCS = {"round": _round, "min": _minmax(min), "max": _minmax(max), "abs": abs, "int": _int, "float": float,
         "sum": _sum, "len": _len, "floor": _floor, "ceil": _ceil, "sqrt": _sqrt, "trunc": _int, "fabs": abs}
MATH_FUNCS = {"floor", "ceil", "sqrt", "trunc", "fabs"}
MATH_CONSTS = {"pi": math.pi}

_BINOPS = {ast.Add: lambda a, b: a + b, ast.Sub: lambda a, b: a - b, ast.Mult: lambda a, b: a * b,
           ast.Div: lambda a, b: a / b, ast.FloorDiv: lambda a, b: _floor(a / b), ast.Mod: lambda a, b: a % b}
_CMPS = {ast.Eq: lambda a, b: abs(a - b) < EPS, ast.NotEq: lambda a, b: abs(a - b) >= EPS,
         ast.Lt: lambda a, b: a < b - EPS, ast.LtE: lambda a, b: a <= b + EPS,
         ast.Gt: lambda a, b: a > b + EPS, ast.GtE: lambda a, b: a >= b - EPS}

_ALLOWED = (ast.Module, ast.FunctionDef, ast.Return, ast.Assign, ast.AugAssign, ast.If, ast.For,
            ast.While, ast.Break, ast.Continue, ast.Pass, ast.Expr, ast.Import, ast.ImportFrom, ast.alias,
            ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare, ast.IfExp, ast.Call, ast.Name, ast.Constant,
            ast.List, ast.Tuple, ast.Attribute, ast.keyword, ast.arguments, ast.arg, ast.Load, ast.Store,
            ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow, ast.USub, ast.UAdd, ast.Not,
            ast.And, ast.Or, ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE)

_NAMES = {ast.Subscript: "indexing (x[i])", ast.ListComp: "list comprehensions", ast.GeneratorExp:
          "generator expressions", ast.Lambda: "lambda", ast.With: "with", ast.Try: "try", ast.ClassDef: "classes",
          ast.Global: "global", ast.Nonlocal: "nonlocal", ast.Dict: "dictionaries", ast.Set: "sets",
          ast.JoinedStr: "f-strings", ast.Starred: "*args", ast.BitXor: "^ (use ** for powers)", ast.BitAnd: "&",
          ast.BitOr: "|", ast.MatMult: "@", ast.Delete: "del", ast.Assert: "assert", ast.Raise: "raise",
          ast.Yield: "yield", ast.Await: "await", ast.AsyncFunctionDef: "async def", ast.In: "in",
          ast.NotIn: "not in", ast.Is: "is", ast.IsNot: "is not"}


def _num(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def preprocess(code: str) -> str:
    code = textwrap.dedent(code.replace("\t", "    ")).strip("\n")
    code = PLACEHOLDER_RE.sub(lambda m: f"{m.group(1)}_{m.group(2)}", code)
    code = code.replace("×", "*").replace("÷", "/").replace("−", "-").replace("–", "-")
    return re.sub(r"\$(?=\d)", "", code)


@dataclass
class Program:
    code: str
    tree: ast.Module
    funcs: dict
    entry: str | None
    mode: str
    call: ast.Call | None = None
    params: list[str] = field(default_factory=list)
    used: list[str] = field(default_factory=list)
    from_math: dict = field(default_factory=dict)

    def run(self, values: dict) -> float:
        it = _Interp(self.funcs, self.from_math)
        genv = dict(values)
        try:
            if self.mode == "params":
                it.module(self.tree.body, genv, skip_user_calls=True)
                out = it.call_user(self.entry, [], {}, genv, by_name=self.params)
            elif self.mode == "call":
                out = it.module(self.tree.body, genv, stop_at=self.call)
            else:
                it.module(self.tree.body, genv)
                key = next((k for k in ("answer", "Answer", "ANSWER", "result", "total") if k in genv
                            and k not in values), None) or it.last_assigned
                if key is None:
                    raise RunError("the program assigns nothing")
                out = genv[key]
        except RecursionError:
            raise RunError("recursion too deep") from None
        except ZeroDivisionError:
            raise RunError("division by zero") from None
        except (OverflowError, ArithmeticError) as e:
            raise RunError(f"arithmetic error: {e}") from None
        except (TypeError, ValueError) as e:
            if isinstance(e, (RunError, StaticError)):
                raise
            raise RunError(f"{type(e).__name__}: {e}") from None
        if isinstance(out, NotANumber):
            raise RunError(f"{out.name} is not a number")
        if not _num(out) or not math.isfinite(out):
            raise RunError("the result is not a number")
        return float(out)


def compile_program(code: str, placeholders: dict[str, str] | None = None) -> Program:
    names = {ph[1:-1]: ph[1:-1].rsplit("_", 1)[0] for ph in (placeholders or {})}
    code = preprocess(code)
    if not code.strip():
        raise StaticError("the program is empty")
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        raise StaticError(f"syntax error: {e.msg}") from None
    user_funcs = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    tree.body = _clean(tree.body, user_funcs, set(names))
    ast.fix_missing_locations(tree)
    math_alias, from_math = _check_imports(tree)
    defined = set(names) | set(FUNCS) | set(MATH_CONSTS) | math_alias | set(from_math) | {"__binding__", "range"}
    funcs = {}
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED):
            raise StaticError(f"{_NAMES.get(type(node), type(node).__name__)} not allowed")
        if isinstance(node, ast.Constant) and not (_num(node.value) or isinstance(node.value, bool)):
            raise StaticError("strings are not allowed; return a number" if isinstance(node.value, str)
                              else f"the constant {node.value!r} is not allowed")
        if isinstance(node, ast.FunctionDef):
            a = node.args
            if node.decorator_list or a.vararg or a.kwarg or a.kwonlyargs or a.posonlyargs:
                raise StaticError(f"def {node.name}: decorators, *args, **kwargs and keyword-only arguments "
                                  f"are not allowed")
            funcs[node.name] = node
            defined |= {node.name} | {x.arg for x in a.args}
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            defined.add(node.id)
    for node in ast.walk(tree):
        _check_node(node, funcs, math_alias, from_math)
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            if node.id not in defined:
                raise StaticError(f"name `{node.id}` is not defined")
            if node.id in names and names[node.id] in NON_NUMERIC_TYPES:
                raise StaticError(f"`{node.id}` stands for a name, an employer or an account, not a number")
    used = sorted({n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and n.id in names})
    entry = "solve" if "solve" in funcs else ([s.name for s in tree.body if isinstance(s, ast.FunctionDef)]
                                              or [None])[-1]
    if entry is None:
        if not any(isinstance(s, (ast.Assign, ast.AugAssign)) for s in tree.body):
            raise StaticError("no `def solve(...)` and no assignment to run")
        return Program(code, tree, funcs, None, "script", used=used, from_math=from_math)
    fn = funcs[entry]
    args = [x.arg for x in fn.args.args]
    required = args[:len(args) - len(fn.args.defaults)]
    loads = {n.id for n in ast.walk(fn) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
    if all(x in names or x not in loads for x in required):
        return Program(code, tree, funcs, entry, "params", params=[x for x in args if x in names], used=used,
                       from_math=from_math)
    call = _module_call(tree, entry, set(names))
    if call is None:
        bad = next(x for x in required if x not in names and x in loads)
        raise StaticError(f"the parameter `{bad}` of {entry}() is not a placeholder: name each parameter after "
                          f"the placeholder it stands for, like NUM_1")
    return Program(code, tree, funcs, entry, "call", call=call, used=used, from_math=from_math)


def _is_print(node) -> bool:
    return isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "print"


def _mentions_main(test) -> bool:
    return any(isinstance(n, ast.Name) and n.id == "__name__" for n in ast.walk(test))


def _binding_call(node, user_funcs: set, names: set):
    for n in ast.walk(node):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in user_funcs and any(
                isinstance(m, ast.Name) and m.id in names for a in n.args for m in ast.walk(a)):
            return n
    return None


def _clean(stmts: list, user_funcs: set, names: set) -> list:
    out = []
    for s in stmts:
        if isinstance(s, ast.If) and _mentions_main(s.test):
            out += _clean(s.body, user_funcs, names)
            continue
        if isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant):
            continue
        if isinstance(s, ast.Expr) and _is_print(s.value):
            call = _binding_call(s, user_funcs, names)
            if call is not None:
                out.append(ast.copy_location(ast.Assign([ast.Name("__binding__", ast.Store())], call), s))
            continue
        if isinstance(s, ast.AnnAssign):
            if s.value is None or not isinstance(s.target, ast.Name):
                continue
            s = ast.copy_location(ast.Assign([s.target], s.value), s)
        if isinstance(s, ast.FunctionDef):
            s.returns = None
            for a in s.args.args:
                a.annotation = None
        for fld in ("body", "orelse"):
            body = getattr(s, fld, None)
            if isinstance(body, list) and body:
                setattr(s, fld, _clean(body, user_funcs, names) or [ast.Pass()])
        out.append(s)
    return out


def _check_imports(tree) -> tuple[set, dict]:
    math_alias, from_math = {"math"}, {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name != "math":
                    raise StaticError(f"import {a.name}: only `import math` is allowed")
                math_alias.add(a.asname or "math")
        elif isinstance(node, ast.ImportFrom):
            if node.module != "math":
                raise StaticError(f"from {node.module} import: only math is allowed")
            for a in node.names:
                if a.name not in MATH_FUNCS | set(MATH_CONSTS):
                    raise StaticError(f"math.{a.name} is not allowed")
                from_math[a.asname or a.name] = a.name
    return math_alias, from_math


def _check_node(node, funcs, math_alias, from_math):
    if isinstance(node, ast.Attribute):
        if not (isinstance(node.value, ast.Name) and node.value.id in math_alias
                and node.attr in MATH_FUNCS | set(MATH_CONSTS)):
            raise StaticError(f"the attribute .{node.attr} is not allowed (only math.floor, math.ceil, math.sqrt, "
                              f"math.trunc)")
    elif isinstance(node, ast.Call):
        f = node.func
        ok = (isinstance(f, ast.Name) and (f.id in FUNCS or f.id in funcs or f.id in from_math
                                           or f.id == "range")) \
            or (isinstance(f, ast.Attribute) and f.attr in MATH_FUNCS)
        if not ok:
            name = f.id if isinstance(f, ast.Name) else ast.unparse(f)
            raise StaticError(f"calling {name}() is not allowed")
        if node.keywords and not (isinstance(f, ast.Name) and f.id in funcs) and not (
                isinstance(f, ast.Name) and f.id == "round" and all(k.arg == "ndigits" for k in node.keywords)):
            raise StaticError("keyword arguments are allowed only for round(x, ndigits=...) and own functions")
        if isinstance(f, ast.Name) and f.id == "range" and not 1 <= len(node.args) <= 3:
            raise StaticError("range() needs a stop, and at most a start and a step")
    elif isinstance(node, ast.For):
        if node.orelse:
            raise StaticError("for ... else is not allowed")
        it = node.iter
        if not ((isinstance(it, ast.Call) and isinstance(it.func, ast.Name) and it.func.id == "range")
                or isinstance(it, (ast.List, ast.Tuple))):
            raise StaticError("a for loop may only go over range(...) or a written list")
        if not (isinstance(node.target, ast.Name) or (isinstance(node.target, ast.Tuple)
                                                      and all(isinstance(e, ast.Name) for e in node.target.elts))):
            raise StaticError("a for loop needs a plain name")
    elif isinstance(node, ast.While) and node.orelse:
        raise StaticError("while ... else is not allowed")
    elif isinstance(node, (ast.Assign, ast.AugAssign)):
        for t in (node.targets if isinstance(node, ast.Assign) else [node.target]):
            if not (isinstance(t, ast.Name) or (isinstance(t, ast.Tuple) and not isinstance(node, ast.AugAssign)
                                                and all(isinstance(e, ast.Name) for e in t.elts))):
                raise StaticError("only plain names can be assigned")
    elif isinstance(node, ast.FunctionDef):
        for d in node.args.defaults:
            if not (isinstance(d, ast.Constant) and _num(d.value)) and not (
                    isinstance(d, ast.UnaryOp) and isinstance(d.operand, ast.Constant) and _num(d.operand.value)):
                raise StaticError(f"def {node.name}: a default value must be a number")


def _module_call(tree, entry, names):
    for s in tree.body:
        if isinstance(s, ast.FunctionDef):
            continue
        call = _binding_call(s, {entry}, names)
        if call is not None:
            return call
    return None


def _calls_user(node, funcs) -> bool:
    return any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in funcs
               for n in ast.walk(node))


class _Return(Exception):
    def __init__(self, value):
        self.value = value


class _Break(Exception):
    pass


class _Continue(Exception):
    pass


class _Found(Exception):
    def __init__(self, value):
        self.value = value


class _Interp:
    def __init__(self, funcs: dict, from_math: dict):
        self.funcs, self.from_math = funcs, from_math
        self.steps = self.depth = 0
        self.last_assigned = None
        self.stop_at = None
        self.globals: dict = {}

    def tick(self):
        self.steps += 1
        if self.steps > MAX_STEPS:
            raise RunError(f"the program took more than {MAX_STEPS} steps")

    def module(self, body, env, skip_user_calls=False, stop_at=None):
        self.stop_at, self.globals = stop_at, env
        try:
            for s in body:
                if isinstance(s, (ast.FunctionDef, ast.Import, ast.ImportFrom)):
                    continue
                if skip_user_calls and _calls_user(s, self.funcs):
                    continue
                self.stmt(s, env, top=True)
        except _Found as f:
            return f.value
        if stop_at is not None:
            raise RunError("the call of the solving function was never reached")
        return None

    def block(self, body, env, top=False):
        for s in body:
            self.stmt(s, env, top)

    def stmt(self, s, env, top=False):
        self.tick()
        if isinstance(s, ast.Assign):
            v = self.eval(s.value, env)
            for t in s.targets:
                self.assign(t, v, env, top)
        elif isinstance(s, ast.AugAssign):
            v = self.binop(type(s.op), self.lookup(s.target.id, env), self.eval(s.value, env))
            self.assign(s.target, v, env, top)
        elif isinstance(s, ast.Return):
            raise _Return(self.eval(s.value, env) if s.value is not None else None)
        elif isinstance(s, ast.If):
            self.block(s.body if self.truth(self.eval(s.test, env)) else s.orelse, env, top)
        elif isinstance(s, ast.For):
            for x in self.iterable(s.iter, env):
                self.tick()
                self.assign(s.target, x, env, top)
                try:
                    self.block(s.body, env, top)
                except _Break:
                    break
                except _Continue:
                    continue
        elif isinstance(s, ast.While):
            n = 0
            while self.truth(self.eval(s.test, env)):
                n += 1
                if n > MAX_LOOP:
                    raise RunError(f"a while loop ran more than {MAX_LOOP} times")
                try:
                    self.block(s.body, env, top)
                except _Break:
                    break
                except _Continue:
                    continue
        elif isinstance(s, ast.Break):
            raise _Break()
        elif isinstance(s, ast.Continue):
            raise _Continue()
        elif isinstance(s, ast.Expr):
            self.eval(s.value, env)
        elif isinstance(s, (ast.Pass, ast.Import, ast.ImportFrom, ast.FunctionDef)):
            return
        else:
            raise RunError(f"{type(s).__name__} cannot run")

    def assign(self, target, v, env, top):
        if isinstance(target, ast.Name):
            env[target.id] = v
            if top and target.id != "__binding__":
                self.last_assigned = target.id
            return
        if not isinstance(v, list) or len(v) != len(target.elts):
            raise RunError("cannot unpack")
        for t, x in zip(target.elts, v):
            self.assign(t, x, env, top)

    def iterable(self, node, env):
        if isinstance(node, (ast.List, ast.Tuple)):
            return [self.eval(e, env) for e in node.elts]
        r = range(*[self.whole(self.eval(a, env)) for a in node.args])
        if len(r) > MAX_LOOP:
            raise RunError(f"range() longer than {MAX_LOOP}")
        return r

    def whole(self, x):
        self.number(x)
        if abs(x - round(x)) > EPS:
            raise RunError("range() needs whole numbers")
        return int(round(x))

    def truth(self, x):
        if isinstance(x, NotANumber):
            raise RunError(f"{x.name} is not a number")
        return bool(x)

    def lookup(self, name, env):
        if name in env:
            return env[name]
        if name in self.from_math and self.from_math[name] in MATH_CONSTS:
            return MATH_CONSTS[self.from_math[name]]
        if name in MATH_CONSTS:
            return MATH_CONSTS[name]
        raise RunError(f"name {name} has no value here")

    def number(self, x):
        if isinstance(x, NotANumber):
            raise RunError(f"{x.name} is not a number")
        if not (_num(x) or isinstance(x, bool)):
            raise RunError("something that is not a number is used as one")
        return x

    def eval(self, node, env):
        self.tick()
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            return self.lookup(node.id, env)
        if isinstance(node, ast.BinOp):
            return self.binop(type(node.op), self.eval(node.left, env), self.eval(node.right, env))
        if isinstance(node, ast.UnaryOp):
            x = self.eval(node.operand, env)
            if isinstance(node.op, ast.Not):
                return not self.truth(x)
            self.number(x)
            return -x if isinstance(node.op, ast.USub) else x
        if isinstance(node, ast.BoolOp):
            want = isinstance(node.op, ast.Or)
            v = None
            for e in node.values:
                v = self.eval(e, env)
                if self.truth(v) == want:
                    return v
            return v
        if isinstance(node, ast.Compare):
            left = self.number(self.eval(node.left, env))
            for op, comp in zip(node.ops, node.comparators):
                right = self.number(self.eval(comp, env))
                if not _CMPS[type(op)](left, right):
                    return False
                left = right
            return True
        if isinstance(node, ast.IfExp):
            return self.eval(node.body if self.truth(self.eval(node.test, env)) else node.orelse, env)
        if isinstance(node, (ast.List, ast.Tuple)):
            return [self.eval(e, env) for e in node.elts]
        if isinstance(node, ast.Attribute):
            if node.attr in MATH_CONSTS:
                return MATH_CONSTS[node.attr]
            raise RunError(f"math.{node.attr} is not a value")
        if isinstance(node, ast.Call):
            if node is self.stop_at:
                raise _Found(self.call(node, env))
            return self.call(node, env)
        raise RunError(f"{type(node).__name__} cannot run")

    def binop(self, op, a, b):
        self.number(a), self.number(b)
        if op is ast.Pow:
            if abs(b) > MAX_POW:
                raise RunError(f"an exponent larger than {MAX_POW}")
            v = a ** b
        else:
            v = _BINOPS[op](a, b)
        if isinstance(v, complex):
            raise RunError("a complex number")
        if abs(v) > MAX_ABS or (isinstance(v, float) and not math.isfinite(v)):
            raise RunError("a value grew too large")
        return v

    def call(self, node, env):
        f = node.func
        args = [self.eval(a, env) for a in node.args]
        kw = {k.arg: self.eval(k.value, env) for k in node.keywords}
        if isinstance(f, ast.Name) and f.id in self.funcs:
            return self.call_user(f.id, args, kw, env)
        name = f.attr if isinstance(f, ast.Attribute) else self.from_math.get(f.id, f.id)
        if name == "range":
            raise RunError("range() only in a for loop")
        for a in [*args, *kw.values()]:
            if isinstance(a, NotANumber):
                raise RunError(f"{a.name} is not a number")
            if not isinstance(a, list):
                self.number(a)
        if any(isinstance(a, list) for a in args) and name not in ("min", "max", "sum", "len"):
            raise RunError(f"{name}() of a list")
        return FUNCS[name](*args, **kw)

    def call_user(self, name, args, kw, env, by_name=None):
        fn = self.funcs[name]
        self.depth += 1
        try:
            if self.depth > MAX_DEPTH:
                raise RunError("recursion too deep")
            params = [a.arg for a in fn.args.args]
            defaults = dict(zip(params[len(params) - len(fn.args.defaults):],
                                [self.eval(d, {}) for d in fn.args.defaults]))
            local = dict(self.globals)
            if by_name is not None:
                for p in params:
                    local[p] = self.globals[p] if p in by_name else defaults.get(p)
            else:
                if len(args) > len(params):
                    raise RunError(f"{name}() got too many arguments")
                bound = {**dict(zip(params, args)), **kw}
                for p in params:
                    if p not in bound:
                        if p not in defaults:
                            raise RunError(f"{name}() is missing the argument {p}")
                        bound[p] = defaults[p]
                local.update(bound)
            try:
                self.block(fn.body, local)
            except _Return as r:
                return r.value
            return None
        finally:
            self.depth -= 1
