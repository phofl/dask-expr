from __future__ import annotations

import functools
import numbers
import operator
from collections import defaultdict
from collections.abc import Mapping

import dask
import pandas as pd
from dask.base import normalize_token, tokenize
from dask.dataframe import methods
from dask.dataframe.core import (
    _get_divisions_map_partitions,
    _get_meta_map_partitions,
    apply_and_enforce,
    is_dataframe_like,
    is_index_like,
    is_series_like,
    make_meta,
)
from dask.dataframe.dispatch import meta_nonempty
from dask.utils import M, apply, funcname

from dask_expr.expr import Expr

replacement_rules = []

no_default = "__no_default__"


class FrameExpr(Expr):
    """Primary class for all Expressions

    This mostly includes pandas-like method
    definitions to make us look more like a DataFrame.
    """

    def __hash__(self):
        return hash(self._name)

    @functools.cached_property
    def ndim(self):
        meta = self._meta
        try:
            return meta.ndim
        except AttributeError:
            return 0

    def __getattr__(self, key):
        try:
            return object.__getattribute__(self, key)
        except AttributeError as err:
            # Allow operands to be accessed as attributes
            # as long as the keys are not already reserved
            # by existing methods/properties
            _parameters = type(self)._parameters
            if key in _parameters:
                idx = _parameters.index(key)
                return self.operands[idx]
            if is_dataframe_like(self._meta) and key in self._meta.columns:
                return self[key]
            raise err

    def _layer(self) -> dict:
        """The graph layer added by this expression

        Examples
        --------
        >>> class Add(FrameExpr):
        ...     def _layer(self):
        ...         return {
        ...             (self._name, i): (operator.add, (self.left._name, i), (self.right._name, i))
        ...             for i in range(self.npartitions)
        ...         }

        Returns
        -------
        layer: dict
            The Dask task graph added by this expression

        See Also
        --------
        Expr._task
        Expr.__dask_graph__
        """

        return {(self._name, i): self._task(i) for i in range(self.npartitions)}

    def _simplify_down(self):
        return

    def _simplify_up(self, parent):
        return

    def optimize(self, **kwargs):
        return optimize(self, **kwargs)

    @property
    def index(self):
        return Index(self)

    @property
    def size(self):
        return Size(self)

    @property
    def nbytes(self):
        return NBytes(self)

    def __getitem__(self, other):
        if isinstance(other, FrameExpr):
            return Filter(self, other)  # df[df.x > 1]
        else:
            return Projection(self, other)  # df[["a", "b", "c"]]

    def __add__(self, other):
        return Add(self, other)

    def __radd__(self, other):
        return Add(other, self)

    def __sub__(self, other):
        return Sub(self, other)

    def __rsub__(self, other):
        return Sub(other, self)

    def __mul__(self, other):
        return Mul(self, other)

    def __rmul__(self, other):
        return Mul(other, self)

    def __truediv__(self, other):
        return Div(self, other)

    def __rtruediv__(self, other):
        return Div(other, self)

    def __lt__(self, other):
        return LT(self, other)

    def __rlt__(self, other):
        return LT(other, self)

    def __gt__(self, other):
        return GT(self, other)

    def __rgt__(self, other):
        return GT(other, self)

    def __le__(self, other):
        return LE(self, other)

    def __rle__(self, other):
        return LE(other, self)

    def __ge__(self, other):
        return GE(self, other)

    def __rge__(self, other):
        return GE(other, self)

    def __eq__(self, other):
        return EQ(self, other)

    def __ne__(self, other):
        return NE(self, other)

    def __and__(self, other):
        return And(self, other)

    def __rand__(self, other):
        return And(other, self)

    def __or__(self, other):
        return Or(self, other)

    def __ror__(self, other):
        return Or(other, self)

    def __xor__(self, other):
        return XOr(self, other)

    def __rxor__(self, other):
        return XOr(other, self)

    def __invert__(self):
        return Invert(self)

    def __neg__(self):
        return Neg(self)

    def __pos__(self):
        return Pos(self)

    def sum(self, skipna=True, numeric_only=False, min_count=0):
        return Sum(self, skipna, numeric_only, min_count)

    def prod(self, skipna=True, numeric_only=False, min_count=0):
        return Prod(self, skipna, numeric_only, min_count)

    def mean(self, skipna=True, numeric_only=False, min_count=0):
        return Mean(self, skipna=skipna, numeric_only=numeric_only)

    def max(self, skipna=True, numeric_only=False, min_count=0):
        return Max(self, skipna, numeric_only, min_count)

    def any(self, skipna=True):
        return Any(self, skipna=skipna)

    def all(self, skipna=True):
        return All(self, skipna=skipna)

    def idxmin(self, skipna=True, numeric_only=False):
        return IdxMin(self, skipna=skipna, numeric_only=numeric_only)

    def idxmax(self, skipna=True, numeric_only=False):
        return IdxMax(self, skipna=skipna, numeric_only=numeric_only)

    def mode(self, dropna=True):
        return Mode(self, dropna=dropna)

    def min(self, skipna=True, numeric_only=False, min_count=0):
        return Min(self, skipna, numeric_only, min_count)

    def count(self, numeric_only=False):
        return Count(self, numeric_only)

    def abs(self):
        return Abs(self)

    def astype(self, dtypes):
        return AsType(self, dtypes)

    def clip(self, lower=None, upper=None):
        return Clip(self, lower=lower, upper=upper)

    def combine_first(self, other):
        return CombineFirst(self, other=other)

    def to_timestamp(self, freq=None, how="start"):
        return ToTimestamp(self, freq=freq, how=how)

    def isna(self):
        return IsNa(self)

    def round(self, decimals=0):
        return Round(self, decimals=decimals)

    def apply(self, function, *args, **kwargs):
        return Apply(self, function, args, kwargs)

    def replace(self, to_replace=None, value=no_default, regex=False):
        return Replace(self, to_replace=to_replace, value=value, regex=regex)

    @functools.cached_property
    def divisions(self):
        return tuple(self._divisions())

    def _divisions(self):
        raise NotImplementedError()

    @property
    def known_divisions(self):
        """Whether divisions are already known"""
        return len(self.divisions) > 0 and self.divisions[0] is not None

    @property
    def npartitions(self):
        if "npartitions" in self._parameters:
            idx = self._parameters.index("npartitions")
            return self.operands[idx]
        else:
            return len(self.divisions) - 1

    @property
    def columns(self):
        try:
            return self._meta.columns
        except AttributeError:
            return pd.Index([])

    @property
    def dtypes(self):
        return self._meta.dtypes

    @property
    def _meta(self):
        raise NotImplementedError()


class Literal(FrameExpr):
    """Represent a literal (known) value as an `Expr`"""

    _parameters = ["value"]

    def _divisions(self):
        return (None, None)

    @property
    def _meta(self):
        return make_meta(self.value)

    def _task(self, index: int):
        assert index == 0
        return self.value


class Blockwise(FrameExpr):
    """Super-class for block-wise operations

    This is fairly generic, and includes definitions for `_meta`, `divisions`,
    `_layer` that are often (but not always) correct.  Mostly this helps us
    avoid duplication in the future.

    Note that `Fused` expressions rely on every `Blockwise`
    expression defining a proper `_task` method.
    """

    operation = None
    _keyword_only = []

    @functools.cached_property
    def _meta(self):
        args = [op._meta if isinstance(op, FrameExpr) else op for op in self._args]
        return self.operation(*args, **self._kwargs)

    @functools.cached_property
    def _kwargs(self) -> dict:
        if self._keyword_only:
            return {
                p: self.operand(p)
                for p in self._parameters
                if p in self._keyword_only and self.operand(p) is not no_default
            }
        return {}

    @functools.cached_property
    def _args(self) -> list:
        if self._keyword_only:
            args = [
                self.operand(p) for p in self._parameters if p not in self._keyword_only
            ] + self.operands[len(self._parameters) :]
            return args
        return self.operands

    def _broadcast_dep(self, dep: FrameExpr):
        # Checks if a dependency should be broadcasted to
        # all partitions of this `Blockwise` operation
        return dep.npartitions == 1 and dep.ndim < self.ndim

    def _divisions(self):
        # This is an issue.  In normal Dask we re-divide everything in a step
        # which combines divisions and graph.
        # We either have to create a new Align layer (ok) or combine divisions
        # and graph into a single operation.
        dependencies = self.dependencies()
        for arg in dependencies:
            if not self._broadcast_dep(arg):
                assert arg.divisions == dependencies[0].divisions
        return dependencies[0].divisions

    @functools.cached_property
    def _name(self):
        if self.operation:
            head = funcname(self.operation)
        else:
            head = funcname(type(self)).lower()
        return head + "-" + tokenize(*self.operands)

    def _blockwise_arg(self, arg, i):
        """Return a Blockwise-task argument"""
        if isinstance(arg, FrameExpr):
            # Make key for Expr-based argument
            if self._broadcast_dep(arg):
                return (arg._name, 0)
            else:
                return (arg._name, i)

        else:
            return arg

    def _task(self, index: int):
        """Produce the task for a specific partition

        Parameters
        ----------
        index:
            Partition index for this task.

        Returns
        -------
        task: tuple
        """
        args = [self._blockwise_arg(op, index) for op in self._args]
        if self._kwargs:
            return apply, self.operation, args, self._kwargs
        else:
            return (self.operation,) + tuple(args)


class MapPartitions(Blockwise):
    _parameters = [
        "frame",
        "func",
        "meta",
        "enforce_metadata",
        "transform_divisions",
        "clear_divisions",
        "kwargs",
    ]
    _defaults = {"kwargs": None}

    def __str__(self):
        return f"MapPartitions({funcname(self.func)})"

    def _broadcast_dep(self, dep: FrameExpr):
        # Always broadcast single-partition dependencies in MapPartitions
        return dep.npartitions == 1

    @property
    def args(self):
        return [self.frame] + self.operands[len(self._parameters) :]

    @functools.cached_property
    def _meta(self):
        meta = self.operand("meta")
        args = [arg._meta if isinstance(arg, FrameExpr) else arg for arg in self.args]
        return _get_meta_map_partitions(args, [], self.func, self.kwargs, meta, None)

    def _divisions(self):
        # Unknown divisions
        if self.clear_divisions:
            return (None,) * (self.frame.npartitions + 1)

        # (Possibly) known divisions
        dfs = [arg for arg in self.args if isinstance(arg, FrameExpr)]
        return _get_divisions_map_partitions(
            True,  # Partitions must already be "aligned"
            self.transform_divisions,
            dfs,
            self.func,
            self.args,
            self.kwargs,
        )

    def _task(self, index: int):
        args = [self._blockwise_arg(op, index) for op in self.args]
        kwargs = self.kwargs if self.kwargs is not None else {}
        if self.enforce_metadata:
            kwargs = kwargs.copy()
            kwargs.update(
                {
                    "_func": self.func,
                    "_meta": self._meta,
                }
            )
            return (apply, apply_and_enforce, args, kwargs)
        else:
            return (
                apply,
                self.func,
                args,
                kwargs,
            )


class DropnaSeries(Blockwise):
    _parameters = ["frame"]
    operation = M.dropna


class DropnaFrame(Blockwise):
    _parameters = ["frame", "how", "subset", "thresh"]
    _defaults = {"how": no_default, "subset": None, "thresh": no_default}
    _keyword_only = ["how", "subset", "thresh"]
    operation = M.dropna


class Replace(Blockwise):
    _parameters = ["frame", "to_replace", "value", "regex"]
    _defaults = {"to_replace": None, "value": no_default, "regex": False}
    _keyword_only = ["value", "regex"]
    operation = M.replace


class CombineFirst(Blockwise):
    _parameters = ["frame", "other"]
    operation = M.combine_first


class RenameFrame(Blockwise):
    _parameters = ["frame", "columns"]
    _keyword_only = ["columns"]
    operation = M.rename

    def _simplify_up(self, parent):
        if isinstance(parent, Projection) and isinstance(
            self.operand("columns"), Mapping
        ):
            reverse_mapping = {val: key for key, val in self.operand("columns").items()}
            if not isinstance(parent.columns, pd.Index):
                # Fill this out when Series.rename is implemented
                return
            else:
                columns = [
                    reverse_mapping[col] if col in reverse_mapping else col
                    for col in parent.columns
                ]
            return type(self)(self.frame[columns], *self.operands[1:])


class Sample(Blockwise):
    _parameters = ["frame", "state_data", "frac", "replace"]
    operation = staticmethod(methods.sample)

    @functools.cached_property
    def _meta(self):
        args = [self.operands[0]._meta] + [self.operands[1][0]] + self.operands[2:]
        return self.operation(*args)

    def _task(self, index: int):
        args = [self._blockwise_arg(self.frame, index)] + [
            self.state_data[index],
            self.frac,
            self.replace,
        ]
        return (self.operation,) + tuple(args)


class ToFrame(Blockwise):
    _parameters = ["frame", "name"]
    _defaults = {"name": no_default}
    _keyword_only = ["name"]
    operation = M.to_frame


class ToFrameIndex(Blockwise):
    _parameters = ["frame", "index", "name"]
    _defaults = {"name": no_default, "index": True}
    _keyword_only = ["name", "index"]
    operation = M.to_frame


class Elemwise(Blockwise):
    """
    This doesn't really do anything, but we anticipate that future
    optimizations, like `len` will care about which operations preserve length
    """

    pass


class Clip(Elemwise):
    _parameters = ["frame", "lower", "upper"]
    _defaults = {"lower": None, "upper": None}
    operation = M.clip

    def _simplify_up(self, parent):
        if isinstance(parent, Projection):
            if self.frame.columns.equals(parent.columns):
                # Don't introduce unnecessary projections
                return
            return type(self)(self.frame[parent.operand("columns")], *self.operands[1:])


class Between(Elemwise):
    _parameters = ["frame", "left", "right", "inclusive"]
    _defaults = {"inclusive": "both"}
    operation = M.between


class ToTimestamp(Elemwise):
    _parameters = ["frame", "freq", "how"]
    _defaults = {"freq": None, "how": "start"}
    operation = M.to_timestamp

    def _divisions(self):
        return tuple(
            pd.Index(self.frame.divisions).to_timestamp(freq=self.freq, how=self.how)
        )


class AsType(Elemwise):
    """A good example of writing a trivial blockwise operation"""

    _parameters = ["frame", "dtypes"]
    operation = M.astype


class IsNa(Elemwise):
    _parameters = ["frame"]
    operation = M.isna


class Round(Elemwise):
    _parameters = ["frame", "decimals"]
    operation = M.round


class Abs(Elemwise):
    _parameters = ["frame"]
    operation = M.abs


class Apply(Elemwise):
    """A good example of writing a less-trivial blockwise operation"""

    _parameters = ["frame", "function", "args", "kwargs"]
    _defaults = {"args": (), "kwargs": {}}
    operation = M.apply

    @property
    def _meta(self):
        return self.frame._meta.apply(self.function, *self.args, **self.kwargs)

    def _task(self, index: int):
        return (
            apply,
            M.apply,
            [
                (self.frame._name, index),
                self.function,
            ]
            + list(self.args),
            self.kwargs,
        )


class Map(Elemwise):
    _parameters = ["frame", "arg", "na_action"]
    _defaults = {"na_action": None}
    operation = M.map

    @property
    def _meta(self):
        return self.frame._meta

    def _divisions(self):
        if is_index_like(self.frame._meta):
            # Implement this consistently with dask.dataframe, e.g. add option to
            # control monotonic map func
            return (None,) * len(self.frame.divisions)
        return super()._divisions()


class ExplodeSeries(Blockwise):
    _parameters = ["frame"]
    operation = M.explode


class ExplodeFrame(ExplodeSeries):
    _parameters = ["frame", "column"]


class Assign(Elemwise):
    """Column Assignment"""

    _parameters = ["frame", "key", "value"]
    operation = staticmethod(methods.assign)

    def _node_label_args(self):
        return [self.frame, self.key, self.value]


class Filter(Blockwise):
    _parameters = ["frame", "predicate"]
    operation = operator.getitem

    def _simplify_up(self, parent):
        if isinstance(parent, Projection):
            return self.frame[parent.operand("columns")][self.predicate]
        if isinstance(parent, Index):
            return self.frame.index[self.predicate]


class Projection(Elemwise):
    """Column Selection"""

    _parameters = ["frame", "columns"]
    operation = operator.getitem

    @property
    def columns(self):
        if isinstance(self.operand("columns"), list):
            return pd.Index(self.operand("columns"))
        else:
            return self.operand("columns")

    @property
    def _meta(self):
        if is_dataframe_like(self.frame._meta):
            return super()._meta
        # if we are not a DataFrame and have a scalar, we reduce to a scalar
        if not isinstance(self.operand("columns"), list) and not hasattr(
            self.operand("columns"), "dtype"
        ):
            return meta_nonempty(self.frame._meta).iloc[0]
        # Avoid column selection for Series/Index
        return self.frame._meta

    def _node_label_args(self):
        return [self.frame, self.operand("columns")]

    def __str__(self):
        base = str(self.frame)
        if " " in base:
            base = "(" + base + ")"
        return f"{base}[{repr(self.columns)}]"

    def _simplify_down(self):
        if str(self.frame.columns) == str(self.columns):
            # TODO: we should get more precise around Expr.columns types
            return self.frame
        if isinstance(self.frame, Projection):
            # df[a][b]
            a = self.frame.operand("columns")
            b = self.operand("columns")

            if not isinstance(a, list):
                # df[scalar][b] -> First selection coerces to Series
                return
            elif isinstance(b, list):
                assert all(bb in a for bb in b)
            else:
                assert b in a

            return self.frame.frame[b]


class Index(Elemwise):
    """Column Selection"""

    _parameters = ["frame"]
    operation = getattr

    @property
    def _meta(self):
        meta = self.frame._meta
        # Handle scalar results
        if is_series_like(meta) or is_dataframe_like(meta):
            return self.frame._meta.index
        return meta

    def _task(self, index: int):
        return (
            getattr,
            (self.frame._name, index),
            "index",
        )


class Lengths(FrameExpr):
    """Returns a tuple of partition lengths"""

    _parameters = ["frame"]

    @property
    def _meta(self):
        return tuple()

    def _divisions(self):
        return (None, None)

    def _simplify_down(self):
        if isinstance(self.frame, Elemwise):
            child = max(self.frame.dependencies(), key=lambda expr: expr.npartitions)
            return Lengths(child)

    def _layer(self):
        name = "part-" + self._name
        dsk = {
            (name, i): (len, (self.frame._name, i))
            for i in range(self.frame.npartitions)
        }
        dsk[(self._name, 0)] = (tuple, list(dsk.keys()))
        return dsk


class ResetIndex(Elemwise):
    """Reset the index of a Series or DataFrame"""

    _parameters = ["frame", "drop"]
    _defaults = {"drop": False}
    _keyword_only = ["drop"]
    operation = M.reset_index

    def _divisions(self):
        return (None,) * (self.frame.npartitions + 1)


class Head(FrameExpr):
    """Take the first `n` rows of the first partition"""

    _parameters = ["frame", "n"]
    _defaults = {"n": 5}

    @property
    def _meta(self):
        return self.frame._meta

    def _divisions(self):
        return self.frame.divisions[:2]

    def _task(self, index: int):
        raise NotImplementedError()

    def _simplify_down(self):
        if isinstance(self.frame, Elemwise):
            operands = [
                Head(op, self.n) if isinstance(op, FrameExpr) else op
                for op in self.frame.operands
            ]
            return type(self.frame)(*operands)
        if not isinstance(self, BlockwiseHead):
            # Lower to Blockwise
            return BlockwiseHead(Partitions(self.frame, [0]), self.n)
        if isinstance(self.frame, Head):
            return Head(self.frame.frame, min(self.n, self.frame.n))


class BlockwiseHead(Head, Blockwise):
    """Take the first `n` rows of every partition

    Typically used after `Partitions(..., [0])` to take
    the first `n` rows of an entire collection.
    """

    def _divisions(self):
        return self.frame.divisions

    def _task(self, index: int):
        return (M.head, (self.frame._name, index), self.n)


class Tail(FrameExpr):
    """Take the last `n` rows of the last partition"""

    _parameters = ["frame", "n"]
    _defaults = {"n": 5}

    @property
    def _meta(self):
        return self.frame._meta

    def _divisions(self):
        return self.frame.divisions[-2:]

    def _task(self, index: int):
        raise NotImplementedError()

    def _simplify_down(self):
        if isinstance(self.frame, Elemwise):
            operands = [
                Tail(op, self.n) if isinstance(op, FrameExpr) else op
                for op in self.frame.operands
            ]
            return type(self.frame)(*operands)
        if not isinstance(self, BlockwiseTail):
            # Lower to Blockwise
            return BlockwiseTail(Partitions(self.frame, [-1]), self.n)
        if isinstance(self.frame, Tail):
            return Tail(self.frame.frame, min(self.n, self.frame.n))


class BlockwiseTail(Tail, Blockwise):
    """Take the last `n` rows of every partition

    Typically used after `Partitions(..., [-1])` to take
    the last `n` rows of an entire collection.
    """

    def _divisions(self):
        return self.frame.divisions

    def _task(self, index: int):
        return (M.tail, (self.frame._name, index), self.n)


class Binop(Elemwise):
    _parameters = ["left", "right"]

    def __str__(self):
        return f"{self.left} {self._operator_repr} {self.right}"

    def _simplify_up(self, parent):
        if isinstance(parent, Projection):
            if isinstance(self.left, FrameExpr):
                left = self.left[
                    parent.operand("columns")
                ]  # TODO: filter just the correct columns
            else:
                left = self.left
            if isinstance(self.right, FrameExpr):
                right = self.right[parent.operand("columns")]
            else:
                right = self.right
            return type(self)(left, right)

    def _node_label_args(self):
        return [self.left, self.right]


class Add(Binop):
    operation = operator.add
    _operator_repr = "+"

    def _simplify_down(self):
        if (
            isinstance(self.left, FrameExpr)
            and isinstance(self.right, FrameExpr)
            and self.left._name == self.right._name
        ):
            return 2 * self.left


class Sub(Binop):
    operation = operator.sub
    _operator_repr = "-"


class Mul(Binop):
    operation = operator.mul
    _operator_repr = "*"

    def _simplify_down(self):
        if (
            isinstance(self.right, Mul)
            and isinstance(self.left, numbers.Number)
            and isinstance(self.right.left, numbers.Number)
        ):
            return (self.left * self.right.left) * self.right.right


class Div(Binop):
    operation = operator.truediv
    _operator_repr = "/"


class LT(Binop):
    operation = operator.lt
    _operator_repr = "<"


class LE(Binop):
    operation = operator.le
    _operator_repr = "<="


class GT(Binop):
    operation = operator.gt
    _operator_repr = ">"


class GE(Binop):
    operation = operator.ge
    _operator_repr = ">="


class EQ(Binop):
    operation = operator.eq
    _operator_repr = "=="


class NE(Binop):
    operation = operator.ne
    _operator_repr = "!="


class And(Binop):
    operation = operator.and_
    _operator_repr = "&"


class Or(Binop):
    operation = operator.or_
    _operator_repr = "|"


class XOr(Binop):
    operation = operator.xor
    _operator_repr = "^"


class Unaryop(Elemwise):
    _parameters = ["frame"]

    def __str__(self):
        return f"{self._operator_repr} {self.frame}"

    def _simplify_up(self, parent):
        if isinstance(parent, Projection):
            if isinstance(self.frame, FrameExpr):
                frame = self.frame[
                    parent.operand("columns")
                ]  # TODO: filter just the correct columns
            else:
                frame = self.frame
            return type(self)(frame)

    def _node_label_args(self):
        return [self.frame]


class Invert(Unaryop):
    operation = operator.inv
    _operator_repr = "~"


class Neg(Unaryop):
    operation = operator.neg
    _operator_repr = "-"


class Pos(Unaryop):
    operation = operator.pos
    _operator_repr = "+"


class Partitions(FrameExpr):
    """Select one or more partitions"""

    _parameters = ["frame", "partitions"]

    @property
    def _meta(self):
        return self.frame._meta

    def _divisions(self):
        divisions = []
        for part in self.partitions:
            divisions.append(self.frame.divisions[part])
        divisions.append(self.frame.divisions[part + 1])
        return tuple(divisions)

    def _task(self, index: int):
        return (self.frame._name, self.partitions[index])

    def _simplify_down(self):
        if isinstance(self.frame, Blockwise) and not isinstance(
            self.frame, (BlockwiseIO, Fused)
        ):
            operands = [
                Partitions(op, self.partitions)
                if (isinstance(op, FrameExpr) and not self.frame._broadcast_dep(op))
                else op
                for op in self.frame.operands
            ]
            return type(self.frame)(*operands)
        elif isinstance(self.frame, PartitionsFiltered):
            if self.frame._partitions:
                partitions = [self.frame._partitions[p] for p in self.partitions]
            else:
                partitions = self.partitions
            # We assume that expressions defining a special "_partitions"
            # parameter can internally capture the same logic as `Partitions`
            operands = [
                partitions if self.frame._parameters[i] == "_partitions" else op
                for i, op in enumerate(self.frame.operands)
            ]
            return type(self.frame)(*operands)

    def _node_label_args(self):
        return [self.frame, self.partitions]


class PartitionsFiltered(FrameExpr):
    """Mixin class for partition filtering

    A `PartitionsFiltered` subclass must define a
    `_partitions` parameter. When `_partitions` is
    defined, the following expresssions must produce
    the same output for `cls: PartitionsFiltered`:
      - `cls(expr: Expr, ..., _partitions)`
      - `Partitions(cls(expr: Expr, ...), _partitions)`

    In order to leverage the default `Expr._layer`
    method, subclasses should define `_filtered_task`
    instead of `_task`.
    """

    @property
    def _filtered(self) -> bool:
        """Whether or not output partitions have been filtered"""
        return self.operand("_partitions") is not None

    @property
    def _partitions(self) -> list | tuple | range:
        """Selected partition indices"""
        if self._filtered:
            return self.operand("_partitions")
        else:
            return range(self.npartitions)

    @functools.cached_property
    def divisions(self):
        # Common case: Use self._divisions()
        full_divisions = super().divisions
        if not self._filtered:
            return full_divisions

        # Specific case: Specific partitions were selected
        new_divisions = []
        for part in self._partitions:
            new_divisions.append(full_divisions[part])
        new_divisions.append(full_divisions[part + 1])
        return tuple(new_divisions)

    @property
    def npartitions(self):
        if self._filtered:
            return len(self._partitions)
        return super().npartitions

    def _task(self, index: int):
        return self._filtered_task(self._partitions[index])

    def _filtered_task(self, index: int):
        raise NotImplementedError()


@normalize_token.register(FrameExpr)
def normalize_expression(expr):
    return expr._name


## Utilites for Expr fusion


def optimize(expr: Expr, fuse: bool = True) -> Expr:
    """High level query optimization

    This leverages three optimization passes:

    1.  Class based simplification using the ``_simplify`` function and methods
    2.  Blockwise fusion

    Parameters
    ----------
    expr:
        Input expression to optimize
    fuse:
        whether or not to turn on blockwise fusion

    See Also
    --------
    simplify
    optimize_blockwise_fusion
    """
    expr = expr.simplify()

    if fuse:
        expr = optimize_blockwise_fusion(expr)

    return expr


def optimize_blockwise_fusion(expr):
    """Traverse the expression graph and apply fusion"""

    def _fusion_pass(expr):
        # Full pass to find global dependencies
        seen = set()
        stack = [expr]
        dependents = defaultdict(set)
        dependencies = {}
        while stack:
            next = stack.pop()

            if next._name in seen:
                continue
            seen.add(next._name)

            if isinstance(next, Blockwise):
                dependencies[next] = set()
                if next not in dependents:
                    dependents[next] = set()

            for operand in next.operands:
                if isinstance(operand, FrameExpr):
                    stack.append(operand)
                    if isinstance(operand, Blockwise):
                        if next in dependencies:
                            dependencies[next].add(operand)
                        dependents[operand].add(next)

        # Traverse each "root" until we find a fusable sub-group.
        # Here we use root to refer to a Blockwise Expr node that
        # has no Blockwise dependents
        roots = [
            k
            for k, v in dependents.items()
            if v == set() or all(not isinstance(_expr, Blockwise) for _expr in v)
        ]
        while roots:
            root = roots.pop()
            seen = set()
            stack = [root]
            group = []
            while stack:
                next = stack.pop()

                if next._name in seen:
                    continue
                seen.add(next._name)

                group.append(next)
                for dep in dependencies[next]:
                    if (dep.npartitions == root.npartitions) and not (
                        dependents[dep] - set(stack) - set(group)
                    ):
                        # All of deps dependents are contained
                        # in the local group (or the local stack
                        # of expr nodes that we know we will be
                        # adding to the local group).
                        # All nodes must also have the same number
                        # of partitions, since broadcasting within
                        # a group is not allowed.
                        stack.append(dep)
                    elif dep not in roots and dependencies[dep]:
                        # Couldn't fuse dep, but we may be able to
                        # use it as a new root on the next pass
                        roots.append(dep)

            # Replace fusable sub-group
            if len(group) > 1:
                group_deps = []
                local_names = [_expr._name for _expr in group]
                for _expr in group:
                    group_deps += [
                        operand
                        for operand in _expr.dependencies()
                        if operand._name not in local_names
                    ]
                to_replace = {group[0]: Fused(group, *group_deps)}
                return expr.substitute(to_replace), not roots

        # Return original expr if no fusable sub-groups were found
        return expr, True

    while True:
        original_name = expr._name
        expr, done = _fusion_pass(expr)
        if done or expr._name == original_name:
            break

    return expr


class Fused(Blockwise):
    """Fused ``Blockwise`` expression

    A ``Fused`` corresponds to the fusion of multiple
    ``Blockwise`` expressions into a single ``Expr`` object.
    Before graph-materialization time, the behavior of this
    object should be identical to that of the first element
    of ``Fused.exprs`` (i.e. the top-most expression in
    the fused group).

    Parameters
    ----------
    exprs : List[Expr]
        Group of original ``Expr`` objects being fused together.
    *dependencies:
        List of external ``Expr`` dependencies. External-``Expr``
        dependencies correspond to any ``Expr`` operand that is
        not already included in ``exprs``. Note that these
        dependencies should be defined in the order of the ``Expr``
        objects that require them (in ``exprs``). These
        dependencies do not include literal operands, because those
        arguments should already be captured in the fused subgraphs.
    """

    _parameters = ["exprs"]

    @functools.cached_property
    def _meta(self):
        return self.exprs[0]._meta

    def _tree_repr_lines(self, indent=0, recursive=True):
        header = f"Fused({self._name[-5:]}):"
        if not recursive:
            return [header]

        seen = set()
        lines = []
        stack = [(self.exprs[0], 2)]
        fused_group = [_expr._name for _expr in self.exprs]
        dependencies = {dep._name: dep for dep in self.dependencies()}
        while stack:
            expr, _indent = stack.pop()

            if expr._name in seen:
                continue
            seen.add(expr._name)

            line = expr._tree_repr_lines(_indent, recursive=False)[0]
            lines.append(line.replace(" ", "|", 1))
            for dep in expr.dependencies():
                if dep._name in fused_group:
                    stack.append((dep, _indent + 2))
                elif dep._name in dependencies:
                    dependencies.pop(dep._name)
                    lines.extend(dep._tree_repr_lines(_indent + 2))

        for dep in dependencies.values():
            lines.extend(dep._tree_repr_lines(2))

        lines = [header] + lines
        lines = [" " * indent + line for line in lines]

        return lines

    def __str__(self):
        names = [expr._name.split("-")[0] for expr in self.exprs]
        if len(names) > 3:
            names = [names[0], f"{len(names) - 2}", names[-1]]
        descr = "-".join(names)
        return f"Fused-{descr}"

    @functools.cached_property
    def _name(self):
        return f"{str(self)}-{tokenize(self.exprs)}"

    def _divisions(self):
        return self.exprs[0]._divisions()

    def _broadcast_dep(self, dep: FrameExpr):
        # Always broadcast single-partition dependencies in Fused
        return dep.npartitions == 1

    def _task(self, index):
        graph = {self._name: (self.exprs[0]._name, index)}
        for _expr in self.exprs:
            if isinstance(_expr, Fused):
                (_, subgraph, name) = _expr._task(index)
                graph.update(subgraph)
                graph[(name, index)] = name
            else:
                graph[(_expr._name, index)] = _expr._task(index)

        for i, dep in enumerate(self.dependencies()):
            graph[self._blockwise_arg(dep, index)] = "_" + str(i)

        return (
            Fused._execute_task,
            graph,
            self._name,
        ) + tuple(self._blockwise_arg(dep, index) for dep in self.dependencies())

    @staticmethod
    def _execute_task(graph, name, *deps):
        for i, dep in enumerate(deps):
            graph["_" + str(i)] = dep
        return dask.core.get(graph, name)


from dask_expr.io import BlockwiseIO
from dask_expr.reductions import (
    All,
    Any,
    Count,
    IdxMax,
    IdxMin,
    Max,
    Mean,
    Min,
    Mode,
    NBytes,
    Prod,
    Size,
    Sum,
)