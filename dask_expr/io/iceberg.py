import functools
from typing import Callable

from dask.utils import apply
from pyarrow._fs import PyFileSystem
from pyarrow.fs import FSSpecHandler
from pyiceberg.expressions.visitors import bind, extract_field_ids
from pyiceberg.io.pyarrow import PyArrowFileIO, _task_to_table, schema_to_pyarrow
from pyiceberg.types import ListType, MapType

from dask_expr._expr import PartitionsFiltered
from dask_expr.io import BlockwiseIO


class FromIceberg(PartitionsFiltered, BlockwiseIO):
    _parameters = ["table_scan", "_partitions"]
    _defaults = {"_partitions": None}

    @property
    def _name(self):
        return id(self)

    @functools.cached_property
    def _meta(self):
        return schema_to_pyarrow(self.table_scan.projection()).empty_table().to_pandas()

    @functools.cached_property
    def columns(self) -> list:
        return list(self.table_scan.selected_fields)

    @functools.cached_property
    def iceberg_tasks(self):
        return self.table_scan.plan_files()

    @functools.cached_property
    def projected_schema(self):
        return self.table_scan.projection()

    @functools.cached_property
    def fs(self):
        scheme, _ = PyArrowFileIO.parse_location(self.table_scan.table.location())
        if isinstance(self.table_scan.table.io, PyArrowFileIO):
            return self.table_scan.table.io.get_fs(scheme)
        else:
            try:
                from pyiceberg.io.fsspec import FsspecFileIO

                if isinstance(self.table_scan.table.io, FsspecFileIO):
                    return PyFileSystem(
                        FSSpecHandler(self.table_scan.table.io.get_fs(scheme))
                    )
                else:
                    raise ValueError(
                        f"Expected PyArrowFileIO or FsspecFileIO, got: {self.table_scan.table.io}"
                    )
            except ModuleNotFoundError as e:
                # When FsSpec is not installed
                raise ValueError(
                    f"Expected PyArrowFileIO or FsspecFileIO, got: {self.table_scan.table.io}"
                ) from e

    @property
    def npartitions(self):
        return len(self.iceberg_tasks)

    @functools.cached_property
    def _scan_method(self) -> Callable:
        bound_row_filter = bind(
            self.table_scan.table.schema(),
            self.table_scan.row_filter,
            case_sensitive=self.table_scan.case_sensitive,
        )
        projected_field_ids = {
            id
            for id in self.projected_schema.field_ids
            if not isinstance(self.projected_schema.find_type(id), (MapType, ListType))
        }.union(extract_field_ids(bound_row_filter))

        return functools.partial(
            _task_to_table,
            fs=self.fs,
            bound_row_filter=bound_row_filter,
            projected_schema=self.projected_schema,
            projected_field_ids=projected_field_ids,
            positional_deletes=None,
            case_sensitive=self.table_scan.case_sensitive,
            rows_counter=None,
            limit=None,
        )

    def _filtered_task(self, index: int):
        return (
            apply,
            lambda task: self._scan_method(task=task).to_pandas(),
            (),
            {"task": self.iceberg_tasks[index]},
        )

    def _simplify_up(self, parent):
        return
