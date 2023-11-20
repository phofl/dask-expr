import pytest

from dask_expr import from_pandas
from dask_expr._groupby import GroupByApplyTransformBlockwise
from dask_expr._reductions import TreeReduce
from dask_expr._shuffle import Shuffle
from dask_expr.tests._util import _backend_library, assert_eq, xfail_gpu

# Set DataFrame backend for this module
lib = _backend_library()


@pytest.fixture
def pdf():
    idx = lib.date_range('2000-01-01', periods=12, freq='T')
    pdf = lib.DataFrame({'foo': range(len(idx))}, index=idx)
    yield pdf


@pytest.fixture
def df(pdf):
    yield from_pandas(pdf, npartitions=4)


def test_resample_count(df, pdf):
    r = df.resample('2T').count()
    p = pdf.resample('2T').count()
    breakpoint()
    assert_eq(r, )

