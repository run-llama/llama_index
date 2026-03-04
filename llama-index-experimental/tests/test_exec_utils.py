import sys

import pytest

from llama_index.experimental.exec_utils import (
    _contains_protected_access,
    safe_eval,
    safe_exec,
)


# ---------------------------------------------------------------------------
# _contains_protected_access -- existing tests (preserved)
# ---------------------------------------------------------------------------


def test_contains_protected_access() -> None:
    assert not _contains_protected_access("def _a(b): pass"), (
        "definition of dunder function"
    )
    assert _contains_protected_access("a = _b(c)"), "call to protected function"
    assert not _contains_protected_access("a = b(c)"), "call to public function"
    assert _contains_protected_access("_b"), "access to protected name"
    assert not _contains_protected_access("b"), "access to public name"
    assert _contains_protected_access("_b[0]"), "subscript access to protected name"
    assert not _contains_protected_access("b[0]"), "subscript access to public name"
    assert _contains_protected_access("_a.b"), "access to attribute of a protected name"
    assert not _contains_protected_access("a.b"), "access to attribute of a public name"
    assert _contains_protected_access("a._b"), "access to protected attribute of a name"
    assert not _contains_protected_access("a.b"), "access to public attribute of a name"


# ---------------------------------------------------------------------------
# I/O operation blocking -- pandas
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "code",
    [
        'pd.read_csv("/etc/passwd")',
        'df.to_csv("/tmp/data.csv")',
        'pd.read_excel("file.xlsx")',
        'df.to_parquet("out.parquet")',
        'pd.read_sql("SELECT 1", conn)',
        'pd.read_pickle("model.pkl")',
        'df.to_pickle("out.pkl")',
        'pd.read_json("data.json")',
    ],
    ids=[
        "read_csv",
        "to_csv",
        "read_excel",
        "to_parquet",
        "read_sql",
        "read_pickle",
        "to_pickle",
        "read_json",
    ],
)
def test_blocks_pandas_io(code: str) -> None:
    assert _contains_protected_access(code)


# ---------------------------------------------------------------------------
# I/O operation blocking -- numpy
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "code",
    [
        'np.load("file.npy")',
        'np.save("file.npy", x)',
        'np.loadtxt("data.txt")',
        'np.savetxt("out.txt", arr)',
        'np.genfromtxt("data.csv")',
        'np.fromfile("raw.bin")',
        'arr.tofile("out.bin")',
    ],
    ids=["load", "save", "loadtxt", "savetxt", "genfromtxt", "fromfile", "tofile"],
)
def test_blocks_numpy_io(code: str) -> None:
    assert _contains_protected_access(code)


# ---------------------------------------------------------------------------
# I/O operation blocking -- polars
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "code",
    [
        'pl.scan_csv("file.csv")',
        'pl.scan_parquet("file.parquet")',
        'df.write_csv("out.csv")',
        'df.write_parquet("out.parquet")',
    ],
    ids=["scan_csv", "scan_parquet", "write_csv", "write_parquet"],
)
def test_blocks_polars_io(code: str) -> None:
    assert _contains_protected_access(code)


# ---------------------------------------------------------------------------
# I/O operation blocking -- general dangerous calls
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "code",
    [
        'os.system("rm -rf /")',
        'os.popen("cat /etc/shadow")',
    ],
    ids=["system", "popen"],
)
def test_blocks_dangerous_system_calls(code: str) -> None:
    assert _contains_protected_access(code)


# ---------------------------------------------------------------------------
# Allowed operations must still work
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "code",
    [
        'df.groupby("col").mean()',
        'df.merge(df2, on="col")',
        'df.sort_values("col")',
        "df.head(10)",
        "df.describe()",
        'df.rename(columns={"a": "b"})',
        "df.dropna()",
        "df.fillna(0)",
    ],
    ids=[
        "groupby",
        "merge",
        "sort_values",
        "head",
        "describe",
        "rename",
        "dropna",
        "fillna",
    ],
)
def test_allows_safe_dataframe_ops(code: str) -> None:
    assert not _contains_protected_access(code)


# ---------------------------------------------------------------------------
# safe_exec / safe_eval smoke tests
# ---------------------------------------------------------------------------


def test_safe_eval_basic() -> None:
    result = safe_eval("1 + 2")
    assert result == 3


def test_safe_exec_basic() -> None:
    local_vars: dict = {}
    safe_exec("x = 1 + 2", __locals=local_vars)
    assert local_vars["x"] == 3


def test_safe_exec_rejects_io() -> None:
    with pytest.raises(RuntimeError, match="forbidden"):
        safe_exec('pd.read_csv("/etc/passwd")')


def test_safe_eval_rejects_io() -> None:
    with pytest.raises(RuntimeError, match="forbidden"):
        safe_eval('pd.read_csv("/etc/passwd")')


# ---------------------------------------------------------------------------
# Timeout enforcement (Unix only)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(sys.platform == "win32", reason="SIGALRM not available on Windows")
def test_safe_exec_timeout() -> None:
    with pytest.raises(TimeoutError, match="time limit"):
        safe_exec("while True: pass", timeout_seconds=1)


@pytest.mark.skipif(sys.platform == "win32", reason="SIGALRM not available on Windows")
def test_safe_eval_timeout() -> None:
    # A generator expression that loops in Python bytecode (interruptible).
    code = "sum(x * x for x in range(10**18))"
    with pytest.raises(TimeoutError, match="time limit"):
        safe_eval(code, timeout_seconds=1)
