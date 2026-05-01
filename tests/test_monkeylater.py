from __future__ import annotations

import re
import time
import warnings
from concurrent.futures import Future
from types import SimpleNamespace

import pytest

import monkeylater


@pytest.fixture(autouse=True)
def restore_global_patches() -> None:
    try:
        monkeylater.restore_all()
    except BaseException:
        pass
    yield
    try:
        monkeylater.restore_all()
    except BaseException:
        pass


def test_scoped_patch_returns_future_waits_and_restores() -> None:
    calls: list[str] = []

    def write(value: str) -> str:
        time.sleep(0.01)
        calls.append(value)
        return "done"

    owner = SimpleNamespace(write=write)
    original = owner.write

    with pytest.warns(RuntimeWarning, match=re.escape(monkeylater.PATCH_WARNING)):
        with monkeylater.patch((owner, "write")):
            future = owner.write("item")
            assert isinstance(future, Future)

    assert future.done()
    assert future.result() == "done"
    assert calls == ["item"]
    assert owner.write is original


def test_global_patch_flushes_and_restores() -> None:
    calls: list[str] = []

    def write(value: str) -> None:
        calls.append(value)

    owner = SimpleNamespace(write=write)
    original = owner.write

    with pytest.warns(RuntimeWarning, match=re.escape(monkeylater.PATCH_WARNING)):
        monkeylater.patch_global((owner, "write"))

    future = owner.write("item")

    assert isinstance(future, Future)
    assert owner.write is not original

    monkeylater.flush()
    assert calls == ["item"]

    monkeylater.restore_all()
    assert owner.write is original


def test_background_errors_surface_through_future_and_flush() -> None:
    def fail() -> None:
        raise ValueError("boom")

    owner = SimpleNamespace(fail=fail)

    with pytest.warns(RuntimeWarning, match=re.escape(monkeylater.PATCH_WARNING)):
        monkeylater.patch_global((owner, "fail"))

    future = owner.fail()

    with pytest.raises(ValueError, match="boom"):
        future.result()
    with pytest.raises(ValueError, match="boom"):
        monkeylater.flush()

    monkeylater.restore_all()


def test_scoped_exit_surfaces_background_errors_after_restore() -> None:
    def fail() -> None:
        raise RuntimeError("later")

    owner = SimpleNamespace(fail=fail)
    original = owner.fail

    with pytest.raises(RuntimeError, match="later"):
        with pytest.warns(RuntimeWarning, match=re.escape(monkeylater.PATCH_WARNING)):
            with monkeylater.patch((owner, "fail")):
                owner.fail()

    assert owner.fail is original


def test_duplicate_patch_raises_runtime_error() -> None:
    def write() -> None:
        return None

    owner = SimpleNamespace(write=write)

    with pytest.warns(RuntimeWarning, match=re.escape(monkeylater.PATCH_WARNING)):
        with monkeylater.patch((owner, "write")):
            with pytest.raises(RuntimeError, match="already patched"):
                with monkeylater.patch((owner, "write")):
                    pass


def test_duplicate_target_in_same_call_raises_runtime_error() -> None:
    def write() -> None:
        return None

    owner = SimpleNamespace(write=write)

    with pytest.raises(RuntimeError, match="passed more than once"):
        with monkeylater.patch((owner, "write"), (owner, "write")):
            pass


def test_warning_as_error_does_not_leave_target_patched() -> None:
    def write() -> str:
        return "sync"

    owner = SimpleNamespace(write=write)
    original = owner.write

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        with pytest.raises(RuntimeWarning, match="patched functions no longer execute synchronously"):
            with monkeylater.patch((owner, "write")):
                pass

    assert owner.write is original
    assert owner.write() == "sync"


@pytest.mark.parametrize(
    ("target", "error_type"),
    [
        ("not-a-target", TypeError),
        ((SimpleNamespace(), 123), TypeError),
        ((SimpleNamespace(), "missing"), AttributeError),
        ((SimpleNamespace(value=1), "value"), TypeError),
    ],
)
def test_invalid_targets_raise_clear_errors(target: object, error_type: type[Exception]) -> None:
    with pytest.raises(error_type):
        with monkeylater.patch(target):  # type: ignore[arg-type]
            pass
