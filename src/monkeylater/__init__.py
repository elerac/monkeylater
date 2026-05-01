from __future__ import annotations

import atexit
import threading
import warnings
from collections.abc import Callable, Iterable
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from functools import wraps
from types import TracebackType
from typing import Any, TypeAlias

PatchTarget: TypeAlias = tuple[Any, str]

PATCH_WARNING = (
    "Warning: patched functions no longer execute synchronously. Return values "
    "become Future objects, exceptions are deferred until flush/result, and "
    "mutable inputs may need copying. Use this for experiments, logging, and "
    "non-critical output paths."
)

__all__ = [
    "PATCH_WARNING",
    "PatchTarget",
    "flush",
    "patch",
    "patch_global",
    "restore_all",
]

_registry_lock = threading.RLock()
_active_targets: dict[tuple[int, str], "_PatchManager"] = {}
_global_manager: _PatchManager | None = None
_atexit_registered = False


@dataclass(frozen=True)
class _PatchRecord:
    owner: Any
    name: str
    original: Callable[..., Any]


def patch(*targets: PatchTarget) -> "_PatchManager":
    """Patch functions for the lifetime of a ``with`` block."""
    return _PatchManager(targets, is_global=False)


def patch_global(*targets: PatchTarget) -> None:
    """Patch functions process-wide until ``restore_all()`` or interpreter exit."""
    global _global_manager

    with _registry_lock:
        created_manager = False
        if _global_manager is None or _global_manager.closed:
            _global_manager = _PatchManager((), is_global=True)
            created_manager = True
        try:
            _global_manager.add_targets(targets)
        except BaseException:
            if created_manager:
                _global_manager.close(raise_errors=False)
                _global_manager = None
            raise
        else:
            _register_atexit_locked()


def flush() -> None:
    """Wait for all pending global background work and surface deferred errors."""
    with _registry_lock:
        manager = _global_manager

    if manager is not None:
        manager.flush()


def restore_all() -> None:
    """Restore all global monkey patches after waiting for pending work."""
    global _global_manager

    with _registry_lock:
        manager = _global_manager

    try:
        if manager is not None:
            manager.close()
    finally:
        with _registry_lock:
            if _global_manager is manager:
                _global_manager = None


class _PatchManager:
    def __init__(self, targets: Iterable[PatchTarget], *, is_global: bool) -> None:
        self._initial_targets = tuple(targets)
        self._is_global = is_global
        self._executor = ThreadPoolExecutor()
        self._owns_executor = True
        self._records: list[_PatchRecord] = []
        self._futures: list[Future[Any]] = []
        self._lock = threading.RLock()
        self._entered = False
        self.closed = False

    def __enter__(self) -> "_PatchManager":
        with self._lock:
            if self._entered:
                raise RuntimeError("Monkeylater patch scopes cannot be reused.")
            self._entered = True
        try:
            self.add_targets(self._initial_targets)
        except BaseException:
            with self._lock:
                self.closed = True
            if self._owns_executor:
                self._executor.shutdown(wait=False)
            raise
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close(raise_errors=exc_type is None)
        return None

    def add_targets(self, targets: Iterable[PatchTarget]) -> None:
        records = [_validate_target(target) for target in targets]
        if not records:
            return

        with _registry_lock:
            if self.closed:
                raise RuntimeError("Cannot add targets to a closed Monkeylater patch.")
            seen: set[tuple[int, str]] = set()
            for record in records:
                key = _target_key(record.owner, record.name)
                if key in seen:
                    raise RuntimeError(
                        f"{_target_label(record.owner, record.name)} was passed more than once."
                    )
                seen.add(key)
                if key in _active_targets:
                    raise RuntimeError(
                        f"{_target_label(record.owner, record.name)} is already patched by Monkeylater."
                    )

            warnings.warn(PATCH_WARNING, RuntimeWarning, stacklevel=3)

            applied: list[_PatchRecord] = []
            try:
                for record in records:
                    setattr(record.owner, record.name, self._make_wrapper(record))
                    _active_targets[_target_key(record.owner, record.name)] = self
                    self._records.append(record)
                    applied.append(record)
            except BaseException:
                for record in reversed(applied):
                    current = getattr(record.owner, record.name, None)
                    if getattr(current, "__monkeylater_manager__", None) is self:
                        setattr(record.owner, record.name, record.original)
                    _active_targets.pop(_target_key(record.owner, record.name), None)
                    self._records.remove(record)
                raise

    def flush(self) -> None:
        with self._lock:
            futures = tuple(self._futures)

        errors: list[BaseException] = []
        completed: set[Future[Any]] = set()
        for future in as_completed(futures):
            completed.add(future)
            try:
                future.result()
            except BaseException as exc:
                errors.append(exc)

        if completed:
            with self._lock:
                self._futures = [future for future in self._futures if future not in completed]

        if not errors:
            return
        if len(errors) == 1:
            raise errors[0]
        raise ExceptionGroup("Multiple Monkeylater background calls failed.", errors)

    def close(self, *, raise_errors: bool = True) -> None:
        with self._lock:
            if self.closed:
                return
            self.closed = True

        error: BaseException | None = None
        try:
            self.flush()
        except BaseException as exc:
            error = exc
        finally:
            with _registry_lock:
                for record in reversed(self._records):
                    current = getattr(record.owner, record.name, None)
                    if getattr(current, "__monkeylater_manager__", None) is self:
                        setattr(record.owner, record.name, record.original)
                    _active_targets.pop(_target_key(record.owner, record.name), None)
                self._records.clear()

            if self._owns_executor:
                self._executor.shutdown(wait=True)

        if error is not None and raise_errors:
            raise error

    def _make_wrapper(self, record: _PatchRecord) -> Callable[..., Future[Any]]:
        original = record.original

        @wraps(original)
        def wrapper(*args: Any, **kwargs: Any) -> Future[Any]:
            with self._lock:
                if self.closed:
                    raise RuntimeError(
                        f"{_target_label(record.owner, record.name)} was called through a closed "
                        "Monkeylater patch."
                    )
                future = self._executor.submit(original, *args, **kwargs)
                self._futures.append(future)
                return future

        wrapper.__monkeylater_manager__ = self  # type: ignore[attr-defined]
        return wrapper


def _validate_target(target: PatchTarget) -> _PatchRecord:
    if not isinstance(target, tuple) or len(target) != 2:
        raise TypeError("Patch targets must be explicit tuples: (owner, 'name').")

    owner, name = target
    if not isinstance(name, str):
        raise TypeError("Patch target names must be strings.")
    if not hasattr(owner, name):
        raise AttributeError(f"{owner!r} has no attribute {name!r}.")

    original = getattr(owner, name)
    if not callable(original):
        raise TypeError(f"{_target_label(owner, name)} is not callable.")
    if getattr(original, "__monkeylater_manager__", None) is not None:
        raise RuntimeError(f"{_target_label(owner, name)} is already patched by Monkeylater.")

    return _PatchRecord(owner=owner, name=name, original=original)


def _target_key(owner: Any, name: str) -> tuple[int, str]:
    return (id(owner), name)


def _target_label(owner: Any, name: str) -> str:
    module = getattr(owner, "__name__", owner.__class__.__name__)
    return f"{module}.{name}"


def _register_atexit_locked() -> None:
    global _atexit_registered

    if not _atexit_registered:
        atexit.register(_shutdown_global)
        _atexit_registered = True


def _shutdown_global() -> None:
    try:
        restore_all()
    except BaseException as exc:
        warnings.warn(f"Monkeylater failed during interpreter shutdown: {exc!r}", RuntimeWarning)
