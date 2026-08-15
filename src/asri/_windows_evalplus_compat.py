"""Windows compatibility shims for evalplus code execution.

evalplus's execution guard and per-test timeout rely on Unix-only facilities:

- ``import resource`` for RLIMIT-based memory caps (does not exist on Windows)
- ``signal.setitimer`` / ``signal.SIGALRM`` / ``signal.ITIMER_REAL`` for
  per-input timeouts (not implemented on Windows)

evalplus runs each candidate in a spawned child process, so these shims must
be installed in *every* interpreter of the virtual environment, not just the
process that imports :mod:`asri`. The editable install registers
``zz_asri_windows_compat.pth`` in site-packages, which imports this module at
interpreter startup.

Semantics preserved:
- RLIMIT caps do not exist on Windows; the stub makes the guard's remaining
  protections (syscall disabling, io swallowing) still apply. evalplus's own
  per-task wall-clock kill in ``untrusted_check`` remains the hard backstop.
- The alarm emulation raises an asynchronous exception in the main thread,
  which lands in evalplus's ``except BaseException`` per-input handler exactly
  like the Unix TimeoutException path. Non-Python (C-level) hangs are caught
  by the parent's per-task timeout instead.

This module is a no-op on non-Windows platforms.
"""

from __future__ import annotations

import sys


def install() -> None:
    if sys.platform != "win32":
        return

    _install_resource_stub()
    _install_signal_alarm_emulation()


def _install_resource_stub() -> None:
    import types

    if "resource" in sys.modules:
        return
    stub = types.ModuleType("resource")
    stub.RLIMIT_AS = 0
    stub.RLIMIT_DATA = 0
    stub.RLIMIT_STACK = 0
    stub.setrlimit = lambda *args, **kwargs: None
    sys.modules["resource"] = stub


def _install_signal_alarm_emulation() -> None:
    import ctypes
    import signal
    import threading
    import types

    if hasattr(signal, "setitimer") and hasattr(signal, "SIGALRM"):
        return  # genuine POSIX interval timers available

    if not hasattr(signal, "SIGALRM"):
        signal.SIGALRM = 14  # conventional POSIX value, for identity checks only
    if not hasattr(signal, "ITIMER_REAL"):
        signal.ITIMER_REAL = 0

    state = {"timer": None, "main_thread_id": threading.main_thread().ident}

    class _WindowsAlarm(BaseException):
        """Asynchronous alarm, equivalent to a POSIX SIGALRM escape."""

    def _setitimer(which, seconds, *_args):  # noqa: ANN001
        if which != signal.ITIMER_REAL and which != signal.SIGALRM:
            return None
        previous = state["timer"]
        if previous is not None:
            previous.cancel()
            state["timer"] = None
        if seconds and seconds > 0:
            def _fire():
                ctypes.pythonapi.PyThreadState_SetAsyncExc(
                    ctypes.c_long(state["main_thread_id"]),
                    ctypes.py_object(_WindowsAlarm),
                )

            timer = threading.Timer(float(seconds), _fire)
            timer.daemon = True
            state["timer"] = timer
            timer.start()
        return None

    signal.setitimer = _setitimer  # type: ignore[attr-defined]

    _orig_signal = signal.signal

    def _signal(signum, handler):  # noqa: ANN001
        if signum in (signal.SIGALRM,) and signum not in _VALID_WINDOWS_SIGNALS:
            # Alarm delivery is emulated above; registering the handler is a no-op.
            return None
        return _orig_signal(signum, handler)

    _VALID_WINDOWS_SIGNALS = _valid_windows_signals()
    signal.signal = _signal  # type: ignore[assignment]


def _valid_windows_signals() -> frozenset:
    import signal

    candidates = (
        "SIGABRT", "SIGFPE", "SIGILL", "SIGINT", "SIGSEGV",
        "SIGTERM", "SIGBREAK", "SIGCTRL_C", "SIGCTRL_BREAK",
    )
    valid = set()
    for name in candidates:
        value = getattr(signal, name, None)
        if value is not None:
            try:
                signal.getsignal(value)
                valid.add(value)
            except (ValueError, OSError, RuntimeError):
                continue
    return frozenset(valid)


install()
