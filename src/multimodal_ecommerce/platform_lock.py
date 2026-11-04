"""Expose the same nonblocking byte-range lock on Windows and POSIX hosts."""

from __future__ import annotations

import os
import sys

if sys.platform == "win32":
    import msvcrt as _windows_lock  # pylint: disable=import-error  # Windows-only standard library.
else:
    import fcntl as _posix_lock

LK_NBLCK = 2
LK_UNLCK = 0


def locking(fd: int, mode: int, length: int) -> None:
    """Lock or unlock bytes starting at the descriptor's current offset.

    Args:
        fd: Open file descriptor. The caller owns and closes it.
        mode: LK_NBLCK for an exclusive nonblocking lock, or LK_UNLCK to release.
        length: Number of bytes covered by the lock.

    Raises:
        ValueError: The lock mode is unsupported or the byte count is not positive.
        OSError: The descriptor is invalid or another process holds the lock.
    """
    if mode not in (LK_NBLCK, LK_UNLCK) or length <= 0:
        raise ValueError("Expected a supported lock mode and a positive byte count")
    if sys.platform == "win32":
        _windows_lock.locking(fd, mode, length)
    else:
        operation = (
            _posix_lock.LOCK_UN
            if mode == LK_UNLCK
            else _posix_lock.LOCK_EX | _posix_lock.LOCK_NB
        )
        _posix_lock.lockf(fd, operation, length, 0, os.SEEK_CUR)
