"""Internal helpers for publishing simple-store snapshots."""

import logging
import os
import stat
from contextlib import contextmanager
from typing import Iterator, TextIO
from uuid import uuid4

from fsspec import AbstractFileSystem
from fsspec.implementations.local import LocalFileSystem

logger = logging.getLogger(__name__)


@contextmanager
def _atomic_write(persist_path: str, fs: AbstractFileSystem) -> Iterator[TextIO]:
    """
    Stage local snapshots beside their destination before replacing it.

    Remote/custom filesystems and caller-owned transactions retain their own
    write semantics. This provides atomic visibility of one local file, not
    crash durability or a transaction across a StorageContext's files.
    """
    if type(fs) is not LocalFileSystem or fs._intrans:
        with fs.open(persist_path, "w", encoding="utf-8") as stream:
            yield stream
        return

    # Match local open() behavior when the destination is a symlink.
    destination = os.path.realpath(fs._strip_protocol(persist_path))
    if fs.auto_mkdir:
        os.makedirs(os.path.dirname(destination), exist_ok=True)

    mode = None
    try:
        # Check existing-file write permissions without truncating its contents.
        fd = os.open(destination, os.O_WRONLY)
    except FileNotFoundError:
        pass
    else:
        try:
            mode = stat.S_IMODE(os.fstat(fd).st_mode) & 0o777
        finally:
            os.close(fd)

    staged = os.path.join(os.path.dirname(destination), f".llama-{uuid4().hex}.tmp")
    # Exclusive creation uses normal open() permissions/umask for new snapshots.
    stream = fs.open(staged, "x", encoding="utf-8")
    try:
        with stream as file:
            if mode is not None:
                os.chmod(staged, mode)
            yield file
        os.replace(staged, destination)
    finally:
        try:
            os.unlink(staged)
        except FileNotFoundError:
            pass
        except OSError:
            logger.warning("Could not remove staged snapshot %s", staged, exc_info=True)
