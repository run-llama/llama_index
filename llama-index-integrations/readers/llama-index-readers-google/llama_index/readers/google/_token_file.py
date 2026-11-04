"""Write OAuth token files that only the current user can read."""

import contextlib
import os


def write_token_file(path: str, content: str) -> None:
    """
    Save OAuth credentials to ``path`` with owner-only (0o600) permissions.

    On Windows the mode has no effect and the file keeps its default ACL.
    """
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        token = os.fdopen(fd, "w", encoding="utf-8")
    except BaseException:
        os.close(fd)
        raise
    with token:
        if hasattr(os, "fchmod"):
            # The mode given to os.open only applies to a new file, so also
            # tighten one that an earlier version left readable. Best effort:
            # a file we can write but do not own cannot be chmod-ed.
            with contextlib.suppress(OSError):
                os.fchmod(fd, 0o600)
        token.write(content)
