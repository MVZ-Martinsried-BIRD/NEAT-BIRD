from __future__ import annotations

import gzip
from pathlib import Path
from typing import TextIO


def open_output(
    file_path: Path, mode: str = "wt", is_gzip: bool = None
) -> TextIO | gzip.GzipFile:
    """
    A helper function that will open a file for writing, and handle gzipping if applicable

    :param file_path: the path to the file to open
    :param mode: the mode to open the file in (default: "wt")
    :param is_gzip: whether to gzip the file (default: None, autodetect from extension)
    :return: an open file handle
    """
    if is_gzip is None:
        is_gzip = file_path.name.endswith((".gz", ".bgz"))

    # Ensure parent directory exists
    file_path.parent.mkdir(parents=True, exist_ok=True)

    if is_gzip:
        # gzip.open mode can be 'rt', 'wt', 'at' for text, or 'rb', 'wb', 'ab' for binary.
        # We assume 'w' means 'wt' and 'a' means 'at' if 'b' is not in mode.
        effective_mode = mode
        if "w" in mode and "b" not in mode and "t" not in mode:
            effective_mode = "wt"
        elif "a" in mode and "b" not in mode and "t" not in mode:
            effective_mode = "at"
        # Binary modes 'wb', 'ab' should pass through correctly.
        return gzip.open(file_path, effective_mode)
    else:
        return open(file_path, mode)


def open_input(
    file_path: Path, mode: str = "rt", is_gzip: bool = None
) -> TextIO | gzip.GzipFile:
    """
    A helper function that will open a file for reading, and handle gzipping if applicable

    :param file_path: the path to the file to open
    :param mode: the mode to open the file in (default: "rt")
    :param is_gzip: whether the file is gzipped (default: None, autodetect from extension)
    :return: an open file handle
    """
    if is_gzip is None:
        is_gzip = file_path.name.endswith((".gz", ".bgz"))

    if is_gzip:
        return gzip.open(file_path, mode)
    else:
        return open(file_path, mode)
