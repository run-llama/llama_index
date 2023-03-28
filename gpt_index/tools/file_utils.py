"""File utility."""
import os


def add_prefix_suffix_to_file_path(
    in_path: str, prefix: str = "", suffix: str = ""
) -> str:
    assert prefix or suffix, "Neither prefix nor suffix is specified."
    # Split the file path into directory, base filename, and extension
    directory, filename = os.path.split(in_path)
    base_filename, ext = os.path.splitext(filename)

    # Add the prefix to the base filename
    new_base_filename = prefix + base_filename + suffix

    # Combine the modified base filename and the extension
    new_filename = new_base_filename + ext

    # Combine the directory and the new filename to get the output path
    out_path = os.path.join(directory, new_filename)

    return out_path
