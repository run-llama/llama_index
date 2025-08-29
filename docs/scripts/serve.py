import argparse
import os
import subprocess
import sys

import yaml


def _serve(skip_notebooks=False, skip_reference=False):
    with open("mkdocs.yml") as f:
        config = yaml.safe_load(f)

    if skip_notebooks:
        config["plugins"] = [
            p for p in config["plugins"] if "mkdocs-jupyter" not in p
        ]

    if skip_reference:
        config["plugins"] = [
            p for p in config["plugins"] if "mkdocstrings" not in p
        ]

    tmp_config_path = ".mkdocs.tmp.yml"
    with open(tmp_config_path, "w") as f:
        yaml.safe_dump(config, f)

    try:
        process = subprocess.Popen(
            ["mkdocs", "serve", "-f", tmp_config_path],
            stdout=sys.stdout,
            stderr=sys.stderr,
        )

        process.wait()
    except KeyboardInterrupt:
        pass

    finally:
        os.unlink(tmp_config_path)  # Clean up the temporary file
        if process.poll() is None:
            process.terminate()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--skip-notebooks",
        help="Do not process notebooks (faster build)",
        action="store_true",
    )
    parser.add_argument(
        "--skip-reference",
        help="Do not process API reference (faster build)",
        action="store_true",
    )
    args = parser.parse_args()
    _serve(
        skip_notebooks=args.skip_notebooks, skip_reference=args.skip_reference
    )


if __name__ == "__main__":
    main()
