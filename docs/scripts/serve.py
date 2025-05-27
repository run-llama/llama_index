import argparse
import os
import subprocess
import sys

import yaml


def _skip_notebook_conversion():
    with open("mkdocs.yml") as f:
        config = yaml.safe_load(f)

    config["plugins"] = [
        p for p in config["plugins"] if "mkdocs-jupyter" not in p
    ]

    tmp_config_path = ".mkdocs.tmp.yml"
    with open(tmp_config_path, "w") as f:
        yaml.safe_dump(config, f)

    try:
        process = subprocess.Popen(
            ["mkdocs", "serve", "--dirty", "-f", tmp_config_path],
            stdout=sys.stdout,
            stderr=sys.stderr,
        )

        process.wait()

    finally:
        os.unlink(tmp_config_path)  # Clean up the temporary file
        if process.poll() is None:
            process.terminate()


def _serve():
    try:
        subprocess.run(["mkdocs", "serve", "--dirty"])
    except KeyboardInterrupt:
        pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--skip-notebooks",
        help="Do not process notebooks (faster build)",
        action="store_true",
    )
    args = parser.parse_args()
    if args.skip_notebooks:
        _skip_notebook_conversion()
    else:
        _serve()


if __name__ == "__main__":
    main()
