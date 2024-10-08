import argparse
import subprocess
import yaml


def _skip_notebook_conversion():
    with open("mkdocs.yml") as f:
        config = yaml.safe_load(f)

    config["plugins"] = [
        p for p in config["plugins"] if "mkdocs-jupyter" not in p
    ]

    try:
        subprocess.run(
            ["mkdocs", "serve", "-f", "-"],
            input=yaml.safe_dump(config).encode("utf-8"),
        )
    except KeyboardInterrupt:
        pass


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
