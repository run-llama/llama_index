import click
from rich.table import Table

from llama_dev.utils import find_all_packages, is_llama_index_package, load_pyproject


@click.command(short_help="Get package details")
@click.argument("package_names", required=False, nargs=-1)
@click.option(
    "--all",
    is_flag=True,
    help="Get info for all the packages in the monorepo",
)
@click.pass_obj
def info(obj: dict, all: bool, package_names: tuple):
    if not all and not package_names:
        raise click.UsageError("Either specify a package name or use the --all flag")

    packages = set()
    if all:
        packages = find_all_packages(obj["repo_root"])
    else:
        for package_name in package_names:
            package_path = obj["repo_root"] / package_name
            if not is_llama_index_package(package_path):
                raise click.UsageError(
                    f"{package_name} is not a path to a LlamaIndex package"
                )
            packages.add(package_path)

    table = Table(box=None)
    table.add_column("Name")
    table.add_column("Version")
    table.add_column("Path")

    for package in packages:
        package_data = load_pyproject(package)
        table.add_row(
            package_data["project"]["name"],
            package_data["project"]["version"],
            str(package),
        )
    obj["console"].print(table)
