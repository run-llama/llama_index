import json
import re
import subprocess
from datetime import date

import click

from llama_dev.utils import find_all_packages, get_changed_packages, load_pyproject


def run_command(command: str) -> str:
    """Helper to run a shell command and return the output."""
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {command}\n{result.stderr}")
    return result.stdout.strip()


@click.command(short_help="Generate the changelog from the previous release tag")
@click.option(
    "--dry-run",
    is_flag=True,
    help="Show the changelog text without altering the CHANGELOG.md file",
)
@click.pass_obj
def changelog(obj: dict, dry_run: bool) -> None:
    """
    Generate the changelog in markdown syntax.

    \b
    This command will:
        - get the list of GitHub PRs that happened since the last release tag
        - create a bullet list in Markdown syntax using the PR titles
        - group the changes per package, depending on the path that changed
    """  # noqa
    console = obj["console"]
    repo_root = obj["repo_root"]
    all_packages = find_all_packages(repo_root)

    try:
        # Get the latest git tag
        latest_tag = run_command('git describe --tags --match "v[0-9]*" --abbrev=0')
        console.print(f"Generating changelog since tag '{latest_tag}'...")

        # Get commits since the last tag and extract PR numbers
        log_output = run_command(f'git log {latest_tag}..HEAD --pretty="format:%H %s"')
        pr_numbers = set()
        pr_pattern = re.compile(r"\(#(\d+)\)")
        for line in log_output.splitlines():
            match = pr_pattern.search(line)
            if match:
                pr_numbers.add(match.group(1))

        if not pr_numbers:
            raise click.ClickException("No pull requests found since the last tag.")

        package_prs = {}
        package_versions = {}

        with click.progressbar(sorted(pr_numbers), label="Fetching PR details") as bar:
            for pr_number in bar:
                try:
                    pr_json_str = run_command(
                        f"gh pr view {pr_number} --json number,title,url,files"
                    )
                    pr_data = json.loads(pr_json_str)
                    files = [repo_root / f["path"] for f in pr_data.get("files", [])]

                    changed_packages = get_changed_packages(files, all_packages)
                    for pkg in changed_packages:
                        pkg_name = pkg.name
                        if pkg_name not in package_prs:
                            package_prs[pkg_name] = []
                            package_data = load_pyproject(pkg)
                            ver = package_data["project"]["version"]
                            package_versions[pkg_name] = ver

                        package_prs[pkg_name].append(pr_data)

                except Exception as e:
                    console.print(
                        f"Warning: Could not fetch details for PR #{pr_number}. {e}",
                        style="error",
                    )

        # Generate the markdown output
        changelog_text = f"<!--- generated changelog --->\n\n## [{date.today().strftime('%Y-%m-%d')}]"
        sorted_pkgs = sorted(package_prs.keys())
        for pkg in sorted_pkgs:
            changelog_text += f"\n\n### {pkg} [{package_versions[pkg]}]"
            prs = sorted(package_prs[pkg], key=lambda p: p["number"])
            for pr in prs:
                changelog_text += f"\n- {pr['title']} ([#{pr['number']}]({pr['url']}))"

        if dry_run:
            click.echo(changelog_text)
        else:
            with open(repo_root / "CHANGELOG.md", "r+") as f:
                content = f.read()
                f.seek(0)
                f.truncate()
                f.write(
                    content.replace("<!--- generated changelog --->", changelog_text)
                )

    except FileNotFoundError:
        click.ClickException(
            "Error: 'gh' command not found. "
            "Please ensure it's installed and in your PATH."
        )
