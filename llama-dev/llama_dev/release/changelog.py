import json
import re
import subprocess
from collections import defaultdict

import click


def run_command(command: str) -> str:
    """Helper to run a shell command and return the output."""
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {command}\n{result.stderr}")
    return result.stdout.strip()


@click.command(short_help="Generate the changelog from the previous release tag")
@click.pass_obj
def changelog(obj: dict) -> None:
    """
    Generate the changelog in markdown syntax.

    This command will:
        - get the list of GitHub Pull Requests that happened since the last tag `v1.2.3`
        - create a bullet list in Markdown syntax using the PR titles
        - group the changes depending on the path that changed according to the format:
           - `llama_index/llama-index-core`
           - `llama_index/llama-index-integrations/<INTEGRATION_NAME>`
           - `llama_index/llama-dev`
    """
    console = obj["console"]
    repo_root = obj["repo_root"]

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

        groups = defaultdict(list)
        integration_pattern = re.compile(r"^llama-index-integrations/[^/]+/([^/]+)")

        with click.progressbar(sorted(pr_numbers), label="Fetching PR details") as bar:
            for pr_number in bar:
                try:
                    pr_json_str = run_command(
                        f"gh pr view {pr_number} --json number,title,url,files"
                    )
                    pr_data = json.loads(pr_json_str)
                    files = [f["path"] for f in pr_data.get("files", [])]

                    assigned_group = None
                    for file_path in files:
                        if file_path.startswith("llama-index-core/"):
                            assigned_group = "llama-index-core"
                            break
                        elif file_path.startswith("llama-dev/"):
                            assigned_group = "llama-dev"
                            break
                        elif file_path.startswith("llama-index-integrations/"):
                            match = integration_pattern.match(file_path)
                            if match:
                                integration_name = match.group(1)
                                assigned_group = integration_name
                                break

                    if assigned_group:
                        groups[assigned_group].append(pr_data)

                except RuntimeError as e:
                    click.echo(
                        f"\nWarning: Could not fetch details for PR #{pr_number}. {e}",
                        err=True,
                    )
                except FileNotFoundError:
                    click.ClickException(
                        "Error: 'gh' command not found. "
                        "Please ensure it's installed and in your PATH."
                    )

        # Generate the markdown output
        sorted_groups = sorted(groups.keys())
        for group_name in sorted_groups:
            click.echo(f"\n### {group_name}\n")
            prs = sorted(groups[group_name], key=lambda p: p["number"])
            for pr in prs:
                click.echo(f"* [#{pr['number']}]({pr['url']}) {pr['title']}")

    except RuntimeError as e:
        click.echo(f"Error: {e}", err=True)
