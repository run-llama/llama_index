import os
import subprocess


def get_latest_version_tag(repo_path):
    """Get the latest version tag from the repository."""
    try:
        # Fetch all tags from remote
        subprocess.run(
            ["git", "-C", repo_path, "fetch", "--tags"],
            check=True,
            capture_output=True,
        )

        # Get the latest version tag (sorted by version)
        result = subprocess.run(
            [
                "git",
                "-C",
                repo_path,
                "tag",
                "-l",
                "v*",
                "--sort=-version:refname",
            ],
            check=True,
            capture_output=True,
            text=True,
        )

        tags = result.stdout.strip().split("\n")
        if tags and tags[0]:
            return tags[0]
        else:
            print("No version tags found, falling back to main branch")
            return "main"
    except subprocess.CalledProcessError as e:
        print(f"Error getting latest tag: {e}")
        return "main"


def main():
    # clone the llama_deploy repo
    if not os.path.exists("llama_deploy"):
        os.system("git clone https://github.com/run-llama/llama_deploy.git")
        print("Cloned llama_deploy")
    else:
        print("llama_deploy repo already exists")

    # Get the latest version tag and checkout
    latest_tag = get_latest_version_tag("llama_deploy")
    print(f"Checking out latest version tag: {latest_tag}")

    if latest_tag != "main":
        os.system(f"git -C llama_deploy checkout {latest_tag}")
        print(f"Checked out tag: {latest_tag}")
    else:
        os.system("git -C llama_deploy pull")
        print("Updated to latest main branch")

    # copy the llama_deploy/docs/docs/api_reference/llama_deploy to the current docs/api_reference
    os.system(
        "cp -r llama_deploy/docs/docs/api_reference/llama_deploy ./docs/api_reference/"
    )
    print("Copied in latest llama-deploy reference")

    # copy the module guides
    os.system(
        "cp -r llama_deploy/docs/docs/module_guides/llama_deploy/* ./docs/module_guides/llama_deploy/"
    )
    print("Copied in latest llama-deploy docs")


if __name__ == "__main__":
    main()
