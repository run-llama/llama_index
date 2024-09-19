import os
import yaml

# clone the llama_deploy repo
if not os.path.exists("llama_deploy"):
    os.system("git clone https://github.com/run-llama/llama_deploy.git")

    print("Cloned llama_deploy")
else:
    os.system("git -C llama_deploy pull")

    print("Updated llama_deploy")

# copy the llama_deploy/docs/docs/api_reference/llama_deploy to the current docs/api_reference
os.system(
    "cp -r llama_deploy/docs/docs/api_reference/llama_deploy ./docs/docs/api_reference/"
)

# copy the module guides
os.system(
    "cp -r llama_deploy/docs/docs/module_guides/workflow/deployment.md ./docs/docs/module_guides/workflow/deployment.md"
)

print("Copied in latest llama-deploy docs")

# open current mkdocs.yml
with open("./docs/mkdocs.yml") as f:
    mkdocs = yaml.safe_load(f)

# open llama-deploy mkdocs.yml
with open("./llama_deploy/docs/mkdocs.yml") as f:
    llama_deploy_mkdocs = yaml.safe_load(f)

llama_docs_api_reference_idx = 0
for idx, item in enumerate(llama_deploy_mkdocs["nav"]):
    if isinstance(item, dict) and "API Reference" in item:
        llama_docs_api_reference_idx = idx
        break

# Add links to llama-deploy api reference to nav
for nav_idx, item in enumerate(mkdocs["nav"]):
    if isinstance(item, dict) and "API Reference" in item:
        api_reference = item["API Reference"]
        for api_ref_idx, api_ref in enumerate(api_reference):
            if isinstance(api_ref, dict) and "LLMs" in api_ref:
                # Find the Llama Deploy API reference in llama_deploy_mkdocs
                break

        api_reference.insert(
            api_ref_idx,
            {
                "Llama Deploy": llama_deploy_mkdocs["nav"][
                    llama_docs_api_reference_idx
                ]["API Reference"]
            },
        )
        break

print("Merged Llama Deploy API Reference")

# Add search paths from llama-deploy mkdocs.yml
mkdocs_plugins_idx = 0
for idx, item in enumerate(mkdocs["plugins"]):
    if isinstance(item, dict) and "mkdocstrings" in item:
        mkdocs_plugins_idx = idx
        break

mkdocs["plugins"][mkdocs_plugins_idx]["mkdocstrings"]["handlers"]["python"][
    "paths"
].append("../llama_deploy")

print("Updated search paths")

# Add the deployment guide to the nav
for value in mkdocs["nav"]:
    if isinstance(value, dict) and "Component Guides" in value:
        for item in value["Component Guides"]:
            if isinstance(item, dict) and "Querying" in item:
                for section in item["Querying"]:
                    if isinstance(section, dict) and "Workflows" in section:
                        if isinstance(section["Workflows"], str):
                            section["Workflows"] = [
                                section["Workflows"],
                                "./module_guides/workflow/deployment.md",
                            ]
                        else:
                            section["Workflows"].append(
                                "./module_guides/workflow/deployment.md"
                            )

print("Updated nav")

# Save the updated mkdocs.yml
with open("./docs/mkdocs.yml", "w") as f:
    yaml.dump(mkdocs, f, sort_keys=False)

print("Updated mkdocs.yml saved")
