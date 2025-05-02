import os


def main():
    # clone the llama_deploy repo
    if not os.path.exists("llama_deploy"):
        os.system("git clone https://github.com/run-llama/llama_deploy.git")
        print("Cloned llama_deploy")
    else:
        os.system("git -C llama_deploy pull")
        print("Updated llama_deploy")

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
