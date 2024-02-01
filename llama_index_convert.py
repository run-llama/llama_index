import argparse
import json

with open("./mappings.json", "r") as f:
    mappings = json.load(f)


def convert_nb_file(file_path, mappings):
    with open(file_path, "r") as f:
        notebook = json.load(f)

    installed_modules = ["llama-index-core"]  # default installs
    cur_cells = []
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            code = cell["source"]

            new_lines = []
            imported_modules = []
            parsing_modules = False
            for line in code:
                if "from llama_index." in line or "from llama_index import" in line:
                    imported_modules = line.split(" import ")[-1].strip()
                    if imported_modules.startswith("("):
                        imported_modules = []
                        parsing_modules = True
                    else:
                        imported_modules = imported_modules.split(", ")

                if parsing_modules:
                    if ")" in line:
                        parsing_modules = False
                    elif "(" not in line:
                        imported_modules.append(line.strip().replace(",", ""))

                if not parsing_modules and len(imported_modules) > 0:
                    imported_modules = [x.strip() for x in imported_modules]
                    new_imports = {}
                    for module in imported_modules:
                        if module in mappings:
                            new_import_parent = mappings[module]
                            if new_import_parent not in new_imports:
                                new_imports[new_import_parent] = [module]
                            else:
                                new_imports[new_import_parent].append(module)
                        else:
                            print(f"Module not found: {module}")

                    new_installs = []
                    for new_import_parent, new_imports in new_imports.items():
                        new_install_parent = new_import_parent.replace(
                            ".", "-"
                        ).replace("_", "-")
                        if new_install_parent not in installed_modules:
                            overlap = [
                                x for x in installed_modules if x in new_install_parent
                            ]
                            if len(overlap) == 0:
                                installed_modules.append(new_install_parent)
                                new_installs.append(
                                    f"!pip install {new_install_parent}\n"
                                )
                        new_imports = ", ".join(new_imports)
                        new_lines.append(
                            f"from {new_import_parent} import {new_imports}\n"
                        )

                        parsing_modules = False
                        new_imports = {}
                        imported_modules = []

                    if len(new_installs) > 0:
                        cur_cells.append(
                            {
                                "cell_type": "code",
                                "execution_count": None,
                                "metadata": {},
                                "outputs": [],
                                "source": new_installs,
                            }
                        )

                elif not parsing_modules:
                    new_lines.append(line)

            cell["source"] = new_lines

        cur_cells.append(cell)

    notebook["cells"] = cur_cells
    with open(file_path, "w") as f:
        json.dump(notebook, f, indent=1, ensure_ascii=False)


def convert_py_file(file_path, mappings):
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert deprecated llama_index imports to new llama_index imports"
    )
    parser.add_argument(
        "file", help="The file to convert. Should be a .ipynb or .py file."
    )

    args = parser.parse_args()

    if args.file.endswith(".ipynb"):
        convert_nb_file(args.file, mappings)
    elif args.file.endswith(".py"):
        convert_py_file(args.file, mappings)
    else:
        print(f"Unsupported file type: {args.file}")
        exit(1)
