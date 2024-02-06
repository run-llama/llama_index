import json
import os
from pathlib import Path
from typing import List, Tuple

mappings_path = os.path.join(os.path.dirname(__file__), "mappings.json")


def parse_lines(
    lines: List[str], installed_modules: List[str]
) -> Tuple[List[str], List[str]]:
    with open(mappings_path) as f:
        mappings = json.load(f)

    imported_modules = []
    new_installs = []
    new_lines = []

    parsing_modules = False
    for line in lines:
        if (
            "from llama_index." in line
            or "from llama_index import" in line
            or "from llama_hub." in line
        ):
            imported_modules = [line, line.split(" import ")[-1].strip()]
            if imported_modules[-1].startswith("("):
                imported_modules[-1] = []
                parsing_modules = True
            else:
                imported_modules[-1] = imported_modules[-1].split(", ")

        if parsing_modules:
            if ")" in line:
                parsing_modules = False
            elif "(" not in line:
                imported_modules[-1].append(line.strip().replace(",", ""))

        if not parsing_modules and len(imported_modules) > 0:
            imported_module_names = [x.strip() for x in imported_modules[-1]]
            new_imports = {}
            for module in imported_module_names:
                if module in mappings:
                    new_import_parent = mappings[module]
                    if new_import_parent not in new_imports:
                        new_imports[new_import_parent] = [module]
                    else:
                        new_imports[new_import_parent].append(module)
                else:
                    print(f"Module not found: {module}\nSwitching to core")
                    new_import_parent = (
                        imported_modules[0]
                        .split(" import ")[0]
                        .split("from ")[-1]
                        .replace("llama_index", "llama_index.core")
                    )
                    if new_import_parent not in new_imports:
                        new_imports[new_import_parent] = [module]
                    else:
                        new_imports[new_import_parent].append(module)

            for new_import_parent, new_imports in new_imports.items():
                new_install_parent = new_import_parent.replace(".", "-").replace(
                    "_", "-"
                )
                if new_install_parent not in installed_modules:
                    overlap = [x for x in installed_modules if x in new_install_parent]
                    if len(overlap) == 0:
                        installed_modules.append(new_install_parent)
                        new_installs.append(f"%pip install {new_install_parent}\n")
                new_imports = ", ".join(new_imports)
                new_lines.append(f"from {new_import_parent} import {new_imports}\n")

                parsing_modules = False
                new_imports = {}
                imported_modules = []

        elif not parsing_modules:
            new_lines.append(line)

    return new_lines, list(set(new_installs))


def _cell_installs_llama_hub(cell) -> bool:
    lines = cell["source"]
    if len(lines) > 1:
        return False
    if cell["cell_type"] == "code" and "pip install llama-hub" in lines[0]:
        return True
    return False


def _format_new_installs(new_installs):
    if new_installs:
        return new_installs[:-1] + [new_installs[-1].replace("\n", "")]
    return new_installs


def upgrade_nb_file(file_path):
    with open(file_path) as f:
        notebook = json.load(f)

    installed_modules = ["llama-index-core"]  # default installs
    cur_cells = []
    new_installs = []
    first_code_idx = -1
    for idx, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            if first_code_idx == -1:
                first_code_idx = idx

            code = cell["source"]

            new_lines, cur_new_installs = parse_lines(code, installed_modules)
            new_installs += cur_new_installs

            cell["source"] = new_lines

        cur_cells.append(cell)

    if len(new_installs) > 0:
        notebook["cells"] = cur_cells
        new_cell = {
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": _format_new_installs(new_installs),
        }
        cur_cells.insert(first_code_idx, new_cell)

    cur_cells = [cell for cell in cur_cells if not _cell_installs_llama_hub(cell)]
    notebook["cells"] = cur_cells
    with open(file_path, "w") as f:
        json.dump(notebook, f, indent=1, ensure_ascii=False)


def upgrade_py_md_file(file_path: str) -> None:
    with open(file_path) as f:
        lines = f.readlines()

    installed_modules = ["llama-index-core"]  # default installs
    new_lines, new_installs = parse_lines(lines, installed_modules)

    with open(file_path, "w") as f:
        f.write("".join(_format_new_installs(new_installs)))
        f.write("".join(new_lines))

    if len(new_installs) > 0:
        print("New installs:")
    for install in new_installs:
        print(install.strip().replace("%", ""))


def upgrade_file(file_path: str) -> None:
    if file_path.endswith(".ipynb"):
        upgrade_nb_file(file_path)
    elif file_path.endswith((".py", ".md")):
        upgrade_py_md_file(file_path)
    else:
        raise Exception(f"File type not supported: {file_path}")


def _is_hidden(path: Path) -> bool:
    return any(part.startswith(".") and part not in [".", ".."] for part in path.parts)


def upgrade_dir(input_dir: str) -> None:
    file_refs = list(Path(input_dir).rglob("*.py"))
    file_refs += list(Path(input_dir).rglob("*.ipynb"))
    file_refs += list(Path(input_dir).rglob("*.md"))
    file_refs = [x for x in file_refs if not _is_hidden(x)]
    for file_ref in file_refs:
        if file_ref.is_file():
            upgrade_file(str(file_ref))
