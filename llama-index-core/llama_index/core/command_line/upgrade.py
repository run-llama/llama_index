import json
import os
from pathlib import Path
from typing import List, Tuple

mappings_path = os.path.join(os.path.dirname(__file__), "mappings.json")


def parse_lines(lines: List[str], installed_modules: List[str]) -> Tuple[List[str], List[str]]:
    with open(mappings_path, "r") as f:
        mappings = json.load(f)
     
    imported_modules = []
    new_installs = []
    new_lines = []

    parsing_modules = False
    for line in lines:
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
                new_install_parent = new_import_parent.replace(".", "-").replace("_", "-")
                if new_install_parent not in installed_modules:
                    overlap = [x for x in installed_modules if x in new_install_parent]
                    if len(overlap) == 0:
                        installed_modules.append(new_install_parent)
                        new_installs.append(f"!pip install {new_install_parent}\n")
                new_imports = ", ".join(new_imports)
                new_lines.append(f"from {new_import_parent} import {new_imports}\n")
                
                parsing_modules = False
                new_imports = {}
                imported_modules = []
            
        elif not parsing_modules:
            new_lines.append(line)

    return new_lines, new_installs


def upgrade_nb_file(file_path):     
    with open(file_path, "r") as f:
        notebook = json.load(f)
    
    installed_modules = ["llama-index-core"]  # default installs
    cur_cells = []
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            code = cell["source"]

            new_lines, new_installs = parse_lines(code, installed_modules)

            if len(new_installs) > 0:
                cur_cells.append({
                    "cell_type": "code",
                    "metadata": {},
                    "execution_count": None,
                    "outputs": [],
                    "source": new_installs,
                })
            
            cell["source"] = new_lines
        
        cur_cells.append(cell)
    
    notebook["cells"] = cur_cells
    with open(file_path, "w") as f:
        json.dump(notebook, f, indent=1, ensure_ascii=False)



def upgrade_py_file(file_path: str) -> None:
    with open(file_path, "r") as f:
        lines = f.readlines()
    
    installed_modules = ["llama-index-core"]  # default installs
    new_lines, new_installs = parse_lines(lines, installed_modules)

    with open(file_path, "w") as f:
        f.write("".join(new_installs))
        f.write("".join(new_lines))

    if len(new_installs) > 0:
        print("New installs:")
    for install in new_installs:
        print(install.strip().replace("!", ""))


def upgrade_file(file_path: str) -> None:
    if file_path.endswith(".ipynb"):
        upgrade_nb_file(file_path)
    elif file_path.endswith(".py"):
        upgrade_py_file(file_path)
    else:
        raise Exception(f"File type not supported: {file_path}")

def _is_hidden(path: Path) -> bool:
    return any(part.startswith(".") and part not in [".", ".."] for part in path.parts)

def upgrade_dir(input_dir: str) -> None:
    file_refs = [x for x in Path(input_dir).rglob("*.py")]
    file_refs += [x for x in Path(input_dir).rglob("*.ipynb")]
    file_refs = [x for x in file_refs if not _is_hidden(x)]
    for file_ref in file_refs:
        if file_ref.is_file():
            upgrade_file(str(file_ref))
