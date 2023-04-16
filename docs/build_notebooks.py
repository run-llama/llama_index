import json
import os

source_dir = "../examples/"
dest_dir = "./guides/notebooks/"
relative_path = "../../../../examples"


for example_dir in os.listdir(source_dir):
    example_dir_path = os.path.join(source_dir, example_dir)

    for nb_name in os.listdir(example_dir_path):
        if not nb_name.endswith(".ipynb"):
            continue

        # make dest folder in docs
        os.makedirs(os.path.join(dest_dir, example_dir), exist_ok=True)

        # build link text
        relative_nb_path = os.path.join(relative_path, example_dir, nb_name)
        nb_link_text = json.dumps({"path": relative_nb_path})

        # write nbsphinx-link document
        nbsphinx_name = nb_name.replace(".ipynb", ".nblink")
        nbsphinx_path = os.path.join(dest_dir, example_dir, nbsphinx_name)
        with open(nbsphinx_path, "w") as f:
            f.write(nb_link_text)
