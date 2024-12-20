# Protein Data Bank (PDB) publication Loader

```bash
pip install llama-index-readers-pdb
```

This loader fetches the abstract of PDB entries using the RCSB (Research Collaboratory for Structural Bioinformatics) or EBI (European Bioinformatics Institute) REST api.

## Usage

To use this loader, simply pass an array of PDB ids into `load_data`:

```python
from llama_index.readers.pdb import PdbAbstractReader

loader = PdbAbstractReader()
documents = loader.load_data(pdb_id=["1cbs"])
```

This loader is designed to be used as a way to load data into [LlamaIndex](https://github.com/run-llama/llama_index/).
