# UniProt Reader for LlamaIndex

This package provides a reader for UniProt [Swiss-Prot](https://web.expasy.org/docs/userman.html) format files, allowing you to load protein data into LlamaIndex for further processing and analysis.

## Features

- Efficient parsing of large UniProt files with optional lazy loading.
- Structured output with both text containing entire UniProt record and metadata containing protein ID.
- Configurable field selection

## Installation

```bash
pip install llama-index-readers-uniprot
```

## Usage

```python
from llama_index.readers.uniprot import UniProtReader

# Initialize the reader
reader = UniProtReader()

# Load data from a UniProt file
documents = reader.load_data("path/to/uniprot_sprot.dat")

# Access the documents
for doc in documents:
    print(f"Protein ID: {doc.metadata['id']}")
```

### Lazy Loading for Large Files

Since UniProt files are large (several GB) it's recommended to use lazy loading to process records one at a time,
without loading the entire database into memory:

```python
# Initialize the reader
reader = UniProtReader()

# Load data lazily from a UniProt file
for doc in reader.lazy_load_data("path/to/uniprot_sprot.dat"):
    print(f"Protein ID: {doc.metadata['id']}")
    print("---")
```

### Example of building an index from a lazy loaded UniProt file

```Python
from llama_index.readers.uniprot import UniProtReader
from llama_index.core import VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter

reader = UniProtReader(max_records=10000)

# Load existing protein IDs from the index
existing_protein_ids = {
    node.metadata.get('id')
    for node in index.storage_context.docstore.docs.values()
    if node.metadata.get('id')
}

text_splitter = SentenceSplitter(chunk_size=2048)
index = VectorStoreIndex([], transformations=[text_splitter], show_progress=True)
documents_gen = reader.lazy_load_data("path/to/uniprot_sprot.dat")

# Process documents in batches
batch_size = 10
current_batch = []

for doc in documents_gen:
  protein_id = doc.metadata.get('id')

  if protein_id in existing_protein_ids:
    print(f"Skipping document {protein_id} - already indexed")
    continue


  current_batch.append(doc)

  if len(current_batch) >= batch_size:
      index.refresh_ref_docs(documents=current_batch)
      current_batch = []

# Process any remaining documents
if current_batch:
    index.refresh_ref_docs(documents=current_batch)

# Define persist directory
persist_dir = "path/to/persist/directory"
index.storage_context.persist(persist_dir=persist_dir)
```

### Customizing Field Selection

You can specify which fields to include in the output:

```python
# Only include specific fields
reader = UniProtReader(include_fields={"id", "description", "sequence"})
documents = reader.load_data("path/to/uniprot_sprot.dat")
```

Available fields:

- `id`: Protein identifier
- `accession`: Accession numbers
- `description`: Protein description
- `gene_names`: Gene names
- `organism`: Organism name
- `comments`: Comments and annotations
- `keywords`: Keywords
- `sequence_length`: Length of the protein sequence
- `sequence_mw`: Molecular weight of the protein
- `taxonomy`: Taxonomic classification
- `taxonomy_id`: Taxonomic database identifiers
- `citations`: Literature citations
- `cross_references`: Cross-references to other databases
- `features`: Protein features

By default, all fields are included.

### Limiting Number of Records

You can limit the number of records to parse using the `max_records` parameter:

```python
# Parse only first 1000 records
reader = UniProtReader(max_records=1000)
documents = reader.load_data("path/to/uniprot_sprot.dat")

# Works with lazy loading too
for doc in reader.lazy_load_data(
    "path/to/uniprot_sprot.dat", max_records=1000
):
    print(f"Protein ID: {doc.metadata['id']}")
```

## Contributing

We welcome contributions! Please see our [contributing guidelines](https://github.com/run-llama/llama_index/blob/main/CONTRIBUTING.md) for details.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
