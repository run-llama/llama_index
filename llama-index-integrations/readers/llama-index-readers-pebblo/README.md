# LlamaIndex Readers Integration: Pebblo

## Pebblo Safe DocumentReader

[Pebblo](https://github.com/daxa-ai/pebblo) enables developers to safely load data and promote their Gen AI app to deployment without worrying about the organization’s compliance and security requirements. The project identifies semantic topics and entities found in the loaded data and summarizes them on the UI or a PDF report.

### Pebblo has two components.

1. Pebblo Safe DocumentReader for Llama
2. Pebblo Daemon

This document describes how to augment your existing Llama DocumentReader with Pebblo Safe DocumentReader to get deep data visibility on the types of Topics and Entities ingested into the Gen-AI Llama application. For details on `Pebblo Daemon` see this [pebblo daemon](https://daxa-ai.github.io/pebblo-docs/daemon.html) document.

Pebblo Safe DocumentReader enables safe data ingestion for Llama `DocumentReader`. This is done by wrapping the document reader call with `Pebblo Safe DocumentReader`

#### How to Pebblo enable Document Reading?

Assume a Llama RAG application snippet using `CSVReader` to read a CSV document for inference.

Here is the snippet of Document loading using `CSVReader`

```
from pathlib import Path
from llama_index.readers.file import CSVReader
reader = CSVReader()
documents = reader.load_data(file=Path('data/corp_sens_data.csv'))
print(documents)
```

The Pebblo SafeReader can be installed and enabled with few lines of code change to the above snippet.

##### Install PebbloSafeReader

```
pip install llama-index-readers-pebblo
```

##### Use PebbloSafeReader

```
from pathlib import Path
from llama_index.readers.pebblo import PebbloSafeReader
from llama_index.readers.file import CSVReader
reader = CSVReader()
pebblo_reader = PebbloSafeReader(reader, name="acme-corp-rag-1", # App name (Mandatory)
owner="Joe Smith", # Owner (Optional)
description="Support productivity RAG application")
documents = pebblo_reader.load_data(file=Path('data/corp_sens_data.csv'))
```
