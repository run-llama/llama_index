# SEC DATA DOWNLOADER

```bash
pip install llama-index-readers-sec-filings
```

Please checkout this repo that I am building on SEC Question Answering Agent [SEC-QA](https://github.com/Athe-kunal/SEC-QA-Agent)

This repository downloads all the texts from SEC documents (10-K and 10-Q). Currently, it is not supporting documents that are amended, but that will be added in the near futures.

Install the required dependencies

```
python install -r requirements.txt
```

The SEC Downloader expects 5 attributes

- tickers: It is a list of valid tickers
- amount: Number of documents that you want to download
- filing_type: 10-K or 10-Q filing type
- num_workers: It is for multithreading and multiprocessing. We have multi-threading at the ticker level and multi-processing at the year level for a given ticker
- include_amends: To include amendments or not.

## Usage

```python
from llama_index.readers.sec_filings import SECFilingsLoader

loader = SECFilingsLoader(tickers=["TSLA"], amount=3, filing_type="10-K")
loader.load_data()
```

It will download the data in the following directories and sub-directories

```yaml
- AAPL
  - 2018
  - 10-K.json
  - 2019
  - 10-K.json
  - 2020
  - 10-K.json
  - 2021
  - 10-K.json
  - 10-Q_12.json
  - 2022
  - 10-K.json
  - 10-Q_03.json
  - 10-Q_06.json
  - 10-Q_12.json
  - 2023
  - 10-Q_04.json
- GOOGL
  - 2018
  - 10-K.json
  - 2019
  - 10-K.json
  - 2020
  - 10-K.json
  - 2021
  - 10-K.json
  - 10-Q_09.json
  - 2022
  - 10-K.json
  - 10-Q_03.json
  - 10-Q_06.json
  - 10-Q_09.json
  - 2023
  - 10-Q_03.json
- TSLA
  - 2018
  - 10-K.json
  - 2019
  - 10-K.json
  - 2020
  - 10-K.json
  - 2021
  - 10-K.json
  - 10-KA.json
  - 10-Q_09.json
  - 2022
  - 10-K.json
  - 10-Q_03.json
  - 10-Q_06.json
  - 10-Q_09.json
  - 2023
  - 10-Q_03.json
```

Here for each ticker we have separate folders with 10-K data inside respective years and 10-Q data is saved in the respective year along with the month. `10-Q_03.json` means March data of 10-Q document. Also, the amended documents are stored in their respective year

## EXAMPLES

This loader is can be used with both Langchain and LlamaIndex.

### LlamaIndex

```python
from llama_index.core import VectorStoreIndex, download_loader
from llama_index.core import SimpleDirectoryReader

from llama_index.readers.sec_filings import SECFilingsLoader

loader = SECFilingsLoader(tickers=["TSLA"], amount=3, filing_type="10-K")
loader.load_data()

documents = SimpleDirectoryReader("data\TSLA\2022").load_data()
index = VectorStoreIndex.from_documents(documents)
index.query("What are the risk factors of Tesla for the year 2022?")
```

### Langchain

```python
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import DirectoryLoader
from langchain.indexes import VectorstoreIndexCreator

from llama_index.readers.sec_filings import SECFilingsLoader

loader = SECFilingsLoader(tickers=["TSLA"], amount=3, filing_type="10-K")
loader.load_data()

dir_loader = DirectoryLoader("data\TSLA\2022")

index = VectorstoreIndexCreator().from_loaders([dir_loader])
retriever = index.vectorstore.as_retriever()
qa = RetrievalQA.from_chain_type(
    llm=OpenAI(), chain_type="stuff", retriever=retriever
)

query = "What are the risk factors of Tesla for the year 2022?"
qa.run(query)
```

## REFERENCES

1. Unstructured SEC Filings API: [repo link](https://github.com/Unstructured-IO/pipeline-sec-filings/tree/main)
2. SEC Edgar Downloader: [repo link](https://github.com/jadchaar/sec-edgar-downloader)
