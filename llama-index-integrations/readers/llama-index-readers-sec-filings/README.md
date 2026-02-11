# SEC DATA READER

```bash
pip install llama-index-readers-sec-filings
```

Please checkout this repo that I am building on SEC Question Answering Agent [SEC-QA](https://github.com/Athe-kunal/SEC-QA-Agent)

This package provides two readers for SEC filings:

1. **SECFilingsLoader** - Downloads filings to local files (original loader)
2. **SECFilingsStreamingReader** - Streams filings directly without local storage (NEW!)

## SECFilingsStreamingReader (Recommended)

The streaming reader provides several advantages:

- **No local storage**: Documents are returned directly in memory
- **8-K Support**: Full support for 8-K current report filings
- **Structured Section Extraction**: Extract specific sections like Item 1A (Risk Factors)
- **Rich Metadata**: CIK, company name, filing date, accession number

### Basic Usage

```python
from llama_index.readers.sec_filings import SECFilingsStreamingReader

# Fetch 10-K and 8-K filings for Apple and Microsoft
reader = SECFilingsStreamingReader(
    tickers=["AAPL", "MSFT"],
    filing_types=["10-K", "8-K"],
    num_filings=5,
)

documents = reader.load_data()

# Each document has rich metadata
for doc in documents:
    print(f"Company: {doc.metadata['company_name']}")
    print(f"CIK: {doc.metadata['cik']}")
    print(f"Filing Type: {doc.metadata['filing_type']}")
    print(f"Filing Date: {doc.metadata['filing_date']}")
    print(f"Accession Number: {doc.metadata['accession_number']}")
```

### Extract Specific Sections

```python
# For 10-K filings: Extract Risk Factors and MD&A
reader = SECFilingsStreamingReader(
    tickers=["AAPL"],
    filing_types=["10-K"],
    num_filings=3,
    sections=["ITEM_1A", "ITEM_7"],  # Risk Factors and Management's Discussion
)
documents = reader.load_data()

# For 8-K filings: Extract specific items
reader = SECFilingsStreamingReader(
    tickers=["AAPL"],
    filing_types=["8-K"],
    num_filings=10,
    sections=[
        "2.02",
        "7.01",
        "8.01",
    ],  # Results of Operations, Reg FD, Other Events
)
documents = reader.load_data()
```

### Available Sections

**10-K Sections:**

- `ITEM_1` - Business
- `ITEM_1A` - Risk Factors
- `ITEM_1B` - Unresolved Staff Comments
- `ITEM_2` - Properties
- `ITEM_3` - Legal Proceedings
- `ITEM_7` - Management's Discussion and Analysis
- `ITEM_7A` - Quantitative and Qualitative Disclosures About Market Risk
- `ITEM_8` - Financial Statements
- And more...

**10-Q Sections:**

- `PART_I_ITEM_1` - Financial Statements
- `PART_I_ITEM_2` - Management's Discussion and Analysis
- `PART_I_ITEM_3` - Quantitative and Qualitative Disclosures About Market Risk
- `PART_II_ITEM_1A` - Risk Factors
- And more...

**8-K Items:**

- `1.01` - Entry into a Material Definitive Agreement
- `2.02` - Results of Operations and Financial Condition
- `5.02` - Departure of Directors or Certain Officers
- `7.01` - Regulation FD Disclosure
- `8.01` - Other Events
- And more...

### Filter by Date

```python
reader = SECFilingsStreamingReader(
    tickers=["TSLA"],
    filing_types=["10-K"],
    num_filings=10,
    start_date="2020-01-01",
    end_date="2023-12-31",
)
```

### Include Amended Filings

```python
reader = SECFilingsStreamingReader(
    tickers=["TSLA"],
    filing_types=["10-K"],
    num_filings=5,
    include_amends=True,  # Include 10-K/A filings
)
```

---

## SECFilingsLoader (Original)

This reader downloads all the texts from SEC documents (10-K and 10-Q). Currently, it is not supporting documents that are amended, but that will be added in the near futures.

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
