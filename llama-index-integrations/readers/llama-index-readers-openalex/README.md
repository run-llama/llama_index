# OpenAlex Reader

```bash
pip install llama-index-readers-openalex
```

This loader will search for papers in OpenAlex and load them in llama-index. The main advantage of using OpenAlex is that you can search the full-text for Open Access papers as well.

## Usage

```python
from llama_index.readers.openalex import OpenAlexReader

openalex_reader = OpenAlexReader(email="shauryr@gmail.com")
query = "biases in large language models"

# changing this to full_text=True will let you search full-text
documents = openalex_reader.load_data(query, full_text=False)
```

## What can it do?

As shown in [demo.ipynb](demo.ipynb) we can get answers with citations.

```python
query = "biases in large language models"
response = query_engine.query(
    "list the biases in large language models in a markdown table"
)
```

#### Output

| Source    | Biases                                                                                           |
| --------- | ------------------------------------------------------------------------------------------------ |
| Source 1  | Data selection bias, social bias (gender, age, sexual orientation, ethnicity, religion, culture) |
| Source 2  | Biases of what is right and wrong to do, reflecting ethical and moral norms of society           |
| Source 3  | Anti-Muslim bias                                                                                 |
| Source 6  | Gender bias                                                                                      |
| Source 9  | Anti-LGBTQ+ bias                                                                                 |
| Source 10 | Potential bias in the output                                                                     |

## Credits

- OpenAlex API details are listed [here](https://docs.openalex.org/how-to-use-the-api/get-lists-of-entities/search-entities)

- Some code adopted from [pyAlex](https://github.com/J535D165/pyalex/blob/435287ac20d84ca047e84c71e2c32a6bb84f61a1/pyalex/api.py#L95)
