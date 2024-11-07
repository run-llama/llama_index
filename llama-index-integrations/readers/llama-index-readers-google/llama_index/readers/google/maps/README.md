# Google Maps Text Search Reader

`pip install llama-index-readers-google`

This loader reads your upcoming Google Calendar events and parses the relevant info into `Documents`.

As a prerequisite, you will need to create Google Maps API credentials and set up a billing account. See [here](https://developers.google.com/maps/gmp-get-started) for instructions. Please also enable the [Places API (New)](https://console.cloud.google.com/apis/library/places.googleapis.com) to enable the text search for your API key.

[Here](https://developers.google.com/maps/documentation/places/web-service/text-search). is the detailed information on text search.

## Usage

Here's an example usage of the GoogleMapsTextSearchReader.

You can set API key by providing `api_key` parameter

```bash
export GOOGLE_MAPS_API_KEY="YOUR_API_KEY"
```

or setting `GOOGLE_MAPS_API_KEY` environment variable.

```python
import os

os.environ["GOOGLE_MAPS_API_KEY"] = "YOUR_API_KEY"
```

Provide `number_of_results` to specify the number of results you want to get from the API. The default value is 20.

Provide `text` to search for places based on the text query.

## Example

This loader is designed to be used as a way to load data into [LlamaIndex](https://github.com/run-llama/llama_index/tree/main/llama_index).

### LlamaIndex

```python
from llama_index.readers.google import GoogleMapsTextSearchReader
from llama_index.core import VectorStoreIndex

loader = GoogleMapsTextSearchReader()
documents = loader.load_data(
    text="I want to eat quality Turkish food in Istanbul",
    number_of_results=160,
)


index = VectorStoreIndex.from_documents(documents)
index.query("Which Turkish restaurant has the best reviews?")
```

## Attribution

Please ensure to provide the necessary attribution when using Google Maps data. For more information, refer to the [Google Maps Platform Branding Guidelines](https://developers.google.com/maps/documentation/urls/branding).
