# LlamaIndex Vector_Stores Integration: Moorcheh
Welcome to the Moorcheh Vector Store that integrates Llama-Index.

This module adds official support for Moorcheh, a semantic vector database developed by EdgeAI Innovations, as a vector store integration in LlamaIndex. 

Moorcheh provides fast and intelligent document retrieval using hybrid scoring and generative answer capabilities. For more information on the Moorcheh SDK, visit:  
[https://github.com/mjfekri/moorcheh-python-sdk](https://github.com/mjfekri/moorcheh-python-sdk) 

To see a demonstration of the integration in action, view the example in the demo section or our Google Colab notebook here:  
<https://colab.research.google.com/drive/1iUoMpNYcJxmu1xTySMNJZBPbOQIkUEEs?usp=sharing>. 

Please see below for a short demo of the Moorcheh Vector Store

First, please install the following packages

```
pip install llama_index
pip install moorcheh_sdk
```

Next, run the following code
```
from llama_index.core import VectorStoreIndex
from llama_index.llama_index_integrations.vector_stores.llama_index_vector_stores_moorcheh.llama_index.vector_stores-moorcheh import base, init, utils
```
