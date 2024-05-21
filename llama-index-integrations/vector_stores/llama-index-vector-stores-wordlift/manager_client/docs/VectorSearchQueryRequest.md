# VectorSearchQueryRequest

A query request.

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**filters** | [**List[Filter]**](Filter.md) | A list of prefilters. | [optional] 
**query_embedding** | **List[float]** | The list of embeddings, not required if &#x60;query_string&#x60; is provided. | [optional] 
**query_string** | **str** | The query string, not required if the &#x60;query_embeddings&#x60; are provided. Please note that the &#x60;query_string&#x60; is ignored if the &#x60;query_embeddings&#x60; are provided. | [optional] 
**similarity_top_k** | **int** | The similarity top K. | [optional] [default to 2]

## Example

```python
from manager_client.models.vector_search_query_request import VectorSearchQueryRequest

# TODO update the JSON string below
json = "{}"
# create an instance of VectorSearchQueryRequest from a JSON string
vector_search_query_request_instance = VectorSearchQueryRequest.from_json(json)
# print the JSON string representation of the object
print(VectorSearchQueryRequest.to_json())

# convert the object into a dict
vector_search_query_request_dict = vector_search_query_request_instance.to_dict()
# create an instance of VectorSearchQueryRequest from a dict
vector_search_query_request_form_dict = vector_search_query_request.from_dict(vector_search_query_request_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


