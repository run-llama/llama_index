# VectorSearchQueryResponseItem

An array of objects.

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**embeddings** | **List[float]** | The embeddings for the text. | [optional] 
**id** | **str** |  | [optional] 
**metadata** | **Dict[str, object]** | A nodes extra metadata. | [optional] 
**node_id** | **str** | A nodes id. | 
**score** | **float** | The similarity score between the node and the query embeddings. | 
**text** | **str** | The text of a node from which the embeddings were generated. | 

## Example

```python
from manager_client.models.vector_search_query_response_item import VectorSearchQueryResponseItem

# TODO update the JSON string below
json = "{}"
# create an instance of VectorSearchQueryResponseItem from a JSON string
vector_search_query_response_item_instance = VectorSearchQueryResponseItem.from_json(json)
# print the JSON string representation of the object
print(VectorSearchQueryResponseItem.to_json())

# convert the object into a dict
vector_search_query_response_item_dict = vector_search_query_response_item_instance.to_dict()
# create an instance of VectorSearchQueryResponseItem from a dict
vector_search_query_response_item_form_dict = vector_search_query_response_item.from_dict(vector_search_query_response_item_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


