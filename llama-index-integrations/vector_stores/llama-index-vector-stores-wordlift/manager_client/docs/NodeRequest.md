# NodeRequest

A node request.

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**embeddings** | **List[float]** | A list of embeddings. | [optional] 
**entity_id** | **str** | The entity id in the form on an IRI, e.g. https://data.example.org/dataset/entity. | 
**metadata** | **Dict[str, object]** | A map of metadata properties. | [optional] 
**node_id** | **str** | The node id generally expressed in the form of a UUID. | 
**text** | **str** | The original text. | [optional] 

## Example

```python
from manager_client.models.node_request import NodeRequest

# TODO update the JSON string below
json = "{}"
# create an instance of NodeRequest from a JSON string
node_request_instance = NodeRequest.from_json(json)
# print the JSON string representation of the object
print(NodeRequest.to_json())

# convert the object into a dict
node_request_dict = node_request_instance.to_dict()
# create an instance of NodeRequest from a dict
node_request_form_dict = node_request.from_dict(node_request_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


