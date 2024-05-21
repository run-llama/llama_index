# manager_client.VectorSearchNodesApi

All URIs are relative to *https://api.wordlift.io*

Method | HTTP request | Description
------------- | ------------- | -------------
[**update_nodes_collection**](VectorSearchNodesApi.md#update_nodes_collection) | **PUT** /vector-search/nodes-collection | Update


# **update_nodes_collection**
> update_nodes_collection(node_request)

Update

### Example

* Api Key Authentication (ApiKey):

```python
import manager_client
from manager_client.models.node_request import NodeRequest
from manager_client.rest import ApiException
from pprint import pprint

# Defining the host is optional and defaults to https://api.wordlift.io
# See configuration.py for a list of all supported configuration parameters.
configuration = manager_client.Configuration(
    host = "https://api.wordlift.io"
)

# The client must configure the authentication and authorization parameters
# in accordance with the API server security policy.
# Examples for each auth method are provided below, use the example that
# satisfies your auth use case.

# Configure API key authorization: ApiKey
configuration.api_key['ApiKey'] = os.environ["API_KEY"]

# Uncomment below to setup prefix (e.g. Bearer) for API key, if needed
# configuration.api_key_prefix['ApiKey'] = 'Bearer'

# Enter a context with an instance of the API client
async with manager_client.ApiClient(configuration) as api_client:
    # Create an instance of the API class
    api_instance = manager_client.VectorSearchNodesApi(api_client)
    node_request = [manager_client.NodeRequest()] # List[NodeRequest] | 

    try:
        # Update
        await api_instance.update_nodes_collection(node_request)
    except Exception as e:
        print("Exception when calling VectorSearchNodesApi->update_nodes_collection: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **node_request** | [**List[NodeRequest]**](NodeRequest.md)|  | 

### Return type

void (empty response body)

### Authorization

[ApiKey](../README.md#ApiKey)

### HTTP request headers

 - **Content-Type**: application/json
 - **Accept**: Not defined

### HTTP response details

| Status code | Description | Response headers |
|-------------|-------------|------------------|
**200** | Found. |  -  |
**401** | Authentication Failure |  -  |
**404** | Not Found |  -  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

