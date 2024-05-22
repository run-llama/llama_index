# manager_client.VectorSearchQueriesApi

All URIs are relative to *https://api.wordlift.io*

Method | HTTP request | Description
------------- | ------------- | -------------
[**create_query**](VectorSearchQueriesApi.md#create_query) | **POST** /vector-search/queries | Create


# **create_query**
> PageVectorSearchQueryResponseItem create_query(vector_search_query_request)

Create

### Example

* Api Key Authentication (ApiKey):

```python
import manager_client
from manager_client.models.page_vector_search_query_response_item import PageVectorSearchQueryResponseItem
from manager_client.models.vector_search_query_request import VectorSearchQueryRequest
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
    api_instance = manager_client.VectorSearchQueriesApi(api_client)
    vector_search_query_request = manager_client.VectorSearchQueryRequest() # VectorSearchQueryRequest | 

    try:
        # Create
        api_response = await api_instance.create_query(vector_search_query_request)
        print("The response of VectorSearchQueriesApi->create_query:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling VectorSearchQueriesApi->create_query: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **vector_search_query_request** | [**VectorSearchQueryRequest**](VectorSearchQueryRequest.md)|  | 

### Return type

[**PageVectorSearchQueryResponseItem**](PageVectorSearchQueryResponseItem.md)

### Authorization

[ApiKey](../README.md#ApiKey)

### HTTP request headers

 - **Content-Type**: application/json
 - **Accept**: application/json

### HTTP response details

| Status code | Description | Response headers |
|-------------|-------------|------------------|
**200** | Found. |  -  |
**401** | Authentication Failure |  -  |
**404** | Not Found |  -  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

