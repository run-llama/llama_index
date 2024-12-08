import time
from typing import Any
from vessl import vessl_api

def wait_for_gateway_enabled(gateway: Any, service_name: str, max_timeout_sec: int = 8*60, print_output: bool = False) -> bool:
    def _check_gateway_enabled(gateway):
        return gateway.enabled and gateway.status == "success" and gateway.endpoint is not None
    if not _check_gateway_enabled(gateway):
        print("Endpoint update in progress. Please wait a moment.")
        gateway_update_timeout = max_timeout_sec
        print(f">> Wait for the endpoint to be ready...")
        while gateway_update_timeout > 0:
            gateway = read_service(service_name=service_name).gateway_config
            if _check_gateway_enabled(gateway):
                break
            gateway_update_timeout -= 5
            time.sleep(5)
            print(f">> Waiting for {max_timeout_sec - gateway_update_timeout} seconds...")
    
        if gateway_update_timeout <= 0:
            print("Endpoint update timeout. Please check the status of the endpoint.")
            return False
    return True

def read_service(service_name: str) -> Any:
    """Get a service from a service name.

    Args:
        service_name(str): The name of the service.

    Example:
        ```python
        vessl.read_service(service_name="my-service")
        ```
    """
    return vessl_api.model_service_read_api(
        organization_name=vessl_api.organization.name, model_service_name=service_name
        )