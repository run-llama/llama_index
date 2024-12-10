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

def _request_abort_rollout(service_name: str):
    resp = vessl_api.model_service_rollout_abort_api(
        organization_name=vessl_api.organization.name,
        model_service_name=service_name,
    )
    if resp.rollback_requested:
        print("Current update aborted. Rollback is requested.\n")
        return True
    else:
        print("Current update aborted.")
        print("Could not determine the original status. Rollback is not requested.\n")
        print(f"Please check the status of the service and the gateway at: https://app.vessl.ai/{vessl_api.organization.name}/services/{service.name}\n")
        return False
    
def _get_recent_rollout(service_name: str) -> Any:
    resp = vessl_api.model_service_rollout_list_api(
            organization_name=vessl_api.organization.name,
            model_service_name=service_name,
        )
    recent_rollout = resp.rollouts[0] if resp.rollouts else None
    return recent_rollout

def abort_in_progress_rollout_by_name(service_name):
    recent_rollout = _get_recent_rollout(service_name)
    if recent_rollout and recent_rollout.status == "rolling_out":
        print(f"The service {service_name} is currently rolling out.")
        ## Abort the existing rollout if ignore_rollout
        if _request_abort_rollout(service_name):
            print("Waiting for the existing rollout to be aborted...")
            time.sleep(30)
    else:
        print("No existing rollout found.")
