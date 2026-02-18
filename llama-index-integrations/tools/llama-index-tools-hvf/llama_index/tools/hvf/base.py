"""Hudson Valley Forestry (HVF) Tool Spec."""

from typing import Any, Dict, List, Optional

import requests
from llama_index.core.tools.tool_spec.base import BaseToolSpec


HVF_API_BASE_URL = "https://app.hudsonvalleyforestry.com/api"


class HVFToolSpec(BaseToolSpec):
    """Hudson Valley Forestry (HVF) Tool Spec.

    This tool provides an LLM agent with the ability to query and submit data
    to the Hudson Valley Forestry public API. It supports checking service
    health, retrieving available forestry services, and submitting service
    inquiry forms for both residential and commercial clients.

    API Base URL: https://app.hudsonvalleyforestry.com/api

    Example usage::

        from llama_index.tools.hvf import HVFToolSpec
        from llama_index.core.agent.workflow import FunctionAgent
        from llama_index.llms.openai import OpenAI

        hvf_tool = HVFToolSpec()
        agent = FunctionAgent(
            tools=hvf_tool.to_tool_list(),
            llm=OpenAI(model="gpt-4.1"),
        )

        response = await agent.run(
            "What forestry services does Hudson Valley Forestry offer?"
        )
    """

    spec_functions = [
        "health_check",
        "get_services",
        "submit_residential_inquiry",
        "submit_commercial_inquiry",
    ]

    def __init__(
        self,
        base_url: str = HVF_API_BASE_URL,
        timeout: int = 30,
    ) -> None:
        """Initialize the HVFToolSpec.

        Args:
            base_url (str): Base URL for the HVF API.
                Defaults to ``https://app.hudsonvalleyforestry.com/api``.
            timeout (int): Request timeout in seconds. Defaults to ``30``.
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._session = requests.Session()
        self._session.headers.update(
            {
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": "llama-index-tools-hvf/0.1.0",
            }
        )

    # ------------------------------------------------------------------
    # Tool methods
    # ------------------------------------------------------------------

    def health_check(self) -> Dict[str, Any]:
        """Check whether the HVF API is reachable and healthy.

        Use this tool first to verify the API is available before making
        other requests.

        Returns:
            Dict[str, Any]: A dictionary with keys ``status`` (str) and
            ``message`` (str) indicating the health of the API endpoint.
            On success: ``{"status": "ok", "message": "HVF API is healthy"}``.
            On failure: ``{"status": "error", "message": "<reason>"}``.
        """
        try:
            resp = self._session.get(
                f"{self.base_url}/health", timeout=self.timeout
            )
            resp.raise_for_status()
            return {"status": "ok", "message": "HVF API is healthy"}
        except requests.RequestException as exc:
            return {"status": "error", "message": str(exc)}

    def get_services(self, category: Optional[str] = None) -> List[Dict[str, Any]]:
        """Retrieve the list of forestry services offered by Hudson Valley Forestry.

        Returns a list of service objects describing available forestry
        offerings such as timber harvesting, land clearing, lidar mapping,
        invasive species management, and more.

        Args:
            category (str, optional): Filter services by category. Common
                values are ``"residential"`` and ``"commercial"``. If
                ``None``, all services are returned.

        Returns:
            List[Dict[str, Any]]: A list of service dictionaries. Each
            dictionary typically contains:

            - ``name`` (str): Human-readable service name.
            - ``description`` (str): Description of the service.
            - ``category`` (str): Service category (``"residential"`` or
              ``"commercial"``).
            - ``url`` (str): URL of the service page on the HVF website.
        """
        try:
            params: Dict[str, str] = {}
            if category:
                params["category"] = category
            resp = self._session.get(
                f"{self.base_url}/services",
                params=params,
                timeout=self.timeout,
            )
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as exc:
            return [{"error": str(exc)}]

    def submit_residential_inquiry(
        self,
        name: str,
        email: str,
        phone: str,
        address: str,
        service_type: str,
        message: str,
        acreage: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Submit a residential forestry service inquiry to Hudson Valley Forestry.

        Use this tool when a residential homeowner or landowner wants to
        request information or schedule a consultation for forestry services
        on their property.

        Args:
            name (str): Full name of the contact person.
            email (str): Contact email address.
            phone (str): Contact phone number (e.g. ``"845-555-1234"``).
            address (str): Property address or town/county location.
            service_type (str): Type of service requested. Examples:
                ``"Tree Removal"``, ``"Land Clearing"``, ``"Timber Harvest"``,
                ``"Invasive Species Management"``, ``"Lidar Mapping"``.
            message (str): Additional details or questions from the client.
            acreage (float, optional): Approximate acreage of the property
                or area requiring service.

        Returns:
            Dict[str, Any]: A dictionary with keys:

            - ``success`` (bool): Whether the submission was accepted.
            - ``message`` (str): Confirmation or error message.
            - ``inquiry_id`` (str, optional): Unique identifier for the
              submitted inquiry, if provided by the API.
        """
        payload: Dict[str, Any] = {
            "name": name,
            "email": email,
            "phone": phone,
            "address": address,
            "service_type": service_type,
            "message": message,
            "client_type": "residential",
        }
        if acreage is not None:
            payload["acreage"] = acreage

        try:
            resp = self._session.post(
                f"{self.base_url}/inquiry",
                json=payload,
                timeout=self.timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            return {
                "success": True,
                "message": data.get(
                    "message", "Inquiry submitted successfully."
                ),
                "inquiry_id": data.get("inquiry_id"),
            }
        except requests.HTTPError as exc:
            try:
                detail = exc.response.json().get("error", str(exc))
            except Exception:
                detail = str(exc)
            return {"success": False, "message": detail}
        except requests.RequestException as exc:
            return {"success": False, "message": str(exc)}

    def submit_commercial_inquiry(
        self,
        company_name: str,
        contact_name: str,
        email: str,
        phone: str,
        address: str,
        service_type: str,
        message: str,
        acreage: Optional[float] = None,
        project_timeline: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Submit a commercial forestry service inquiry to Hudson Valley Forestry.

        Use this tool when a business, municipality, land trust, or other
        commercial entity wants to request forestry services or schedule a
        consultation.

        Args:
            company_name (str): Name of the company or organization.
            contact_name (str): Full name of the primary contact person.
            email (str): Contact email address.
            phone (str): Contact phone number (e.g. ``"845-555-9876"``).
            address (str): Property or project address/location.
            service_type (str): Type of service requested. Examples:
                ``"Commercial Timber Harvest"``, ``"Land Development Clearing"``,
                ``"Lidar Topographic Survey"``, ``"Invasive Species Control"``,
                ``"Municipal Tree Management"``.
            message (str): Detailed project description or questions.
            acreage (float, optional): Approximate acreage of the project area.
            project_timeline (str, optional): Desired project timeline or
                deadline (e.g. ``"Spring 2026"``).

        Returns:
            Dict[str, Any]: A dictionary with keys:

            - ``success`` (bool): Whether the submission was accepted.
            - ``message`` (str): Confirmation or error message.
            - ``inquiry_id`` (str, optional): Unique identifier for the
              submitted inquiry, if provided by the API.
        """
        payload: Dict[str, Any] = {
            "company_name": company_name,
            "contact_name": contact_name,
            "email": email,
            "phone": phone,
            "address": address,
            "service_type": service_type,
            "message": message,
            "client_type": "commercial",
        }
        if acreage is not None:
            payload["acreage"] = acreage
        if project_timeline is not None:
            payload["project_timeline"] = project_timeline

        try:
            resp = self._session.post(
                f"{self.base_url}/inquiry/commercial",
                json=payload,
                timeout=self.timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            return {
                "success": True,
                "message": data.get(
                    "message", "Commercial inquiry submitted successfully."
                ),
                "inquiry_id": data.get("inquiry_id"),
            }
        except requests.HTTPError as exc:
            try:
                detail = exc.response.json().get("error", str(exc))
            except Exception:
                detail = str(exc)
            return {"success": False, "message": detail}
        except requests.RequestException as exc:
            return {"success": False, "message": str(exc)}
