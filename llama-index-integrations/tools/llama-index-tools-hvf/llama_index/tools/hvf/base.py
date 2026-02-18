"""Hudson Valley Forestry (HVF) Tool Spec."""

from typing import Any, Dict, List, Optional

import requests
from llama_index.core.tools.tool_spec.base import BaseToolSpec


HVF_API_BASE_URL = "https://app.hudsonvalleyforestry.com"


class HVFToolSpec(BaseToolSpec):
    """Hudson Valley Forestry (HVF/HVG/O&G) Tool Spec.

    Provides LLM agents with native access to the Hudson Valley Forestry
    agent-native REST API. Covers three divisions:

    * **HVF Residential** — forestry mulching, selective thinning,
      clearcut & grub, LiDAR mapping. Service area: Hudson Valley NY,
      Berkshires MA, Western CT (~110-mile radius).
    * **HVG Goat Grazing** — targeted goat grazing for invasive species,
      steep slopes, and equipment-inaccessible areas.
    * **Commercial O&G** — pipeline ROW clearing, vegetation management,
      site prep, LiDAR corridor mapping. Service area: Northeast US.

    OpenAPI spec: https://app.hudsonvalleyforestry.com/openapi.json
    Interactive docs: https://app.hudsonvalleyforestry.com/api/docs

    Example usage::

        from llama_index.tools.hvf import HVFToolSpec
        from llama_index.core.agent.workflow import FunctionAgent
        from llama_index.llms.openai import OpenAI

        tools = HVFToolSpec().to_tool_list()
        agent = FunctionAgent(tools=tools, llm=OpenAI(model="gpt-4o"))

        result = await agent.run(
            "Check if a 5-acre property at lat=41.8, lng=-73.9 "
            "is eligible for HVF forestry mulching and get a price estimate."
        )
    """

    spec_functions = [
        "hvf_get_services",
        "hvf_assess_property",
        "hvf_submit_quote",
        "hvg_get_services",
        "hvg_assess_property",
        "hvg_submit_quote",
        "og_get_services",
        "og_assess_project",
        "og_submit_quote",
    ]

    def __init__(
        self,
        base_url: str = HVF_API_BASE_URL,
        timeout: int = 30,
    ) -> None:
        """Initialize HVFToolSpec.

        Args:
            base_url: API base URL. Defaults to
                ``https://app.hudsonvalleyforestry.com``.
            timeout: HTTP request timeout in seconds. Defaults to ``30``.
        """
        self._base = base_url.rstrip("/")
        self._timeout = timeout
        self._session = requests.Session()
        self._session.headers.update(
            {
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": "llama-index-tools-hvf/0.1.0",
            }
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get(self, path: str) -> Any:
        try:
            resp = self._session.get(
                f"{self._base}{path}", timeout=self._timeout
            )
            resp.raise_for_status()
            return resp.json()
        except requests.HTTPError as exc:
            return {"error": f"HTTP {exc.response.status_code}", "detail": exc.response.text}
        except requests.RequestException as exc:
            return {"error": "request_failed", "detail": str(exc)}

    def _post(self, path: str, payload: Dict[str, Any]) -> Any:
        try:
            resp = self._session.post(
                f"{self._base}{path}", json=payload, timeout=self._timeout
            )
            resp.raise_for_status()
            return resp.json()
        except requests.HTTPError as exc:
            return {"error": f"HTTP {exc.response.status_code}", "detail": exc.response.text}
        except requests.RequestException as exc:
            return {"error": "request_failed", "detail": str(exc)}

    # ------------------------------------------------------------------
    # HVF Residential
    # ------------------------------------------------------------------

    def hvf_get_services(self) -> Any:
        """Get Hudson Valley Forestry residential service catalog with pricing.

        Returns all available services (forestry mulching, selective
        thinning, clearcut & grub, LiDAR mapping) with base price per
        acre and minimum acreage requirements.

        Service area: Hudson Valley NY, Berkshires MA, Western CT.

        Returns:
            List of service objects, each containing ``service_type``,
            ``name``, ``description``, ``base_price_per_acre``,
            ``min_acres``.
        """
        return self._get("/api/agent/services")

    def hvf_assess_property(
        self,
        lat: float,
        lng: float,
        acreage: float,
        service_type: str,
        vegetation_density: str = "unknown",
    ) -> Any:
        """Check if a residential property is in HVF's service area and get a price estimate.

        Args:
            lat: Property latitude in decimal degrees.
            lng: Property longitude in decimal degrees.
            acreage: Property size in acres.
            service_type: One of ``forestry_mulching``,
                ``selective_thinning``, ``clearcut_grub``,
                ``lidar_mapping``.
            vegetation_density: ``light``, ``medium``, ``heavy``, or
                ``unknown`` (default). Affects price multiplier.

        Returns:
            Dict with keys ``eligible`` (bool), ``price_estimate_usd``
            (dict with ``low`` and ``high``), ``lead_time_days`` (int),
            and ``message`` (str).
        """
        return self._post("/api/agent/assess", {
            "lat": lat,
            "lng": lng,
            "acreage": acreage,
            "service_type": service_type,
            "vegetation_density": vegetation_density,
        })

    def hvf_submit_quote(
        self,
        email: str,
        name: str,
        acreage: float,
        service_type: str,
        property_description: str,
        phone: str = "",
        address: str = "",
        lat: Optional[float] = None,
        lng: Optional[float] = None,
    ) -> Any:
        """Submit a Hudson Valley Forestry residential quote request.

        Creates an Odoo CRM lead and sends a Mattermost notification.
        Response within 1–2 business days.

        Args:
            email: Customer email address (required).
            name: Customer full name (required).
            acreage: Property size in acres (required).
            service_type: Service type ID from ``hvf_get_services``
                (required).
            property_description: Brief description of the property and
                work needed (required).
            phone: Customer phone number (optional).
            address: Property street address (optional).
            lat: Property latitude — improves routing (optional).
            lng: Property longitude — improves routing (optional).

        Returns:
            Dict with ``quote_id`` (str) and ``message`` (str).
        """
        payload: Dict[str, Any] = {
            "email": email,
            "name": name,
            "acreage": acreage,
            "service_type": service_type,
            "property_description": property_description,
        }
        if phone:
            payload["phone"] = phone
        if address:
            payload["address"] = address
        if lat is not None:
            payload["lat"] = lat
        if lng is not None:
            payload["lng"] = lng
        return self._post("/api/agent/quote", payload)

    # ------------------------------------------------------------------
    # HVG Goat Grazing
    # ------------------------------------------------------------------

    def hvg_get_services(self) -> Any:
        """Get Hudson Valley Goats service catalog for targeted goat grazing.

        Goat grazing is ideal for invasive species removal, steep slopes,
        and areas where mechanical equipment cannot access.

        Service area: Hudson Valley NY, Berkshires MA, Western CT.

        Returns:
            List of goat grazing service objects with pricing.
        """
        return self._get("/api/agent/goat/services")

    def hvg_assess_property(
        self,
        lat: float,
        lng: float,
        acreage: float,
        vegetation_type: str = "unknown",
    ) -> Any:
        """Check if a property is eligible for Hudson Valley Goats grazing service.

        Minimum property size is 0.5 acres.

        Args:
            lat: Property latitude.
            lng: Property longitude.
            acreage: Property size in acres (minimum 0.5).
            vegetation_type: ``invasive_brush``, ``mixed_brush``,
                ``light_grass``, or ``unknown`` (default).

        Returns:
            Dict with ``eligible`` (bool), ``price_estimate_usd``,
            and ``message`` (str).
        """
        return self._post("/api/agent/goat/assess", {
            "lat": lat,
            "lng": lng,
            "acreage": acreage,
            "vegetation_type": vegetation_type,
        })

    def hvg_submit_quote(
        self,
        email: str,
        name: str,
        acreage: float,
        service_type: str,
        property_description: str,
        phone: str = "",
        address: str = "",
        lat: Optional[float] = None,
        lng: Optional[float] = None,
    ) -> Any:
        """Submit a Hudson Valley Goats grazing quote request.

        Creates an Odoo CRM lead and sends a Mattermost notification.
        Response within 1–2 business days.

        Args:
            email: Customer email address (required).
            name: Customer full name (required).
            acreage: Property acreage (required).
            service_type: Use ``goat_grazing`` (required).
            property_description: Description of property and vegetation
                (required).
            phone: Customer phone number (optional).
            address: Property address (optional).
            lat: Property latitude (optional).
            lng: Property longitude (optional).

        Returns:
            Dict with ``quote_id`` (str) and ``message`` (str).
        """
        payload: Dict[str, Any] = {
            "email": email,
            "name": name,
            "acreage": acreage,
            "service_type": service_type,
            "property_description": property_description,
        }
        if phone:
            payload["phone"] = phone
        if address:
            payload["address"] = address
        if lat is not None:
            payload["lat"] = lat
        if lng is not None:
            payload["lng"] = lng
        return self._post("/api/agent/goat/quote", payload)

    # ------------------------------------------------------------------
    # Commercial O&G
    # ------------------------------------------------------------------

    def og_get_services(self) -> Any:
        """Get Hudson Valley Forestry commercial oil & gas service catalog.

        Services: pipeline ROW clearing, vegetation management, site prep
        (compressor/well pads), and LiDAR corridor mapping.

        Service area: Northeast US (NY, NJ, CT, MA, VT, NH, ME, PA, RI, DE, MD).
        All commercial pricing is custom quoted — no per-acre rates returned.

        Returns:
            List of commercial service objects.
        """
        return self._get("/api/agent/commercial/services")

    def og_assess_project(
        self,
        lat: float,
        lng: float,
        service_type: str = "",
        project_description: str = "",
        acreage: Optional[float] = None,
        corridor_miles: Optional[float] = None,
    ) -> Any:
        """Check if a commercial O&G project location is in the service area.

        No price estimate is returned — all commercial work is custom-quoted.
        Service area covers northeast US states.

        Args:
            lat: Project latitude (required).
            lng: Project longitude (required).
            service_type: ``row_clearing``, ``vegetation_management``,
                ``site_prep``, or ``lidar_corridor_mapping`` (optional).
            project_description: Brief project scope description (optional).
            acreage: Site acreage for site_prep / vegetation_management
                (optional).
            corridor_miles: Pipeline corridor length in miles for
                row_clearing / lidar_corridor_mapping (optional).

        Returns:
            Dict with ``eligible`` (bool) and ``message`` (str).
        """
        payload: Dict[str, Any] = {"lat": lat, "lng": lng}
        if service_type:
            payload["service_type"] = service_type
        if project_description:
            payload["project_description"] = project_description
        if acreage is not None:
            payload["acreage"] = acreage
        if corridor_miles is not None:
            payload["corridor_miles"] = corridor_miles
        return self._post("/api/agent/commercial/assess", payload)

    def og_submit_quote(
        self,
        email: str,
        name: str,
        service_type: str,
        project_description: str,
        phone: str = "",
        company: str = "",
        address: str = "",
        lat: Optional[float] = None,
        lng: Optional[float] = None,
        acreage: Optional[float] = None,
        corridor_miles: Optional[float] = None,
        timeline: str = "",
    ) -> Any:
        """Submit a commercial oil & gas quote request to Hudson Valley Forestry.

        Creates an Odoo CRM lead and sends a Mattermost notification.
        Response within 1 business day for commercial inquiries.

        Args:
            email: Contact email address (required).
            name: Contact full name (required).
            service_type: ``row_clearing``, ``vegetation_management``,
                ``site_prep``, or ``lidar_corridor_mapping`` (required).
            project_description: Project scope and requirements (required).
            phone: Contact phone number (optional).
            company: Company or organization name (optional).
            address: Project location address (optional).
            lat: Project latitude (optional).
            lng: Project longitude (optional).
            acreage: Project acreage (optional).
            corridor_miles: Corridor length in miles (optional).
            timeline: Desired start/completion timeline, e.g.
                ``"Q3 2026"`` (optional).

        Returns:
            Dict with ``quote_id`` (str) and ``message`` (str).
        """
        payload: Dict[str, Any] = {
            "email": email,
            "name": name,
            "service_type": service_type,
            "project_description": project_description,
        }
        if phone:
            payload["phone"] = phone
        if company:
            payload["company"] = company
        if address:
            payload["address"] = address
        if lat is not None:
            payload["lat"] = lat
        if lng is not None:
            payload["lng"] = lng
        if acreage is not None:
            payload["acreage"] = acreage
        if corridor_miles is not None:
            payload["corridor_miles"] = corridor_miles
        if timeline:
            payload["timeline"] = timeline
        return self._post("/api/agent/commercial/quote", payload)
