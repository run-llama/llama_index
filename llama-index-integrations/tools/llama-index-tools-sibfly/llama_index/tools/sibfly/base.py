"""SibFly ground-motion tool spec."""

import os
from typing import Optional

import requests

from llama_index.core.schema import Document
from llama_index.core.tools.tool_spec.base import BaseToolSpec

DEFAULT_BASE_URL = "https://sibfly.com"

# Fields kept from the raw SibFly response (drop the rest / nulls).
_KEEP = (
    "status",
    "velocity_vertical_mm_yr",
    "velocity_vertical_in_yr",
    "velocity_uncertainty_mm_yr",
    "seasonal_amplitude_mm",
    "total_motion_mm",
    "trend",
    "assessment",
    "assessment_code",
    "confidence",
    "near_threshold",
    "neighbor_consistent",
    "data_age_days",
    "data_freshness",
    "frame",
    "pixel_id",
    "n_measurements",
    "last_observation",
    "data_coverage",
    "covered",
    "cost_usd",
    "would_cost_usd",
    "credits_remaining_usd",
    "message",
    "request_id",
    "error",
    "code",
    "top_up_url",
)


class SibflyToolSpec(BaseToolSpec):
    """
    SibFly ground-motion tool spec.

    Measured ground motion for any US address. Wraps the SibFly API
    (https://sibfly.com) so an agent can find out how fast the ground is
    sinking or rising (mm/year, negative = sinking) under a given address,
    measured from NASA OPERA Sentinel-1 satellite radar (InSAR) — measured, not
    modeled. Pricing is agent-friendly: $0.40 per covered report, and misses
    (out-of-coverage, no-data, too-stale, low-confidence) are free. Use
    ``check_coverage`` (free) to preflight a point before spending.
    """

    spec_functions = ["check_ground_motion", "check_coverage"]

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 60.0,
    ) -> None:
        """
        Initialize with parameters.

        Args:
            api_key: SibFly API key (``sf_live_...``). Falls back to the
                ``SIBFLY_API_KEY`` environment variable. An agent can obtain a
                key with no human in the loop via SibFly's autonomous
                registration endpoint (``POST /api/v1/autonomous/register``).
            base_url: SibFly API base URL. Defaults to ``https://sibfly.com``.
            timeout: Per-request timeout in seconds.

        """
        api_key = api_key or os.environ.get("SIBFLY_API_KEY")
        if not api_key:
            raise ValueError(
                "A SibFly API key is required. Pass api_key=... or set the "
                "SIBFLY_API_KEY environment variable. Get a free key at "
                "https://sibfly.com."
            )
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _get(self, path: str, params: dict) -> dict:
        response = requests.get(
            f"{self.base_url}{path}",
            params=params,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Accept": "application/json",
            },
            timeout=self.timeout,
        )
        try:
            data = response.json()
        except ValueError:
            data = None
        # SibFly errors are structured JSON with a stable ``error`` code (e.g.
        # 402 insufficient_credits carries a top_up_url) — surface those as data
        # rather than raising, so the agent can act on them.
        if response.status_code >= 400 and not (
            isinstance(data, dict) and (data.get("error") or data.get("code"))
        ):
            response.raise_for_status()
        return data if isinstance(data, dict) else {"raw": data}

    @staticmethod
    def _params(address, lat, lon) -> dict:
        if address:
            return {"address": address}
        if lat is None or lon is None:
            raise ValueError("Provide an 'address' or both 'lat' and 'lon'.")
        return {"lat": lat, "lon": lon}

    @staticmethod
    def _shape(data: dict) -> dict:
        out = {k: data[k] for k in _KEEP if data.get(k) is not None}
        q = data.get("query")
        if isinstance(q, dict):
            out["query"] = {
                k: q[k]
                for k in ("address", "geocoded_address", "lat", "lon")
                if q.get(k) is not None
            }
        return out

    def check_ground_motion(
        self,
        address: Optional[str] = None,
        lat: Optional[float] = None,
        lon: Optional[float] = None,
        dry_run: bool = False,
    ) -> Document:
        """
        Measure ground motion (subsidence/uplift) for a US address or point.

        Returns how fast the ground is sinking or rising in mm/year (negative =
        sinking), measured from NASA satellite radar. Costs $0.40 for a covered
        report; out-of-coverage / low-quality results are free. Pass
        ``dry_run=True`` for a free coverage + price preview with no charge.

        Args:
            address: A US street address, e.g. ``1100 Congress Ave, Austin, TX``.
                Provide this OR lat+lon.
            lat: Latitude (use with lon instead of address).
            lon: Longitude (use with lat instead of address).
            dry_run: If True, return a free preview (coverage, confidence,
                would_cost_usd) without buying the report.

        Returns:
            A Document whose ``text`` is a short verdict and whose ``metadata``
            holds the structured result (velocity_vertical_mm_yr,
            assessment_code, confidence, data_age_days, cost_usd, ...). Route
            logic on ``assessment_code``, not the human ``assessment`` string.

        """
        params = self._params(address, lat, lon)
        if dry_run:
            params["dry_run"] = 1
        data = self._get("/api/v1/motion", params)
        shaped = self._shape(data)

        if shaped.get("error"):
            summary = f"Error: {shaped.get('error')} - {shaped.get('message', '')}"
        elif shaped.get("velocity_vertical_mm_yr") is not None:
            summary = (
                f"{address or 'point'}: {shaped['velocity_vertical_mm_yr']} mm/yr "
                f"({shaped.get('assessment_code')}), confidence "
                f"{shaped.get('confidence')}, data ~{shaped.get('data_age_days')} "
                f"days old. Cost ${shaped.get('cost_usd')}."
            )
        else:
            summary = (
                f"{address or 'point'}: {shaped.get('status', 'no data')} "
                f"(free, $0). {shaped.get('message', '')}"
            ).strip()
        return Document(text=summary, metadata=shaped)

    def check_coverage(
        self,
        address: Optional[str] = None,
        lat: Optional[float] = None,
        lon: Optional[float] = None,
    ) -> Document:
        """
        FREE preflight: is a US address/point covered, how stale is the data,
        and what would a report cost? Always $0. Call this before spending.

        Args:
            address: A US street address. Provide this OR lat+lon.
            lat: Latitude (use with lon).
            lon: Longitude (use with lat).

        Returns:
            A Document whose ``metadata`` holds ``covered``, ``data_age_days``,
            and ``would_cost_usd``.

        """
        data = self._get("/api/v1/coverage", self._params(address, lat, lon))
        shaped = self._shape(data)
        if shaped.get("covered"):
            summary = (
                f"Covered: would cost ${shaped.get('would_cost_usd')}, data "
                f"~{shaped.get('data_age_days')} days old."
            )
        else:
            summary = "Not covered at this point (a report would be free)."
        return Document(text=summary, metadata=shaped)
