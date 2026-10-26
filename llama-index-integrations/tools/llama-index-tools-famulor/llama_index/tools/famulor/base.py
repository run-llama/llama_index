"""Famulor tool spec."""

import math
import re
from typing import Any, Dict, List, Optional

from llama_index.core.tools.tool_spec.base import BaseToolSpec

from llama_index.tools.famulor.client import (
    DEFAULT_BASE_URL,
    FamulorClient,
    resolve_api_key,
)

_UUID_RE = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)
_E164_RE = re.compile(r"^\+[1-9]\d{1,14}$")
_VARIABLE_KEY_RE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_FORBIDDEN_VARIABLE_KEYS = frozenset({"constructor", "prototype", "__proto__"})

_PHONE_NUMBER_STATUSES = frozenset({"active", "pending", "released", "suspended"})
_CALL_STATUSES = frozenset(
    {"queued", "ringing", "in_progress", "completed", "failed", "no_answer", "busy"}
)
_CALL_DIRECTIONS = frozenset({"inbound", "outbound", "web"})
_CALL_SORTS = frozenset({"created_at", "updated_at"})
_SENTIMENTS = frozenset({"positive", "neutral", "negative"})
_LEAD_DISPLAY_STATUSES = frozenset(
    {"created", "processing", "rescheduled", "max_retries", "completed", "dnc"}
)
_CAMPAIGN_STATUSES = frozenset({"draft", "running", "paused", "completed", "archived"})
_CAMPAIGN_CHANNELS = frozenset(
    {"phone_call", "whatsapp_template", "whatsapp_call", "sms"}
)


def _require_uuid(name: str, value: str) -> str:
    if not isinstance(value, str) or _UUID_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a UUID.")
    return value


def _optional_uuid(name: str, value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    return _require_uuid(name, value)


def _optional_text(name: str, value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string.")
    return value


def _require_choice(
    name: str, value: Optional[str], allowed: frozenset
) -> Optional[str]:
    if value is None:
        return None
    if value not in allowed:
        choices = ", ".join(sorted(allowed))
        raise ValueError(f"{name} must be one of: {choices}.")
    return value


def _check_page(limit: Optional[int], offset: Optional[int]) -> None:
    if limit is not None and (
        isinstance(limit, bool)
        or not isinstance(limit, int)
        or limit < 1
        or limit > 200
    ):
        raise ValueError("limit must be an integer from 1 to 200.")
    if offset is not None and (
        isinstance(offset, bool) or not isinstance(offset, int) or offset < 0
    ):
        raise ValueError("offset must be an integer greater than or equal to 0.")


def _compact_params(values: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    params: Dict[str, Any] = {}
    for key, value in values.items():
        if value is None:
            continue
        if isinstance(value, bool):
            params[key] = "true" if value else "false"
        else:
            params[key] = value
    if not params:
        return None
    return params


def _optional_object(
    name: str, value: Optional[Dict[str, Any]]
) -> Optional[Dict[str, Any]]:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object.")
    return value


def _optional_tags(tags: Optional[List[str]]) -> Optional[List[str]]:
    if tags is None:
        return None
    if not isinstance(tags, list) or len(tags) > 32:
        raise ValueError("tags must be a list of at most 32 strings.")
    for tag in tags:
        if not isinstance(tag, str) or not tag or len(tag) > 40:
            raise ValueError("Each tag must be a string of 1 to 40 characters.")
    return tags


def _optional_variables(
    variables: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    if variables is None:
        return None
    if not isinstance(variables, dict):
        raise ValueError("variables must be a JSON object.")
    for key, value in variables.items():
        if (
            not isinstance(key, str)
            or key in _FORBIDDEN_VARIABLE_KEYS
            or _VARIABLE_KEY_RE.fullmatch(key) is None
        ):
            raise ValueError(
                "Variable names must match ^[a-z][a-z0-9_]{0,63}$ and must not be "
                "constructor, prototype, or __proto__."
            )
        if isinstance(value, bool):
            continue
        if isinstance(value, str):
            if len(value) > 2000:
                raise ValueError("Variable strings are limited to 2000 characters.")
            continue
        if isinstance(value, int):
            continue
        if isinstance(value, float):
            if not math.isfinite(value):
                raise ValueError("Variable numbers must be finite.")
            continue
        raise ValueError("Variable values must be a string, number, or boolean.")
    return variables


def _optional_country(country: Optional[str]) -> Optional[str]:
    if country is None:
        return None
    if not isinstance(country, str) or not re.fullmatch(r"[A-Za-z]{2}", country):
        raise ValueError("country must be an ISO-2 country code, for example DE.")
    return country.upper()


def _campaign_filter(campaign_id: Optional[str]) -> Optional[str]:
    if campaign_id is None:
        return None
    if campaign_id == "none":
        return campaign_id
    return _require_uuid("campaign_id", campaign_id)


class FamulorToolSpec(BaseToolSpec):
    """
    Famulor voice-agent tool spec.

    Famulor (https://www.famulor.io) runs AI phone agents. This spec calls the
    public Platform v1 REST API at ``https://app.famulor.io/api/v1``. Authenticate
    with a workspace API key (``fam_...``) or an OAuth access token (``fam_at_...``)
    in the ``Authorization: Bearer`` header. The tenant is encoded in the key.
    No workspace header is sent.

    ``make_call`` places a real outbound phone call. The other write methods
    create an Audience contact or a campaign draft. They do not dial anyone.
    Campaign start and stop are not exposed.
    """

    spec_functions = [
        "list_assistants",
        "get_assistant",
        "list_phone_numbers",
        "list_calls",
        "get_call",
        "list_leads",
        "create_lead",
        "get_lead",
        "list_campaigns",
        "get_campaign",
        "create_campaign",
        "make_call",
    ]

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 30.0,
    ) -> None:
        """
        Initialize the Famulor client.

        Args:
            api_key (str): Bearer token (``fam_...``). Defaults to the
                ``FAMULOR_API_KEY`` environment variable.
            base_url (str): Platform v1 origin. Override only for a white-label
                host, for example ``https://voice.example.com/api/v1``.
            timeout (float): HTTP timeout in seconds.

        """
        self.api_key = resolve_api_key(api_key)
        self.base_url = base_url
        self._client = FamulorClient(
            api_key=self.api_key,
            base_url=base_url,
            timeout=timeout,
        )
        self.base_url = self._client.base_url

    def list_assistants(
        self,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        List assistants in the workspace, newest first.

        Calls ``GET /assistants``. Requires scope ``assistants:read``.

        Args:
            limit (int): Page size from 1 to 200. The API default is 50.
            offset (int): Number of items to skip. The API default is 0.

        Returns:
            The API envelope. ``data`` is a list of assistants and
            ``meta.pagination.total`` is the total match count.

        """
        _check_page(limit, offset)
        return self._client.request(
            "GET",
            "/assistants",
            params=_compact_params({"limit": limit, "offset": offset}),
        )

    def get_assistant(self, assistant_id: str) -> Dict[str, Any]:
        """
        Get one assistant by id.

        Calls ``GET /assistants/{id}``. Requires scope ``assistants:read``.

        Args:
            assistant_id (str): Assistant UUID.

        Returns:
            The API envelope. ``data`` is the assistant object.

        """
        assistant_id = _require_uuid("assistant_id", assistant_id)
        return self._client.request("GET", f"/assistants/{assistant_id}")

    def list_phone_numbers(
        self,
        assistant_id: Optional[str] = None,
        status: Optional[str] = None,
        country: Optional[str] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        List phone numbers in the workspace, newest first.

        Calls ``GET /phone-numbers``. Requires scope ``phone_numbers:read``.

        Args:
            assistant_id (str): Only numbers assigned to this assistant UUID.
            status (str): One of ``active``, ``pending``, ``released``,
                ``suspended``.
            country (str): ISO-2 country code, for example ``DE``.
            limit (int): Page size from 1 to 200. The API default is 50.
            offset (int): Number of items to skip. The API default is 0.

        Returns:
            The API envelope. ``data`` is a list of phone numbers.

        """
        _check_page(limit, offset)
        params = _compact_params(
            {
                "assistant_id": _optional_uuid("assistant_id", assistant_id),
                "status": _require_choice("status", status, _PHONE_NUMBER_STATUSES),
                "country": _optional_country(country),
                "limit": limit,
                "offset": offset,
            }
        )
        return self._client.request("GET", "/phone-numbers", params=params)

    def list_calls(
        self,
        assistant_id: Optional[str] = None,
        campaign_id: Optional[str] = None,
        status: Optional[str] = None,
        direction: Optional[str] = None,
        sort: Optional[str] = None,
        created_from: Optional[str] = None,
        created_to: Optional[str] = None,
        sentiment: Optional[str] = None,
        success: Optional[bool] = None,
        q: Optional[str] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        List calls, newest first.

        Calls ``GET /calls``. Requires scope ``calls:read``. ``created_from`` and
        ``created_to`` are sent as the documented ``from`` and ``to`` query
        parameters. ``q`` searches the transcript and summary.

        Args:
            assistant_id (str): Filter by assistant UUID.
            campaign_id (str): Filter by campaign UUID.
            status (str): One of ``queued``, ``ringing``, ``in_progress``,
                ``completed``, ``failed``, ``no_answer``, ``busy``.
            direction (str): One of ``inbound``, ``outbound``, ``web``.
            sort (str): ``created_at`` (default) or ``updated_at``.
            created_from (str): ISO 8601 lower bound, sent as ``from``.
            created_to (str): ISO 8601 upper bound, sent as ``to``.
            sentiment (str): ``positive``, ``neutral``, or ``negative``.
            success (bool): Post-call success verdict.
            q (str): Case-insensitive search of transcript and summary.
            limit (int): Page size from 1 to 200. The API default is 50.
            offset (int): Number of items to skip. The API default is 0.

        Returns:
            The API envelope. ``data`` is a list of calls. Transcript text may
            be absent until the call finishes. Use ``get_call`` for the full
            transcript, summary, and recording link.

        """
        _check_page(limit, offset)
        if success is not None and not isinstance(success, bool):
            raise ValueError("success must be a boolean.")
        params = _compact_params(
            {
                "assistant_id": _optional_uuid("assistant_id", assistant_id),
                "campaign_id": _optional_uuid("campaign_id", campaign_id),
                "status": _require_choice("status", status, _CALL_STATUSES),
                "direction": _require_choice("direction", direction, _CALL_DIRECTIONS),
                "sort": _require_choice("sort", sort, _CALL_SORTS),
                "from": _optional_text("created_from", created_from),
                "to": _optional_text("created_to", created_to),
                "sentiment": _require_choice("sentiment", sentiment, _SENTIMENTS),
                "success": success,
                "q": _optional_text("q", q),
                "limit": limit,
                "offset": offset,
            }
        )
        return self._client.request("GET", "/calls", params=params)

    def get_call(self, call_id: str) -> Dict[str, Any]:
        """
        Get one call, including transcript and summary.

        Calls ``GET /calls/{id}``. Requires scope ``calls:read``. The detail
        adds ``recording_url`` (signed, valid for 1 hour), ``structured_transcript``,
        ``events``, and call insights when the API includes them. ``summary`` is
        the AI post-call summary and is null until analysis finishes.

        Args:
            call_id (str): Call UUID.

        Returns:
            The API envelope. ``data`` is the call detail.

        """
        call_id = _require_uuid("call_id", call_id)
        return self._client.request("GET", f"/calls/{call_id}")

    def list_leads(
        self,
        search: Optional[str] = None,
        campaign_id: Optional[str] = None,
        display_status: Optional[str] = None,
        dnc: Optional[bool] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        List Audience contacts.

        Calls ``GET /leads``. Requires scope ``leads:read``. There is no
        ``GET /leads/{id}``. To read one campaign lead, use ``get_lead``.

        Args:
            search (str): Search contact name, phone, or email.
            campaign_id (str): Campaign UUID, or ``none`` for contacts that
                are not in a campaign.
            display_status (str): One of ``created``, ``processing``,
                ``rescheduled``, ``max_retries``, ``completed``, ``dnc``.
            dnc (bool): True returns only Do-Not-Call contacts. False excludes them.
            limit (int): Page size from 1 to 200. The API default is 50.
            offset (int): Number of items to skip. The API default is 0.

        Returns:
            The API envelope. ``data`` is a list of contacts.

        """
        _check_page(limit, offset)
        if dnc is not None and not isinstance(dnc, bool):
            raise ValueError("dnc must be a boolean.")
        params = _compact_params(
            {
                "search": _optional_text("search", search),
                "campaign_id": _campaign_filter(campaign_id),
                "display_status": _require_choice(
                    "display_status", display_status, _LEAD_DISPLAY_STATUSES
                ),
                "dnc": dnc,
                "limit": limit,
                "offset": offset,
            }
        )
        return self._client.request("GET", "/leads", params=params)

    def create_lead(
        self,
        phone: Optional[str] = None,
        email: Optional[str] = None,
        name: Optional[str] = None,
        tags: Optional[List[str]] = None,
        custom_fields: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Create an Audience contact, or enrich an existing one.

        Calls ``POST /leads``. Requires scope ``leads:write``. This writes a
        contact. It does not place a phone call. A phone number is stored in
        E.164. If the phone or email already exists, the API returns HTTP 200
        and ``data.created`` is false. A new contact returns HTTP 201 and
        ``data.created`` is true.

        Args:
            phone (str): Phone number. Provide ``phone`` or ``email``, or both.
            email (str): Email address. Provide ``phone`` or ``email``, or both.
            name (str): Contact name.
            tags (list): Up to 32 tags, each 1 to 40 characters.
            custom_fields (dict): User-managed contact fields.

        Returns:
            The API envelope. ``data`` contains ``id`` and ``created``.

        """
        phone = _optional_text("phone", phone)
        email = _optional_text("email", email)
        if phone is None and email is None:
            raise ValueError("create_lead requires a phone number or an email address.")
        body: Dict[str, Any] = {}
        if phone is not None:
            body["phone"] = phone
        if email is not None:
            body["email"] = email
        contact_name = _optional_text("name", name)
        if contact_name is not None:
            body["name"] = contact_name
        contact_tags = _optional_tags(tags)
        if contact_tags is not None:
            body["tags"] = contact_tags
        fields = _optional_object("custom_fields", custom_fields)
        if fields is not None:
            body["custom_fields"] = fields
        return self._client.request("POST", "/leads", json_body=body)

    def get_lead(self, campaign_id: str, lead_id: str) -> Dict[str, Any]:
        """
        Get one lead inside a campaign.

        Calls ``GET /campaigns/{id}/leads/{leadId}``. Requires scope
        ``leads:read``. The response includes editable contact data,
        user-managed ``custom_fields``, and read-only ``system_variables``.

        Args:
            campaign_id (str): Campaign UUID.
            lead_id (str): Lead UUID.

        Returns:
            The API envelope. ``data`` is the campaign lead.

        """
        campaign_id = _require_uuid("campaign_id", campaign_id)
        lead_id = _require_uuid("lead_id", lead_id)
        return self._client.request("GET", f"/campaigns/{campaign_id}/leads/{lead_id}")

    def list_campaigns(
        self,
        status: Optional[str] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        List campaigns.

        Calls ``GET /campaigns``. Requires scope ``campaigns:read``.

        Args:
            status (str): One of ``draft``, ``running``, ``paused``,
                ``completed``, ``archived``.
            limit (int): Page size from 1 to 200. The API default is 50.
            offset (int): Number of items to skip. The API default is 0.

        Returns:
            The API envelope. ``data`` is a list of campaigns.

        """
        _check_page(limit, offset)
        params = _compact_params(
            {
                "status": _require_choice("status", status, _CAMPAIGN_STATUSES),
                "limit": limit,
                "offset": offset,
            }
        )
        return self._client.request("GET", "/campaigns", params=params)

    def get_campaign(self, campaign_id: str) -> Dict[str, Any]:
        """
        Get one campaign, including ``total_leads``.

        Calls ``GET /campaigns/{id}``. Requires scope ``campaigns:read``.

        Args:
            campaign_id (str): Campaign UUID.

        Returns:
            The API envelope. ``data`` is the campaign.

        """
        campaign_id = _require_uuid("campaign_id", campaign_id)
        return self._client.request("GET", f"/campaigns/{campaign_id}")

    def create_campaign(
        self,
        name: str,
        assistant_id: Optional[str] = None,
        channel: Optional[str] = None,
        timezone: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Create a campaign draft.

        Calls ``POST /campaigns``. Requires scope ``campaigns:write``. The API
        creates the campaign in status ``draft``. This method does not start
        the campaign and does not place calls. Only ``name``, and optionally
        ``assistant_id``, ``channel``, and ``timezone``, are sent. Other dialer
        settings keep the API defaults.

        Args:
            name (str): Campaign name.
            assistant_id (str): Assistant UUID that would run the calls.
            channel (str): ``phone_call``, ``whatsapp_template``,
                ``whatsapp_call``, or ``sms``.
            timezone (str): IANA timezone, for example ``Europe/Berlin``.

        Returns:
            The API envelope. ``data`` is the created draft campaign.

        """
        campaign_name = _optional_text("name", name)
        if campaign_name is None:
            raise ValueError("name must be a non-empty string.")
        body: Dict[str, Any] = {"name": campaign_name}
        assistant = _optional_uuid("assistant_id", assistant_id)
        if assistant is not None:
            body["assistant_id"] = assistant
        selected_channel = _require_choice("channel", channel, _CAMPAIGN_CHANNELS)
        if selected_channel is not None:
            body["channel"] = selected_channel
        selected_timezone = _optional_text("timezone", timezone)
        if selected_timezone is not None:
            body["timezone"] = selected_timezone
        return self._client.request("POST", "/campaigns", json_body=body)

    def make_call(
        self,
        assistant_id: str,
        to_number: str,
        phone_number_id: Optional[str] = None,
        lead: Optional[Dict[str, Any]] = None,
        variables: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Place a real outbound phone call.

        This calls ``POST /calls`` and dials ``to_number`` with the selected
        assistant. It is a live telephony action, not a draft, preview, or
        simulation. The assistant must be active and needs an outbound-capable
        number, either assigned to the assistant or passed as
        ``phone_number_id``. The API returns the call immediately with status
        ``queued``. Poll ``get_call`` for the transcript, summary, and any
        provider-neutral ``failure`` guidance. A forbidden destination returns
        HTTP 403 ``destination_forbidden``. Requires scope ``calls:write``.

        Args:
            assistant_id (str): UUID of the active assistant that runs the call.
            to_number (str): Destination in E.164 format, for example
                ``+4930123456``.
            phone_number_id (str): Optional caller-id phone number UUID.
                Defaults to the number assigned to the assistant.
            lead (dict): Optional contact context. Custom fields follow the
                assistant contact-field access settings.
            variables (dict): Optional per-call values. Keys match
                ``^[a-z][a-z0-9_]{0,63}$``. Values are strings (max 2000
                characters), numbers, or booleans.

        Returns:
            The API envelope. ``data`` is the created call with status
            ``queued``.

        """
        assistant_id = _require_uuid("assistant_id", assistant_id)
        if not isinstance(to_number, str) or _E164_RE.fullmatch(to_number) is None:
            raise ValueError(
                "to_number must be an E.164 phone number, for example +4930123456."
            )
        body: Dict[str, Any] = {
            "assistant_id": assistant_id,
            "to_number": to_number,
        }
        caller_id = _optional_uuid("phone_number_id", phone_number_id)
        if caller_id is not None:
            body["phone_number_id"] = caller_id
        contact = _optional_object("lead", lead)
        if contact is not None:
            body["lead"] = contact
        call_variables = _optional_variables(variables)
        if call_variables is not None:
            body["variables"] = call_variables
        return self._client.request("POST", "/calls", json_body=body)
