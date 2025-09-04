"""Kaltura eSearch API Reader."""

import json
import logging
from typing import Any, Dict, List, Optional

import requests
from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document

logger = logging.getLogger(__name__)


class KalturaESearchReader(BaseReader):
    """Kaltura eSearch API Reader."""

    def __init__(
        self,
        partner_id: int = 0,
        api_secret: str = "INSERT_YOUR_ADMIN_SECRET",
        user_id: str = "INSERT_YOUR_USER_ID",
        ks_type: int = 2,
        ks_expiry: int = 86400,
        ks_privileges: str = "disableentitlement",
        kaltura_api_endpoint: str = "https://cdnapi-ev.kaltura.com/",
        request_timeout: int = 500,
        should_log_api_calls: bool = False,
    ) -> None:
        """
        Initialize a new instance of KalturaESearchReader.

        Args:
            partner_id (int): The Kaltura Account ID. Default is 0.
            api_secret (str): The Kaltura API Admin Secret. Default is "INSERT_YOUR_ADMIN_SECRET".
            user_id (str): User ID for executing and logging all API actions under. Default is "INSERT_YOUR_USER_ID".
            ks_type (int): Type of Kaltura Session. Default is 2.
            ks_expiry (int): Validity of the Kaltura session in seconds. Default is 86400.
            ks_privileges (str): Kaltura session privileges. Default is "disableentitlement".
            kaltura_api_endpoint (str): The Kaltura API endpoint. Default is "https://cdnapi-ev.kaltura.com/".
            request_timeout (int): API request timeout in seconds. Default is 500.
            should_log_api_calls (bool): Boolean value determining whether to log Kaltura requests. Default is False.

        """
        self.partner_id = partner_id
        self.api_secret = api_secret
        self.user_id = user_id
        self.ks_type = ks_type
        self.ks_expiry = ks_expiry
        self.ks_privileges = ks_privileges
        self.kaltura_api_endpoint = kaltura_api_endpoint
        self.request_timeout = request_timeout
        self.should_log_api_calls = should_log_api_calls
        # Kaltura libraries will be loaded when they are needed
        self._kaltura_loaded = False

    def _load_kaltura(self):
        """Load Kaltura libraries and initialize the Kaltura client."""
        from KalturaClient import KalturaClient
        from KalturaClient.Base import IKalturaLogger, KalturaConfiguration
        from KalturaClient.Plugins.Core import KalturaSessionType

        class KalturaLogger(IKalturaLogger):
            def log(self, msg):
                logging.info(msg)

        try:
            self.config = KalturaConfiguration()
            self.config.requestTimeout = self.request_timeout
            self.config.serviceUrl = self.kaltura_api_endpoint
            if self.should_log_api_calls:
                self.config.setLogger(KalturaLogger())
            self.client = KalturaClient(self.config)
            if self.ks_type is None:
                self.ks_type = KalturaSessionType.ADMIN
            self.ks = self.client.generateSessionV2(
                self.api_secret,
                self.user_id,
                self.ks_type,
                self.partner_id,
                self.ks_expiry,
                self.ks_privileges,
            )
            self.client.setKs(self.ks)
            self._kaltura_loaded = True
        except Exception:
            logger.error("Kaltura Auth failed, check your credentials")

    def _load_from_search_params(
        self, search_params, with_captions: bool = True, max_entries: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Load search parameters and returns a list of entries.

        Args:
            search_params: Search parameters for Kaltura eSearch.
            with_captions (bool): If True, the entries will include captions.
            max_entries (int): Maximum number of entries to return.

        Returns:
            list: A list of entries as dictionaries,
            if captions required entry_info will include all metadata and text will include transcript,
            otherwise info is just entry_id and text is all metadata.

        """
        from KalturaClient.Plugins.Core import KalturaPager

        try:
            entries = []
            pager = KalturaPager()
            pager.pageIndex = 1
            pager.pageSize = max_entries
            response = self.client.elasticSearch.eSearch.searchEntry(
                search_params, pager
            )

            for search_result in response.objects:
                entry = search_result.object
                items_data = search_result.itemsData

                entry_info = {
                    "entry_id": str(entry.id),
                    "entry_name": str(entry.name),
                    "entry_description": str(entry.description or ""),
                    "entry_media_type": int(entry.mediaType.value or 0),
                    "entry_media_date": int(entry.createdAt or 0),
                    "entry_ms_duration": int(entry.msDuration or 0),
                    "entry_last_played_at": int(entry.lastPlayedAt or 0),
                    "entry_application": str(entry.application or ""),
                    "entry_tags": str(entry.tags or ""),
                    "entry_reference_id": str(entry.referenceId or ""),
                }

                if with_captions:
                    caption_search_result = items_data[0].items[0]
                    if hasattr(caption_search_result, "captionAssetId"):
                        # TODO: change this to fetch captions per language, or as for a specific language code
                        caption_asset_id = caption_search_result.captionAssetId
                        entry_dict = {
                            "video_transcript": self._get_json_transcript(
                                caption_asset_id
                            )
                        }
                    else:
                        entry_dict = entry_info.copy()
                        entry_info = {"entry_id": str(entry.id)}
                else:
                    entry_dict = entry_info.copy()
                    entry_info = {"entry_id": str(entry.id)}

                entry_doc = Document(text=json.dumps(entry_dict), extra_info=entry_info)
                entries.append(entry_doc)

            return entries

        except Exception as e:
            if e.code == "INVALID_KS":
                raise ValueError(f"Kaltura Auth failed, check your credentials: {e}")
            logger.error(f"An error occurred while loading with search params: {e}")
            return []

    def _get_json_transcript(self, caption_asset_id):
        """
        Fetch json transcript/captions from a given caption_asset_id.

        Args:
            caption_asset_id: The ID of the caption asset that includes the captions to fetch json transcript for

        Returns:
            A JSON transcript of the captions, or an empty dictionary if none found or an error occurred.

        """
        # TODO: change this to fetch captions per language, or as for a specific language code
        try:
            cap_json_url = self.client.caption.captionAsset.serveAsJson(
                caption_asset_id
            )
            return requests.get(cap_json_url).json()
        except Exception as e:
            logger.error(f"An error occurred while getting captions: {e}")
            return {}

    def load_data(
        self,
        search_params: Any = None,
        search_operator_and: bool = True,
        free_text: Optional[str] = None,
        category_ids: Optional[str] = None,
        with_captions: bool = True,
        max_entries: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Load data from the Kaltura based on search parameters.
        The function returns a list of dictionaries.
        Each dictionary represents a media entry, where the keys are strings (field names) and the values can be of any type.

        Args:
            search_params: search parameters of type KalturaESearchEntryParams with pre-set search queries. If not provided, the other parameters will be used to construct the search query.
            search_operator_and: if True, the constructed search query will have AND operator between query filters, if False, the operator will be OR.
            free_text: if provided, will be used as the free text query of the search in Kaltura.
            category_ids: if provided, will only search for entries that are found inside these category ids.
            withCaptions: determines whether or not to also download captions/transcript contents from Kaltura.
            maxEntries: sets the maximum number of entries to pull from Kaltura, between 0 to 500 (max pageSize in Kaltura).

        Returns:
            List[Dict[str, Any]]: A list of dictionaries representing Kaltura Media Entries with the following fields:
            entry_id:str, entry_name:str, entry_description:str, entry_captions:JSON,
            entry_media_type:int, entry_media_date:int, entry_ms_duration:int, entry_last_played_at:int,
            entry_application:str, entry_tags:str, entry_reference_id:str.
            If with_captions is False, it sets entry_info to only include the entry_id and entry_dict to include all other entry information.
            If with_captions is True, it sets entry_info to include all entry information and entry_dict to only include the entry transcript fetched via self._get_captions(items_data).

        """
        from KalturaClient.Plugins.ElasticSearch import (
            KalturaCategoryEntryStatus,
            KalturaESearchCaptionFieldName,
            KalturaESearchCaptionItem,
            KalturaESearchCategoryEntryFieldName,
            KalturaESearchCategoryEntryItem,
            KalturaESearchEntryOperator,
            KalturaESearchEntryParams,
            KalturaESearchItemType,
            KalturaESearchOperatorType,
            KalturaESearchUnifiedItem,
        )

        # Load and initialize the Kaltura client
        if not self._kaltura_loaded:
            self._load_kaltura()

        # Validate input parameters:
        if search_params is None:
            search_params = KalturaESearchEntryParams()
            # Create an AND/OR relationship between the following search queries -
            search_params.searchOperator = KalturaESearchEntryOperator()
            if search_operator_and:
                search_params.searchOperator.operator = (
                    KalturaESearchOperatorType.AND_OP
                )
            else:
                search_params.searchOperator.operator = KalturaESearchOperatorType.OR_OP
            search_params.searchOperator.searchItems = []
            # Find only entries that have captions -
            if with_captions:
                caption_item = KalturaESearchCaptionItem()
                caption_item.fieldName = KalturaESearchCaptionFieldName.CONTENT
                caption_item.itemType = KalturaESearchItemType.EXISTS
                search_params.searchOperator.searchItems.append(caption_item)
            # Find only entries that are inside these category IDs -
            if category_ids is not None:
                category_item = KalturaESearchCategoryEntryItem()
                category_item.categoryEntryStatus = KalturaCategoryEntryStatus.ACTIVE
                category_item.fieldName = KalturaESearchCategoryEntryFieldName.FULL_IDS
                category_item.addHighlight = False
                category_item.itemType = KalturaESearchItemType.EXACT_MATCH
                category_item.searchTerm = category_ids
                search_params.searchOperator.searchItems.append(category_item)
            # Find only entries that has this freeText found in them -
            if free_text is not None:
                unified_item = KalturaESearchUnifiedItem()
                unified_item.searchTerm = free_text
                unified_item.itemType = KalturaESearchItemType.PARTIAL
                search_params.searchOperator.searchItems.append(unified_item)

        return self._load_from_search_params(search_params, with_captions, max_entries)
