"""Google Calendar tool spec."""

# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import datetime
import json
import os
from typing import Any, List, Optional, Union

from llama_index.core.schema import Document
from llama_index.core.tools.tool_spec.base import BaseToolSpec

SCOPES = ["https://www.googleapis.com/auth/calendar"]


class GoogleCalendarToolSpec(BaseToolSpec):
    """
    Google Calendar tool spec.

    Currently a simple wrapper around the data loader.
    TODO: add more methods to the Google Calendar spec.

    """

    spec_functions = ["load_data", "create_event", "get_date"]

    def __init__(
        self,
        creds: Optional[Any] = None,
        credentials_path: str = "credentials.json",
        token_path: str = "token.json",
        service_account_key_path: str = "service_account_key.json",
        service_account_key: Optional[dict] = None,
        authorized_user_info: Optional[dict] = None,
        is_cloud: bool = False,
    ):
        """
        Initialize the GoogleCalendarToolSpec.

        Args:
            creds (Optional[Any]): Pre-configured credentials to use for authentication.
                                 If provided, these will be used instead of the OAuth flow.
            credentials_path (str): Path to the OAuth client secrets file.
            token_path (str): Path to the token file for storing user credentials.
            service_account_key_path (str): Path to the service account key JSON file.
            service_account_key (Optional[dict]): Service account key info as a dict.
            authorized_user_info (Optional[dict]): Authorized user info as a dict.
            is_cloud (bool): If True, skip writing token file to disk.

        """
        self.creds = creds
        self.credentials_path = credentials_path
        self.token_path = token_path
        self.service_account_key_path = service_account_key_path
        self.service_account_key = service_account_key
        self.authorized_user_info = authorized_user_info
        self.is_cloud = is_cloud
        self.service = None

    def _cache_service(self) -> None:
        if self.service:
            return
        from googleapiclient.discovery import build

        credentials = self._get_credentials()
        self.service = build("calendar", "v3", credentials=credentials)

    def load_data(
        self,
        number_of_results: Optional[int] = 100,
        start_date: Optional[Union[str, datetime.date]] = None,
    ) -> List[Document]:
        """
        Load data from user's calendar.

        Args:
            number_of_results (Optional[int]): the number of events to return. Defaults to 100.
            start_date (Optional[Union[str, datetime.date]]): the start date to return events from in date isoformat. Defaults to today.

        """
        self._cache_service()

        if start_date is None:
            start_date = datetime.date.today()
        elif isinstance(start_date, str):
            start_date = datetime.date.fromisoformat(start_date)

        start_datetime = datetime.datetime.combine(start_date, datetime.time.min)
        start_datetime_utc = start_datetime.strftime("%Y-%m-%dT%H:%M:%S.%fZ")

        events_result = (
            self.service.events()
            .list(
                calendarId="primary",
                timeMin=start_datetime_utc,
                maxResults=number_of_results,
                singleEvents=True,
                orderBy="startTime",
            )
            .execute()
        )

        events = events_result.get("items", [])

        if not events:
            return []

        results = []
        for event in events:
            if "dateTime" in event["start"]:
                start_time = event["start"]["dateTime"]
            else:
                start_time = event["start"]["date"]

            if "dateTime" in event["end"]:
                end_time = event["end"]["dateTime"]
            else:
                end_time = event["end"]["date"]

            event_string = f"Status: {event['status']}, "
            event_string += f"Summary: {event['summary']}, "
            event_string += f"Start time: {start_time}, "
            event_string += f"End time: {end_time}, "

            organizer = event.get("organizer", {})
            display_name = organizer.get("displayName", "N/A")
            email = organizer.get("email", "N/A")
            if display_name != "N/A":
                event_string += f"Organizer: {display_name} ({email})"
            else:
                event_string += f"Organizer: {email}"

            results.append(Document(text=event_string))

        return results

    def _get_credentials(self) -> Any:
        """
        Get valid user credentials from storage.

        Credential resolution order:
        1. Pre-built ``creds`` passed to the constructor.
        2. ``service_account_key`` dict.
        3. ``service_account_key_path`` file.
        4. ``authorized_user_info`` dict.
        5. ``token_path`` file (stored OAuth tokens).
        6. ``InstalledAppFlow`` from ``credentials_path`` (desktop OAuth).

        Returns:
            Credentials, the obtained credential.

        """
        if self.creds is not None:
            return self.creds

        from google.auth.transport.requests import Request
        from google.oauth2 import service_account as sa
        from google.oauth2.credentials import Credentials
        from google_auth_oauthlib.flow import InstalledAppFlow

        if self.service_account_key is not None:
            return sa.Credentials.from_service_account_info(
                self.service_account_key, scopes=SCOPES
            )

        if os.path.isfile(self.service_account_key_path):
            with open(self.service_account_key_path, encoding="utf-8") as f:
                sa_key = json.load(f)
            return sa.Credentials.from_service_account_info(sa_key, scopes=SCOPES)

        creds = None
        if self.authorized_user_info is not None:
            creds = Credentials.from_authorized_user_info(
                self.authorized_user_info, SCOPES
            )
        elif os.path.isfile(self.token_path):
            creds = Credentials.from_authorized_user_file(self.token_path, SCOPES)

        # If there are no (valid) credentials available, let the user log in.
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    self.credentials_path, SCOPES
                )
                creds = flow.run_local_server(port=8080)
            # Save the credentials for the next run
            if not self.is_cloud:
                with open(self.token_path, "w") as token:
                    token.write(creds.to_json())

        return creds

    def create_event(
        self,
        title: Optional[str] = None,
        description: Optional[str] = None,
        location: Optional[str] = None,
        start_datetime: Optional[Union[str, datetime.datetime]] = None,
        end_datetime: Optional[Union[str, datetime.datetime]] = None,
        attendees: Optional[List[str]] = None,
    ) -> str:
        """
            Create an event on the users calendar.

        Args:
            title (Optional[str]): The title for the event
            description (Optional[str]): The description for the event
            location (Optional[str]): The location for the event
            start_datetime Optional[Union[str, datetime.datetime]]: The start datetime for the event
            end_datetime Optional[Union[str, datetime.datetime]]: The end datetime for the event
            attendees Optional[List[str]]: A list of email address to invite to the event

        """
        self._cache_service()

        attendees_list = (
            [{"email": attendee} for attendee in attendees] if attendees else []
        )

        start_time = (
            datetime.datetime.strptime(start_datetime, "%Y-%m-%dT%H:%M:%S%z")
            .astimezone()
            .strftime("%Y-%m-%dT%H:%M:%S.%f%z")
        )
        end_time = (
            datetime.datetime.strptime(end_datetime, "%Y-%m-%dT%H:%M:%S%z")
            .astimezone()
            .strftime("%Y-%m-%dT%H:%M:%S.%f%z")
        )

        event = {
            "summary": title,
            "location": location,
            "description": description,
            "start": {
                "dateTime": start_time,
            },
            "end": {
                "dateTime": end_time,
            },
            "attendees": attendees_list,
        }
        event = self.service.events().insert(calendarId="primary", body=event).execute()
        return (
            "Your calendar event has been created successfully! You can move on to the"
            " next step."
        )

    def get_date(self):
        """
        A function to return todays date. Call this before any other functions if you are unaware of the date.
        """
        return datetime.date.today()
