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
import os
from typing import Any, List, Optional, Union

from llama_index.core.schema import Document
from llama_index.core.tools.tool_spec.base import BaseToolSpec

SCOPES = ["https://www.googleapis.com/auth/calendar"]

PRIMARY_CALENDAR_ID = "primary"


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
        allowed_calendar_ids: Optional[List[str]] = None,
    ):
        """
        Initialize the GoogleCalendarToolSpec.

        Args:
            creds (Optional[Any]): Pre-configured credentials to use for authentication.
                                 If provided, these will be used instead of the OAuth flow.

        """
        self.creds = creds
        self.allowed_calendar_ids = allowed_calendar_ids or [PRIMARY_CALENDAR_ID]

    def load_data(
        self,
        number_of_results: Optional[int] = 100,
        start_date: Optional[Union[str, datetime.date]] = None,
        calendar_id: Optional[str] = PRIMARY_CALENDAR_ID,
    ) -> List[Document]:
        """
        Load data from user's calendar.

        Args:
            number_of_results (Optional[int]): the number of events to return. Defaults to 100.
            start_date (Optional[Union[str, datetime.date]]): the start date to return events from in date isoformat. Defaults to today.
            calendar_id (Optional[str]): the calendar ID to load events from. Defaults to PRIMARY_CALENDAR_ID.

        """
        validation_error = self._validate_calendar_id(calendar_id)
        if validation_error:
            return validation_error

        from googleapiclient.discovery import build

        credentials = self._get_credentials()
        service = build("calendar", "v3", credentials=credentials)

        if start_date is None:
            start_date = datetime.date.today()
        elif isinstance(start_date, str):
            start_date = datetime.date.fromisoformat(start_date)

        start_datetime = datetime.datetime.combine(start_date, datetime.time.min)
        start_datetime_utc = start_datetime.strftime("%Y-%m-%dT%H:%M:%S.%fZ")

        events_result = (
            service.events()
            .list(
                calendarId=calendar_id,
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

        The file token.json stores the user's access and refresh tokens, and is
        created automatically when the authorization flow completes for the first
        time.

        Returns:
            Credentials, the obtained credential.

        """
        if self.creds is not None:
            return self.creds

        from google_auth_oauthlib.flow import InstalledAppFlow

        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials

        creds = None
        if os.path.exists("token.json"):
            creds = Credentials.from_authorized_user_file("token.json", SCOPES)
        # If there are no (valid) credentials available, let the user log in.
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    "credentials.json", SCOPES
                )
                creds = flow.run_local_server(port=8080)
            # Save the credentials for the next run
            with open("token.json", "w") as token:
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
        calendar_id: Optional[str] = PRIMARY_CALENDAR_ID,
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
            calendar_id (Optional[str]): The calendar ID to create the event in. Defaults to PRIMARY_CALENDAR_ID.

        """
        validation_error = self._validate_calendar_id(calendar_id)
        if validation_error:
            return validation_error

        from googleapiclient.discovery import build

        credentials = self._get_credentials()
        service = build("calendar", "v3", credentials=credentials)

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
        event = service.events().insert(calendarId=calendar_id, body=event).execute()
        return (
            "Your calendar event has been created successfully! You can move on to the"
            " next step."
        )

    def _validate_calendar_id(self, calendar_id: str) -> dict:
        if calendar_id not in self.allowed_calendar_ids:
            import logging

            logger = logging.getLogger(__name__)
            logger.error(
                f"Invalid calendar ID '{calendar_id}' attempted. Valid IDs: {self.allowed_calendar_ids}"
            )
            return {
                "error": "Invalid calendar_id",
                "allowed_values": list(self.allowed_calendar_ids),
            }
        return None

    def get_date(self):
        """
        A function to return todays date. Call this before any other functions if you are unaware of the date.
        """
        return datetime.date.today()


def all_calendars(creds) -> List[str]:
    """List all accessible calendar IDs for configuration purposes."""
    from googleapiclient.discovery import build

    service = build("calendar", "v3", credentials=creds)
    calendar_list = service.calendarList().list().execute()
    return [cal["id"] for cal in calendar_list.get("items", [])]
