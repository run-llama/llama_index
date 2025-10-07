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
from zoneinfo import ZoneInfo
import os
from typing import Any, List, Optional, Union

from llama_index.core.tools.tool_spec.base import BaseToolSpec
from googleapiclient.discovery import build

SCOPES = ["https://www.googleapis.com/auth/calendar"]
CALENDAR_DAYS_PADDING = 32
DEFAULT_MONTHS_RANGE = 9
# set to tzinfo "America/Los_Angeles"
DEFAULT_TIME_ZONE = ZoneInfo("America/Los_Angeles")


class GoogleCalendarToolSpec(BaseToolSpec):
    """
    Google Calendar tool spec.

    Currently a simple wrapper around the data loader.
    TODO: add more methods to the Google Calendar spec.

    """

    spec_functions = ["list_events", "create_event", "get_current_date", "day_of_week_for_date"]

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
        self.allowed_calendar_ids = allowed_calendar_ids

    def day_of_week_for_date(self,
                    date: Union[str, datetime.date]) -> str:
        """
        Get the day of the week for a given date.
        """
        if isinstance(date, str):
            date = datetime.date.fromisoformat(date)
        return date.strftime("%A")

    def _convert_to_date(self, date: Optional[Union[str, datetime.date]]) -> datetime.date:
        """
        Convert a given date to a datetime.date object.
        """
        if date is None:
            date = datetime.date.today()
        elif isinstance(date, str):
            date = datetime.date.fromisoformat(date)
        return date

    def _convert_to_timestamp_str(self, date: Optional[Union[str, datetime.date]], end_of_day = False, timezone = DEFAULT_TIME_ZONE) -> str:
        """
        Convert a given date to string format in the given timezone.

        Args:
            date (Optional[Union[str, datetime.date]]): The date to convert. If None, defaults to today.

        Returns:
            str: The UTC formatted string.
        """
        print(f"Converting to timestamp str for date: {date}, end_of_day: {end_of_day}, timezone: {timezone}")
        date = self._convert_to_date(date)
        print(f"Date converted to: {date}")
        start_datetime = datetime.datetime.combine(date, datetime.time.max if end_of_day else datetime.time.min)
        print(f"Start datetime combined: {start_datetime}")
        start_datetime = start_datetime.replace(tzinfo=timezone)
        print(f"Start datetime converted to: {start_datetime}")
        print(f"Start datetime formatted: {start_datetime.strftime('%Y-%m-%dT%H:%M:%S%z')}")
        return start_datetime.strftime("%Y-%m-%dT%H:%M:%S%z")

    def is_room_available(self, start_date: Optional[Union[str, datetime.date]], calendar_id: Optional[str] = None) -> bool:
        """
        Check if the calendar is available on a particular date.
        """
        start_date = self._convert_to_date(start_date)
        end_date = start_date + datetime.timedelta(days=1)

        events = self.get_events(start_date=start_date, end_date=end_date, calendar_id=calendar_id)
        return events is None or len(events) == 0

    def list_events(
        self,
        start_date: Optional[Union[str, datetime.date]] = None,
        end_date: Optional[Union[str, datetime.date]] = None,
        calendar_id: Optional[str] = None,
    ) -> List[dict]:
        """
        Get events from a google calendar.

        Args:
            start_date (Optional[Union[str, datetime.date]]): the start date to return events from in date isoformat. Defaults to today.
            calendar_id (Optional[str]): the calendar ID to load events from. Must be provided and in allowed_calendar_ids list.

        """
        self.init_allowed_calendars()

        if calendar_id:
            validation_error = self._validate_calendar_id(calendar_id)
            if validation_error:
                return validation_error
            calendars = [calendar_id]
        else:
            calendars = self.allowed_calendar_ids

        start_date = self._convert_to_date(start_date)
        number_of_results=1000
        months_range = DEFAULT_MONTHS_RANGE
        if end_date:
            end_date = self._convert_to_date(end_date)
        else:
            end_date = start_date + datetime.timedelta(days=months_range * 30)

        events = []
        for calendar in calendars:
            events_list = self.get_events(number_of_results=number_of_results, start_date=start_date, end_date=end_date, calendar_id=calendar)
            events.extend(events_list)

        if not events:
            return []

        events_by_date = {}
        max_end_date = None
        for event in events:
            date_key = event["date"]
            if date_key not in events_by_date:
                events_by_date[date_key] = []
            events_by_date[date_key].append(event)
            max_end_date = event["end_date"] if max_end_date is None else max(max_end_date, event["end_date"])

        start_datetime_str = self._convert_to_timestamp_str(start_date)

        # if the number of events is less than requested, that means there are no more events on the calendar, so add some padding
        # but don't go past the end date because there may be events after the end date
        if len(events) < number_of_results:
            until_date = max_end_date + datetime.timedelta(days=CALENDAR_DAYS_PADDING)
            until_date = min(until_date, end_date)
        else:
            # otherwise, go until the max end date of the events
            until_date = max_end_date

        results = []
        current_date = datetime.datetime.fromisoformat(start_datetime_str).date()
        while current_date <= until_date:
            result = {"date": current_date.strftime("%A, %B %d, %Y")}
            result["events"] = events_by_date[result["date"]] if result["date"] in events_by_date else []
            results.append(result)
            current_date += datetime.timedelta(days=1)

        return results

    def get_events(
        self,
        number_of_results: Optional[int] = 100,
        start_date: Optional[Union[str, datetime.date]] = None,
        end_date: Optional[Union[str, datetime.date]] = None,
        calendar_id: Optional[str] = None,
    ) -> List[dict]:

        credentials = self._get_credentials()
        service = build("calendar", "v3", credentials=credentials)

        end_datetime_str = self._convert_to_timestamp_str(end_date, end_of_day=True) if end_date else None
        start_datetime_str = self._convert_to_timestamp_str(start_date)

        events_result = (
            service.events()
            .list(
                calendarId=calendar_id,
                timeMin=start_datetime_str,
                timeMax=end_datetime_str,
                maxResults=number_of_results,
                singleEvents=True,
                timeZone=DEFAULT_TIME_ZONE,
                orderBy="startTime",
            )
            .execute()
        )

        events = events_result.get("items", [])

        if not events:
            return []

        results = []
        for event in events:
            start_time = datetime.datetime.fromisoformat(event["start"]["dateTime"] if "dateTime" in event["start"] else event["start"]["date"])
            end_time = datetime.datetime.fromisoformat(event["end"]["dateTime"] if "dateTime" in event["end"] else event["end"]["date"])

            result = {
                "status": event['status'], "summary": event['summary'],
                "date": start_time.date().strftime("%A, %B %d, %Y"),
                "start": start_time.strftime("%I:%M %p"), "end": end_time.strftime("%I:%M %p"),
                "end_date": end_time.date()
            }

            organizer = event.get("organizer", {})
            display_name = organizer.get("displayName", "N/A")
            result["calendar_name"] = display_name
            results.append(result)
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
        calendar_id: Optional[str] = None,
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
            calendar_id (Optional[str]): The calendar ID to create the event in. Must be provided and in allowed_calendar_ids list.

        """
        if calendar_id is None:
            return "Error: calendar_id is required"

        validation_error = self._validate_calendar_id(calendar_id)
        if validation_error:
            return validation_error

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
        if not event or "id" not in event:
            return "Error creating event"
        return (
            "Your calendar event has been created successfully! You can move on to the"
            " next step."
        )

    def init_allowed_calendars(self):
        # TODO: Normally this would be in the init function but it causes problems with the tests
        if self.allowed_calendar_ids is None:
            self.allowed_calendar_ids = all_calendars(self._get_credentials())

    def _validate_calendar_id(self, calendar_id: str) -> dict:
        self.init_allowed_calendars()
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

    def get_current_date(self):
        """
        A function to return today's date.
        """
        return datetime.date.today().strftime("%A, %B %d, %Y")


def all_calendars(creds) -> List[str]:
    """List all accessible calendar IDs for configuration purposes."""
    service = build("calendar", "v3", credentials=creds)
    calendar_list = service.calendarList().list().execute()
    return [cal["id"] for cal in calendar_list.get("items", [])]
