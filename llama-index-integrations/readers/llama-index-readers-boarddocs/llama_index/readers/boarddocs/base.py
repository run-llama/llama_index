"""Reader that pulls in a BoardDocs site."""

import json
from typing import Any, List, Optional

import html2text
import requests
from bs4 import BeautifulSoup
from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document


class BoardDocsReader(BaseReader):
    """
    BoardDocs doc reader.

    Read public agendas included on a BoardDocs site.

    Args:
        site (str): The BoardDocs site you'd like to index, e.g. "ca/redwood"
        committee_id (str): The committee on the site you want to index

    """

    def __init__(
        self,
        site: str,
        committee_id: str,
    ) -> None:
        """Initialize with parameters."""
        self.site = site
        self.committee_id = committee_id
        self.base_url = "https://go.boarddocs.com/" + site + "/Board.nsf"

        # set up the headers required for the server to answer
        self.headers = {
            "accept": "application/json, text/javascript, */*; q=0.01",
            "accept-language": "en-US,en;q=0.9",
            "content-type": "application/x-www-form-urlencoded; charset=UTF-8",
            "sec-ch-ua": (
                '"Google Chrome";v="113", "Chromium";v="113", "Not-A.Brand";v="24"'
            ),
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": '"macOS"',
            "sec-fetch-dest": "empty",
            "sec-fetch-mode": "cors",
            "sec-fetch-site": "same-origin",
            "x-requested-with": "XMLHttpRequest",
        }
        super().__init__()

    def get_meeting_list(self) -> List[dict]:
        """
        Returns a list of meetings for the committee.

        Args:
            None
        Returns:
            List[dict]: A list of meetings, each with a meetingID, date, and unid

        """
        meeting_list_url = self.base_url + "/BD-GetMeetingsList?open"

        data = "current_committee_id=" + self.committee_id
        response = requests.post(meeting_list_url, headers=self.headers, data=data)
        meetingsData = json.loads(response.text)

        return [
            {
                "meetingID": meeting.get("unique", None),
                "date": meeting.get("numberdate", None),
                "unid": meeting.get("unid", None),
            }
            for meeting in meetingsData
        ]

    def process_meeting(
        self, meeting_id: str, index_pdfs: bool = True
    ) -> List[Document]:
        """
        Returns documents from the given meeting.
        """
        agenda_url = self.base_url + "/PRINT-AgendaDetailed"

        # set the meetingID & committee
        data = "id=" + meeting_id + "&" + "current_committee_id=" + self.committee_id

        # POST the request!
        response = requests.post(agenda_url, headers=self.headers, data=data)

        # parse the returned HTML
        soup = BeautifulSoup(response.content, "html.parser")
        agenda_date = soup.find("div", {"class": "print-meeting-date"}).string
        agenda_title = soup.find("div", {"class": "print-meeting-name"}).string
        [fd.a.get("href") for fd in soup.find_all("div", {"class": "public-file"})]
        agenda_data = html2text.html2text(response.text)

        # TODO: index the linked PDFs in agenda_files!

        docs = []
        agenda_doc = Document(
            text=agenda_data,
            doc_id=meeting_id,
            extra_info={
                "committee": self.committee_id,
                "title": agenda_title,
                "date": agenda_date,
                "url": agenda_url,
            },
        )
        docs.append(agenda_doc)
        return docs

    def load_data(
        self, meeting_ids: Optional[List[str]] = None, **load_kwargs: Any
    ) -> List[Document]:
        """
        Load all meetings of the committee.

        Args:
            meeting_ids (List[str]): A list of meeting IDs to load. If None, load all meetings.

        """
        # if a list of meetings wasn't provided, enumerate them all
        if not meeting_ids:
            meeting_ids = [
                meeting.get("meetingID") for meeting in self.get_meeting_list()
            ]

        # process all relevant meetings & return the documents
        docs = []
        for meeting_id in meeting_ids:
            docs.extend(self.process_meeting(meeting_id))
        return docs
