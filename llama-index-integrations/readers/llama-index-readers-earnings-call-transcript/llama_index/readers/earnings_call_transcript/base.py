from datetime import datetime
from typing import List

from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document
from llama_index.readers.earnings_call_transcript.utils import get_earnings_transcript


class EarningsCallTranscript(BaseReader):
    def __init__(self, year: int, ticker: str, quarter: str):
        """
        Get the earning call transcripts for a given company, in a given year and quarter.

        Args:
            year (int): Year of the transcript
            ticker (str): ticker symbol of the stock
            quarter (str): quarter

        """
        curr_year = datetime.now().year
        assert year <= curr_year, "The year should be less than current year"

        assert quarter in [
            "Q1",
            "Q2",
            "Q3",
            "Q4",
        ], 'The quarter should from the list ["Q1","Q2","Q3","Q4"]'
        self.year = year
        self.ticker = ticker
        self.quarter = quarter

    def load_data(self) -> List[Document]:
        resp_dict, speakers_list = get_earnings_transcript(
            self.quarter, self.ticker, self.year
        )
        return Document(
            text=resp_dict["content"],
            extra_info={
                "ticker": resp_dict["symbol"],
                "quarter": "Q" + str(resp_dict["quarter"]),
                "date_time": resp_dict["date"],
                "speakers_list": speakers_list,
            },
        )
