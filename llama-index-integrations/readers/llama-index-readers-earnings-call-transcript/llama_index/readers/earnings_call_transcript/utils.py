import json
import re
from datetime import datetime
from typing import List

import requests
from tenacity import retry, stop_after_attempt, wait_random_exponential


def correct_date(yr, dt):
    """
    Some transcripts have incorrect date, correcting it.

    Args:
        yr (int): actual
        dt (datetime): given date

    Returns:
        datetime: corrected date

    """
    dt = datetime.strptime(dt, "%Y-%m-%d %H:%M:%S")
    if dt.year != yr:
        dt = dt.replace(year=yr)
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def extract_speakers(cont: str) -> List[str]:
    """
    Extract the list of speakers.

    Args:
        cont (str): transcript content

    Returns:
        List[str]: list of speakers

    """
    pattern = re.compile(r"\n(.*?):")
    matches = pattern.findall(cont)

    return list(set(matches))


@retry(wait=wait_random_exponential(min=1, max=5), stop=stop_after_attempt(2))
def get_earnings_transcript(quarter: str, ticker: str, year: int):
    """
    Get the earnings transcripts.

    Args:
        quarter (str)
        ticker (str)
        year (int)

    """
    response = requests.get(
        f"https://discountingcashflows.com/api/transcript/{ticker}/{quarter}/{year}/",
        auth=("user", "pass"),
    )

    resp_text = json.loads(response.text)
    speakers_list = extract_speakers(resp_text[0]["content"])
    corrected_date = correct_date(resp_text[0]["year"], resp_text[0]["date"])
    resp_text[0]["date"] = corrected_date
    return resp_text[0], speakers_list
