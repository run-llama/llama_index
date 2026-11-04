import datetime
from unittest.mock import patch

from llama_index.core.readers.base import BaseReader
from llama_index.readers.toggl import TogglReader


def test_class():
    names_of_base_classes = [b.__name__ for b in TogglReader.__mro__]
    assert BaseReader.__name__ in names_of_base_classes


def test_load_data_default_end_date_is_call_time():
    reader = TogglReader(api_token="token")
    before = datetime.datetime.now()
    with patch.object(TogglReader, "aload_data") as aload_data:
        aload_data.return_value = []
        reader.load_data(workspace_id="ws", project_id="proj")
    after = datetime.datetime.now()

    end_date = aload_data.call_args.args[3]
    assert before <= end_date <= after


def test_load_data_keeps_explicit_end_date():
    reader = TogglReader(api_token="token")
    end = datetime.datetime(2024, 1, 31, 12, 0)
    with patch.object(TogglReader, "aload_data") as aload_data:
        aload_data.return_value = []
        reader.load_data(workspace_id="ws", project_id="proj", end_date=end)

    assert aload_data.call_args.args[3] == end
