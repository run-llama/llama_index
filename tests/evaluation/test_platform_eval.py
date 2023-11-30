import os
import pytest
from llama_index_client.client import PlatformApi

from llama_index.evaluation.eval_utils import upload_eval_dataset

base_url = os.environ.get("PLATFORM_BASE_URL", None)
api_key = os.environ.get("PLATFORM_API_KEY", None)

@pytest.mark.skipif(
        not base_url or not api_key, 
        reason="No platform base url or api keyset"
)
@pytest.mark.integration()
def test_upload_eval_dataste() -> None:
    eval_dataset_id = upload_eval_dataset(
        "test_dataset",
        questions=["foo", "bar"],
        overwrite=True,
    )

    client = PlatformApi(base_url=base_url, token=api_key)
    eval_dataset = client.eval.get_dataset(dataset_id=eval_dataset_id)
    assert eval_dataset.name == "test_dataset"

    eval_questions = client.eval.get_questions(dataset_id=eval_dataset_id)
    assert len(eval_questions) == 2
