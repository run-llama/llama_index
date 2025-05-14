import pytest

from llama_index.postprocessor.ibm import WatsonxRerank


class TestWasonxRerank:
    TEST_URL = "https://us-south.ml.cloud.ibm.com"
    TEST_APIKEY = "test_apikey"
    TEST_PROJECT_ID = "test_project_id"

    TEST_MODEL = "test_rerank_model"

    def test_initialization(self) -> None:
        with pytest.raises(ValueError, match=r"^Did not find") as e_info:
            _ = WatsonxRerank(model_id=self.TEST_MODEL, project_id=self.TEST_PROJECT_ID)

        # Cloud scenario
        with pytest.raises(
            ValueError, match=r"^Did not find 'apikey' or 'token',"
        ) as e_info:
            _ = WatsonxRerank(
                model_id=self.TEST_MODEL,
                url=self.TEST_URL,
                project_id=self.TEST_PROJECT_ID,
            )

        # CPD scenario
        with pytest.raises(ValueError, match=r"^Did not find instance_id") as e_info:
            _ = WatsonxRerank(
                model_id=self.TEST_MODEL,
                token="test-token",
                url="test-cpd-instance",
                project_id=self.TEST_PROJECT_ID,
            )
