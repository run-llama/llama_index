import pytest

from llama_index.postprocessor.ibm import WatsonxRerank


class TestWasonxRerank:
    TEST_URL = "https://us-south.ml.cloud.ibm.com"
    TEST_APIKEY = "test_apikey"
    TEST_PROJECT_ID = "test_project_id"

    TEST_MODEL = "test_rerank_model"

    def test_initialization(self) -> None:
        with pytest.raises(ValueError, match=r"^Did not find"):
            _ = WatsonxRerank(model_id=self.TEST_MODEL, project_id=self.TEST_PROJECT_ID)

        # Cloud scenario
        with pytest.raises(ValueError, match=r"^Did not find 'apikey' or 'token',"):
            _ = WatsonxRerank(
                model_id=self.TEST_MODEL,
                url=self.TEST_URL,
                project_id=self.TEST_PROJECT_ID,
            )

        # CPD scenario with password and missing username
        with pytest.raises(ValueError, match=r"^Did not find username"):
            _ = WatsonxRerank(
                model_id=self.TEST_MODEL,
                password="123",
                url="cpd-instance",
                project_id=self.TEST_PROJECT_ID,
            )

        # CPD scenario with apikey and missing username
        with pytest.raises(ValueError, match=r"^Did not find username"):
            _ = WatsonxRerank(
                model_id=self.TEST_MODEL,
                apikey="123",
                url="cpd-instance",
                project_id=self.TEST_PROJECT_ID,
            )
