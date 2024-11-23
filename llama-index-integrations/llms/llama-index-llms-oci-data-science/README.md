# LlamaIndex Llms Integration: Oracle Cloud Infrastructure (OCI) Data Science Service

Oracle Cloud Infrastructure (OCI) [Data Science](https://www.oracle.com/artificial-intelligence/data-science) is a fully managed and serverless platform for data science teams to build, train, and manage machine learning models in Oracle Cloud Infrastructure.

It offers the [AI Quick Actions](https://docs.oracle.com/en-us/iaas/data-science/using/ai-quick-actions.htm) that can be used to deploy, evaluate and fine tune foundation models in OCI Data Science. AI Quick Actions target a user who wants to quickly leverage the capabilities of AI. They aim to expand the reach of foundation models to a broader set of users by providing a streamlined, code-free and efficient environment for working with foundation models. AI Quick Actions can be accessed from the Data Science Notebook.


## Installation

Install the required packages:

```bash
pip install llama-index-llms-oci-data-science oralce-ads
```

The [oracle-ads](https://accelerated-data-science.readthedocs.io/en/latest/index.html) is required to simplify the authentication within OCI Data Science.


## Basic Usage

```bash
from llama_index.llms.oci_data_science import OCIDataScience

TBD
```

## LLM Implementation example

https://docs.llamaindex.ai/en/stable/examples/llm/oci_data_science/
