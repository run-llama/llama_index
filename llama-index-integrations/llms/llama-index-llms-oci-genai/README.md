# LlamaIndex Llms Integration: Oracle Cloud Infrastructure (OCI) Generative AI

> Oracle Cloud Infrastructure (OCI) [Generative AI](https://docs.oracle.com/en-us/iaas/Content/generative-ai/home.htm) is a fully managed service that provides a set of state-of-the-art,
> customizable large language models (LLMs) that cover a wide range of use cases, and which are available through a single API.
> Using the OCI Generative AI service you can access ready-to-use pretrained models, or create and host your own fine-tuned
> custom models based on your own data on dedicated AI clusters.

## Installation

```bash
pip install llama-index-llms-oci-genai
```

You will also need to install the OCI sdk

```bash
pip install -U oci
```

## Usage

```bash
from llama_index.llms.oci_genai import OCIGenAI

llm = OCIGenAI(
    model="MY_MODEL",
    service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
    compartment_id="MY_OCID",
)
```
