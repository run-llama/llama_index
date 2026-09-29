# LlamaIndex Tools Integration: SEOWebChecker

Automated on-page technical SEO audits, meta tag validation, Open Graph verification, and Core Web Vitals checks by [SEOWebChecker](https://seowebchecker.com/).

## Installation

```bash
pip install llama-index-tools-seowebchecker
```

## Usage

```python
from llama_index.core.agent import FunctionCallingAgent
from llama_index.llms.openai import OpenAI
from llama_index.tools.seowebchecker import SEOWebCheckerToolSpec

# 1. Initialize the tool specification
seo_spec = SEOWebCheckerToolSpec()
tools = seo_spec.to_tool_list()

# 2. Initialize an agent with the SEOWebChecker tools
agent = FunctionCallingAgent.from_tools(tools, llm=OpenAI(model="gpt-4o"))
response = agent.chat("Audit https://seowebchecker.com/ and report on-page SEO health.")
print(str(response))
```

Official web suite: [https://seowebchecker.com/](https://seowebchecker.com/)
