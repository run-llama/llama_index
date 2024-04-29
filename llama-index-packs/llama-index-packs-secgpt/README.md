# SecGPT Pack

SecGPT is an LLM-based system that secures the execution of LLM apps via isolation. The key idea behind SecGPT is to isolate the execution of apps and to allow interaction between apps and the system only through well-defined interfaces with user permission. SecGPT can defend against multiple types of attacks, including app compromise, data stealing, inadvertent data exposure, and uncontrolled system alteration. Learn more about SecGPT in our [paper](https://arxiv.org/abs/2403.04960).

We develop SecGPT using [LlamaIndex](https://www.llamaindex.ai/), an open-source LLM framework. We use LlamaIndex because it supports several LLMs and apps and can be easily extended to include additional LLMs and apps. We implement SecGPT as a personal assistant chatbot, which the users can communicate with using text messages.

A full notebook guide can be found [here](./examples/SecGPT.ipynb).
