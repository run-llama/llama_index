# SecGPT Pack

SecGPT is an LLM-based system that secures the execution of LLM apps via isolation. The key idea behind SecGPT is to isolate the execution of apps and to allow interaction between apps and the system only through well-defined interfaces with user permission. SecGPT can defend against multiple types of attacks, including app compromise, data stealing, inadvertent data exposure, and uncontrolled system alteration. The architecture of SecGPT is shown in the figure below. Learn more about SecGPT in our [paper](https://arxiv.org/abs/2403.04960).

<p align="center"><img src="./examples/architecture.bmp" alt="workflow" width="400"></p>

We develop SecGPT using [LlamaIndex](https://www.llamaindex.ai/), an open-source LLM framework. We use LlamaIndex because it supports several LLMs and apps and can be easily extended to include additional LLMs and apps. We implement SecGPT as a personal assistant chatbot, which the users can communicate with using text messages.

A comprehensive notebook guide is available [here](./examples/SecGPT.ipynb). In the meantime, you can explore its features by comparing the execution flows of SecGPT and VanillaGPT (a non-isolated LLM-based system defined [here](./examples/VanillaGPT.ipynb)) in response to the same query.
