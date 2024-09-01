# SecGPT Pack

SecGPT is an LLM-based system that secures the execution of LLM apps via isolation. The key idea behind SecGPT is to isolate the execution of apps and to allow interaction between apps and the system only through well-defined interfaces with user permission. SecGPT can defend against multiple types of attacks, including app compromise, data stealing, inadvertent data exposure, and uncontrolled system alteration. The architecture of SecGPT is shown in the figure below. Learn more about SecGPT in our [paper](https://arxiv.org/abs/2403.04960).

<p align="center">
  <img src="https://raw.githubusercontent.com/run-llama/llama_index/main/llama-index-packs/llama-index-packs-secgpt/examples/architecture.bmp" alt="Architecture" width="400">
</p>

We develop SecGPT using [LlamaIndex](https://www.llamaindex.ai/), an open-source LLM framework. We use LlamaIndex because it supports several LLMs and apps and can be easily extended to include additional LLMs and apps. We implement SecGPT as a personal assistant chatbot, which the users can communicate with using text messages.

A comprehensive notebook guide is available [here](https://github.com/run-llama/llama_index/blob/main/llama-index-packs/llama-index-packs-secgpt/examples/SecGPT.ipynb). In the meantime, you can explore its features by comparing the execution flows of SecGPT and VanillaGPT (a non-isolated LLM-based system defined [here](https://github.com/run-llama/llama_index/blob/main/llama-index-packs/llama-index-packs-secgpt/examples/VanillaGPT.ipynb)) in response to the same query.

SecGPT original GitHub repository link: [https://github.com/llm-platform-security/SecGPT](https://github.com/llm-platform-security/SecGPT)

If you build on this work, considering citing our paper:

## Citation

```plaintext
@article{wu2024secgpt,
  title={{SecGPT: An Execution Isolation Architecture for LLM-Based Systems}},
  author={Wu, Yuhao and Roesner, Franziska and Kohno, Tadayoshi and Zhang, Ning and Iqbal, Umar},
  journal={arXiv preprint arXiv:2403.04960},
  year={2024},
}
```

## Contribution and Support

We welcome contributions to the project, e.g., through pull requests to the [original GitHub repo](https://github.com/llm-platform-security/SecGPT). Please also feel free to reach out to us if you have questions about the project and if you would like to contribute.

## Research Team

[Yuhao Wu](https://yuhao-w.github.io) (Washington University in St. Louis)
[Franziska Roesner](https://www.franziroesner.com/) (University of Washington)
[Tadayoshi Kohno](https://homes.cs.washington.edu/~yoshi/) (University of Washington)
[Ning Zhang](https://cybersecurity.seas.wustl.edu/) (Washington University in St. Louis)
[Umar Iqbal](https://umariqbal.com) (Washington University in St. Louis)
