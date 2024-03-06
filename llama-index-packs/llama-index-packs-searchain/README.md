# LlamaIndex Packs Integration: Searchain
This LlamaPack implements short form the [SearChain paper by Xu et al..](https://arxiv.org/abs/2304.14732)

You can see its use case in the examples folder.

This implementation is adapted from the author's implementation. You can find the official code repository [here](https://github.com/xsc1234/Search-in-the-Chain).

## Code Usage
First, you need to install SearChainpack using the following code,
```python
from llama_index.core.llama_pack import download_llama_pack

download_llama_pack(
    "SearChainPack",
    "./searchain_pack"
)
```
Next you can load and initialize a searchain object,
```python
from searchain_pack.base import SearChainPack

searchain = SearChainPack(data_path = 'data',dprtokenizer_path = 'dpr_reader_multi',
                          dprmodel_path = 'dpr_reader_multi',crossencoder_name_or_path = 'Quora_cross_encoder')
```
Relevant data can be found [here](https://www.kaggle.com/datasets/anastasiajia/searchain/data). You can run searchain using the following method,
```python
start_idx = 0
while not start_idx == -1:
    start_idx = excute('/hotpotqa/hotpot_dev_fullwiki_v1_line.json',
           start_idx=start_idx)
```
