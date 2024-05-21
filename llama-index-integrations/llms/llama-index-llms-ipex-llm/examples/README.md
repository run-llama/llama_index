# IpexLLM Examples

This folder contains examples showcasing how to use LlamaIndex with `ipex-llm` LLM integration `llama_index.llms.ipex_llm.IpexLLM`.

## Installation

### On CPU

Install `llama-index-llms-ipex-llm`. This will also install `ipex-llm` and its dependencies.

```bash
pip install llama-index-llms-ipex-llm
```

### On GPU

Install `llama-index-llms-ipex-llm`. This will also install `ipex-llm` and its dependencies.

```bash
pip install llama-index-llms-ipex-llm[xpu] --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/
```

## List of Examples

### More Data Types Example

By default, `IpexLLM` loads the model in int4 format. To load a model in different data formats like `sym_int5`, `sym_int8`, etc., you can use the `load_in_low_bit` option in `IpexLLM`. To load a model on different device like `cpu` or `xpu`, you can use the `device_map` option in `IpexLLM`.

The example [more_data_type.py](./more_data_type.py) shows how to use the `load_in_low_bit` option and `device_map` option. Run the example as following:

```bash
python more_data_type.py -m <path_to_model> -t <path_to_tokenizer> -l <low_bit_format> -d <device>
```

> Note: If you're using [meta-llama/Llama-2-7b-hf](https://huggingface.co/meta-llama/Llama-2-7b-hf) model in this example, it is recommended to use transformers version
> <=4.34.

### RAG Example

We use llama-index-ipex-llm to build a Retrieval-Augment-Generation (RAG) pipeline.

- **Database Setup (using PostgreSQL)**:

  - Linux

    - Installation:
      ```bash
      sudo apt-get install postgresql-client
      sudo apt-get install postgresql
      ```
    - Initialization:

      Switch to the **postgres** user and launch **psql** console

      ```bash
      sudo su - postgres
      psql
      ```

      Then, create a new user role:

      ```bash
      CREATE ROLE <user> WITH LOGIN PASSWORD '<password>';
      ALTER ROLE <user> SUPERUSER;
      ```

  - Windows
    - click `Download the installer` in [PostgreSQL](https://www.postgresql.org/download/windows/).
    - Run the downloaded installation package as administrator, then click `next` continuously.
    - Open PowerShell:
    ```bash
        cd C:\Program Files\PostgreSQL\14\bin
    ```
    The exact path will vary depending on your PostgreSQL location.
    - Then in PowerShell:
    ```bash
        .\psql -U postgres
    ```
    Input the password you set in the previous installation. If PowerShell shows `postgres=#`, it indicates a successful connection.
    - Create a new user role:
    ```bash
    CREATE ROLE <user> WITH LOGIN PASSWORD '<password>';
    ALTER ROLE <user> SUPERUSER;
    ```

- **Pgvector Installation**:

  - Linux
    - Follow installation instructions on [pgvector's GitHub](https://github.com/pgvector/pgvector) and refer to the [installation notes](https://github.com/pgvector/pgvector#installation-notes) for additional help.
  - Windows
    - It is recommended to use [pgvector for Windows](https://github.com/pgvector/pgvector?tab=readme-ov-file#windows) instead of others (such as conda-force) to avoid potential errors. Some steps may require running as administrator.

- **Data Preparation**: Download the Llama2 paper and save it as `data/llama2.pdf`, which serves as the default source file for retrieval.
  ```bash
  mkdir data
  wget --user-agent "Mozilla" "https://arxiv.org/pdf/2307.09288.pdf" -O "data/llama2.pdf"
  ```

The example [rag.py](./rag.py) shows how to use RAG pipeline. Run the example as following:

```bash
python rag.py -m <path_to_model> -q <question> -u <vector_db_username> -p <vector_db_password> -e <path_to_embedding_model> -n <num_token> -t <path_to_tokenizer> -x <device>
```
