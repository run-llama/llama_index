Command line interface (experimental)
========

This module providers a way to interactive with llama\_index directly in shell.

Current supported commands:

```shell
# create a local config file in local dir
python -m experimental.cli init

# add file to index
python -m experimental.cli add ../data/

# query
python -m experimental.cli query "Some question?"
```

There're two files put in current directory.

- config.ini stores embedding/predicter model setup along with its parameters
- index.json the index file

