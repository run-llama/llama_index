# 💡 Contributing to GPT Index

Interested in contributing to GPT Index? Here's how to get started! 

## Contributions that we're looking for:
- Bug fixes
- New features

All future tasks are tracked in [Github Issues Page](https://github.com/jerryjliu/gpt_index/issues).
Please feel free to open an issue and/or assign an issue to yourself.

Also, join our Discord for discussions: https://discord.gg/dGcwcsnxhU.

## Environment Setup

GPT Index is a Python package. We've tested primarily with Python versions >= 3.8. Here's a quick
and dirty guide to getting your environment setup.

First, create a fork of GPT Index, by clicking the "Fork" button on the [GPT Index Github page](https://github.com/jerryjliu/gpt_index).
Following [these steps](https://docs.github.com/en/get-started/quickstart/fork-a-repo) for more details
on how to fork the repo and clone the forked repo.

Then, create a new Python virtual environment. The command below creates an environment in `.venv`,
and activates it:
```bash
python -m venv .venv
source .venv/bin/activate
```

Install the required dependencies (this will also install gpt-index through `pip install -e .` 
so that you can start developing on it):

```bash
pip install -r requirements.txt
```

Now you should be set! 


## Validating your Change

Let's make sure to `format/lint` our change. For bigger changes,
let's also make sure to `test` it and perhaps create an `example notebook`.

### Formatting/Linting

You can format and lint your changes with the following commands in the root directory:

```bash
make format; make lint
```

We run an assortment of linters: `black`, `isort`, `mypy`, `flake8`.

### Testing

For bigger changes, you'll want to create a unit test. Our tests are in the `tests` folder.
We use `pytest` for unit testing. To run all unit tests, run the following in the root dir:

```bash
pytest tests
```

### Creating an Example Notebook

For changes that involve entirely new features, it may be worth adding an example Jupyter notebook to showcase
this feature. 

Example notebooks can be found in this folder: https://github.com/jerryjliu/gpt_index/tree/main/examples.


### Creating a pull request

See [these instructions](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request-from-a-fork)
to open a pull request against the main GPT Index repo.













