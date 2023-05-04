#!/bin/bash

# build package
PACKAGE_NAME_OVERRIDE=gpt_index python setup.py sdist bdist_wheel

# publish gpt_index package
twine upload dist/*

# NOTE: use this to test
# twine upload -r testpypi dist/*

# cleanup
rm -rf build dist *.egg-info

