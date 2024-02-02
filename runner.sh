#!/bin/sh

find . -name "pyproject.toml" | xargs -n 1 dirname > pypackages.txt

cat pypackages.txt | while read line
do
        echo $line
        cd $line;
        poetry install -q && poetry update flying-delta-core -q;
        FAILS=$(poetry run pytest tests --tb=line | grep -c "FAILED");
        if [ $FAILS -eq 0 ]; then
           echo ...PASSED
        else
            echo ...FAILED
        fi
        cd -
done
