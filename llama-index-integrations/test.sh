for d in $1/*/ ; do (
    cd "$d" &&
    echo "$d" &&
    poetry install -q &&
    poetry update flying-delta-core -q;
    FAILS=$(poetry run pytest tests --tb=line | grep -c "FAILED")
    if [ $FAILS -eq 0 ]; then
           echo ...PASSED
      else
           echo ...FAILED
      fi
  );
done
