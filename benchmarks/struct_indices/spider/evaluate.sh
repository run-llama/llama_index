# Git clone the spider evaluation repo if it doesn't exist.
if [ ! -d "spider-evaluation" ]; then
  git clone https://github.com/taoyds/spider.git spider-evaluation
fi

BENCHMARK_DIR=$1
PREDICTIONS_DIR=$2

# Check if gold and pred SQL files have the same number of lines.
if [ $(wc -l < $BENCHMARK_DIR/train_gold.sql) -ne $(wc -l < $PREDICTIONS_DIR/train_pred.sql) ]; then
  echo "Number of lines in train_gold.sql and train_pred.sql do not match."
  exit 1
fi
if [ $(wc -l < $BENCHMARK_DIR/dev_gold.sql) -ne $(wc -l < $PREDICTIONS_DIR/dev_pred.sql) ]; then
  echo "Number of lines in dev_gold.sql and dev_pred.sql do not match."
  exit 1
fi

# Run the evaluation script for training examples.
python spider-evaluation/evaluation.py \
  --gold $BENCHMARK_DIR/train_gold.sql \
  --pred $PREDICTIONS_DIR/train_pred.sql \
  --db $BENCHMARK_DIR/database \
  --table $BENCHMARK_DIR/tables.json \
  --etype all > $PREDICTIONS_DIR/train_eval.txt

# Run the evaluation script for dev examples.
python spider-evaluation/evaluation.py \
  --gold $BENCHMARK_DIR/dev_gold.sql \
  --pred $PREDICTIONS_DIR/dev_pred.sql \
  --db $BENCHMARK_DIR/database \
  --table $BENCHMARK_DIR/tables.json \
  --etype all > $PREDICTIONS_DIR/dev_eval.txt
