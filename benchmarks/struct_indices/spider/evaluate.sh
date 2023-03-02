# Git clone the spider evaluation repo if it doesn't exist.
if [ ! -d "spider-evaluation" ]; then
  git clone https://github.com/taoyds/spider.git spider-evaluation
fi

BENCHMARK_DIR=$1
PREDICTIONS_DIR=$2

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
