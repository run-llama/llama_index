# Evaluation using Spider Text-to-SQL Dataset

We want to benchmark LlamaIndex's performance for complex queries on
multiple domains, and measure how each iteration of LLM improves its
Text-to-SQL capability, thus this project.

## Usage

1. Download [benchmark dataset](https://yale-lily.github.io/spider),
the download link is in the left-side bar under section "Get Started". Unzip the file after download.
2. Use `sample_benchmark.py` to sample the benchmark dataset so we don't spend too much money when testing. Skip this step when running the complete benchmark.

```bash
python sample_benchmark.py --input <benchmark path> --output spider-0_001 --sample-factor 0.001
# A smaller benchmark with 1/1000 examples is saved in directory spider-0_001, which we use as our benchmark for testing purpose.
```

3. Use `generate_sql.py` to generate the predicted SQL queries given the input benchmark.

```bash
python generate_sql.py --input spider-0_001 --output spider-0_001-pred
# Predicted SQLs are saved in the output directory.
```

4. Use `evaluate.sh` to evaluate the prediction. The script will download the [Spider Evaluation](https://github.com/taoyds/spider)
code and use it to generate performance reports saved in the
same directory as the predicted SQLs. See [here](https://github.com/taoyds/spider/tree/master/evaluation_examples) to understand the
evaluation metrics.

```bash
./evaluate.sh spider-0_001 spider-0_001-pred
```

## TODO

1. Include the complete schema in the prompt.
2. Auto-course-correction encountering SQL errors using Langchain agent.
3. Use training set to generate in-context learning examples.
