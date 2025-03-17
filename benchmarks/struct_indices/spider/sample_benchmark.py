"""Sample a fraction of the Spider dataset."""

import argparse
import json
import os
import random
import shutil

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create a sampled version of the Spider dataset."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to the Spider dataset directory. "
        "This directory should contain the train.json, dev.json, "
        "and databases, "
        "downloaded from https://yale-lily.github.io/spider.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to the output directory of the sampled benchmark.",
    )
    parser.add_argument(
        "--sample-factor",
        type=float,
        required=True,
        help="The sample factor to apply to sample a fraction "
        "of examples in both the train and dev datasets.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    args = parser.parse_args()

    # Create the output directory if it does not exist.
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    # Load the Spider dataset from the input directory.
    with open(os.path.join(args.input, "train_spider.json")) as f:
        train_spider = json.load(f)
    with open(os.path.join(args.input, "train_others.json")) as f:
        train_others = json.load(f)
    with open(os.path.join(args.input, "dev.json")) as f:
        dev = json.load(f)

    # Randomly sample (without replacement) the indices using the sample factor.
    random.seed(args.seed)
    train_spider_indices = list(range(len(train_spider)))
    train_others_indices = list(range(len(train_others)))
    dev_indices = list(range(len(dev)))
    train_spider_indices = random.choices(
        train_spider_indices, k=int(args.sample_factor * len(train_spider_indices))
    )
    train_others_indices = random.choices(
        train_others_indices, k=int(args.sample_factor * len(train_others_indices))
    )
    dev_indices = random.choices(
        dev_indices, k=int(args.sample_factor * len(dev_indices))
    )
    # Sort the indices to ensure same ordering as the original sql files.
    train_spider_indices.sort()
    train_others_indices.sort()
    dev_indices.sort()

    # Write the sampled datasets to the output directory.
    with open(os.path.join(args.output, "train_spider.json"), "w") as f:
        json.dump([train_spider[i] for i in train_spider_indices], f, indent=2)
    with open(os.path.join(args.output, "train_others.json"), "w") as f:
        json.dump([train_others[i] for i in train_others_indices], f, indent=2)
    with open(os.path.join(args.output, "dev.json"), "w") as f:
        json.dump([dev[i] for i in dev_indices], f, indent=2)

    # Write the sql files to the output directory.
    with open(os.path.join(args.output, "train_gold.sql"), "w") as f:
        for i in train_spider_indices:
            f.write(
                train_spider[i]["query"].replace("\t", " ")
                + "\t"
                + train_spider[i]["db_id"]
                + "\n"
            )
        for i in train_others_indices:
            f.write(
                train_others[i]["query"].replace("\t", " ")
                + "\t"
                + train_others[i]["db_id"]
                + "\n"
            )
    with open(os.path.join(args.output, "dev_gold.sql"), "w") as f:
        for i in dev_indices:
            f.write(dev[i]["query"] + "\t" + dev[i]["db_id"] + "\n")

    # Copy the database to the output directory.
    shutil.copytree(
        os.path.join(args.input, "database"),
        os.path.join(args.output, "database"),
        dirs_exist_ok=True,
    )

    # Copy the tables.json file to the output directory.
    shutil.copyfile(
        os.path.join(args.input, "tables.json"),
        os.path.join(args.output, "tables.json"),
    )

    # Print results.
    print(f"Sampled {len(train_spider_indices)} examples from train_spider.json.")
    print(f"Sampled {len(train_others_indices)} examples from train_others.json.")
    print(f"Sampled {len(dev_indices)} examples from dev.json.")
    print(f"All files written to {args.output}.")
