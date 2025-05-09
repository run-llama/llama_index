import os
import sys
import time

import openai
from openai import OpenAI
from validate_json import validate_json

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def launch_training(data_path: str) -> None:
    validate_json(data_path)

    # TODO: figure out how to specify file name in the new API
    # file_name = os.path.basename(data_path)

    # upload file
    with open(data_path, "rb") as f:
        output = client.files.create(
            file=f,
            purpose="fine-tune",
        )
    print("File uploaded...")

    # launch training
    while True:
        try:
            client.fine_tunes.create(training_file=output.id, model="gpt-3.5-turbo")
            break
        except openai.BadRequestError:
            print("Waiting for file to be ready...")
            time.sleep(60)
    print(f"Training job {output.id} launched. You will be emailed when it's complete.")


if __name__ == "__main__":
    data_path = sys.argv[1]
    if not os.path.exists(data_path):
        raise ValueError(f"Path {data_path} does not exist")
    launch_training(data_path)
