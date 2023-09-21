import os
import openai
import time
import sys
from validate_json import validate_json

openai.api_key = os.getenv("OPENAI_API_KEY")


def launch_training(data_path: str) -> None:
    validate_json(data_path)

    file_name = os.path.basename(data_path)

    # upload file
    with open(data_path, "rb") as f:
        output = openai.File.create(
            file=f,
            purpose="fine-tune",
            user_provided_filename=file_name,
        )
    print("File uploaded...")

    # launch training
    while True:
        try:
            openai.FineTuningJob.create(
                training_file=output["id"], model="gpt-3.5-turbo"
            )
            break
        except openai.error.InvalidRequestError:
            print("Waiting for file to be ready...")
            time.sleep(60)
    print(
        f"Training job {output['id']} launched. You will be emailed when it's complete."
    )


if __name__ == "__main__":
    data_path = sys.argv[1]
    if not os.path.exists(data_path):
        raise ValueError(f"Path {data_path} does not exist")
    launch_training(data_path)
