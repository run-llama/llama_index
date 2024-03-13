from typing import Optional

import requests


def get_pooling_mode(model_name: Optional[str]) -> str:
    pooling_config_url = (
        f"https://huggingface.co/{model_name}/raw/main/1_Pooling/config.json"
    )

    try:
        response = requests.get(pooling_config_url)
        config_data = response.json()

        cls_token = config_data.get("pooling_mode_cls_token", False)
        mean_tokens = config_data.get("pooling_mode_mean_tokens", False)

        if mean_tokens:
            return "mean"
        elif cls_token:
            return "cls"
    except requests.exceptions.RequestException:
        print(
            "Warning: Pooling config file not found; pooling mode is defaulted to 'cls'."
        )
    return "cls"
