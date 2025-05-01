import random
import requests


from basejump.models.schemas import BasejumpDBConn, BasejumpDBConnProd
from basejump.config import globalconfig
from basejump.config.globalconfig import API_SECRET_KEY

if globalconfig.TEST:
    db_params = BasejumpDBConn().dict()
elif not globalconfig.TEST:
    db_params = BasejumpDBConnProd().dict()

base_url = "http://127.0.0.1:8080"


def helper_create_client(client_name="pytest_client"):
    if client_name == "pytest_client":
        client_name += str(round(random.random() * 10e5))
    else:
        client_name = client_name
    response = requests.post(
        url=base_url + "/client/",
        params={
            "client_name": client_name,
        },
        headers={"api-secret": API_SECRET_KEY},
    )
    print(response.status_code)
    print(response.json())
    assert response.status_code == 200

    response_dict = response.json()
    client_uuid = response_dict["client_uuid"]
    client_secret = response_dict["client_secret"]

    return client_uuid, client_secret


def helper_create_team(client_uuid: str, client_secret: str):
    response = requests.post(
        url=base_url + "/team/",
        params={"team_name": "pytest_team", "client_uuid": client_uuid},
        headers={"client-secret": client_secret},
    )

    assert response.status_code == 200

    response_dict = response.json()
    return response_dict["team_uuid"]


def helper_create_user(team_uuid, client_uuid, client_secret):
    response = requests.post(
        url=base_url + "/user/",
        params={
            "username": "pytest_user",
            "client_uuid": client_uuid,
            "team_uuid": team_uuid,
        },
        headers={"client-secret": client_secret},
    )
    assert response.status_code == 200

    response_dict = response.json()
    return response_dict["user_uuid"]


def helper_create_new_test_env(
    client_uuid: str,
    user_uuid: str,
    client_secret: str,
):
    # Get the refresh token
    response = requests.post(
        url=base_url + "/token/",
        params={
            "client_uuid": client_uuid,
            "user_uuid": user_uuid,
        },
        headers={"client-secret": client_secret},
    )
    new_refresh_token = response.json()["refresh_token"]
    # Get the access token
    response = requests.post(
        url=base_url + "/access_token/",
        params={"refresh_token": new_refresh_token, "client_uuid": client_uuid},
        headers={"client-secret": client_secret},
    )
    access_token = response.json()["access_token"]

    # Create a new env
    return {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
    }


if __name__ == "__main__":
    client_uuid, client_secret = helper_create_client()
    team_uuid = helper_create_team(client_uuid=client_uuid, client_secret=client_secret)
    user_uuid = helper_create_user(
        client_uuid=client_uuid, team_uuid=team_uuid, client_secret=client_secret
    )
    headers = helper_create_new_test_env(
        client_uuid=client_uuid,
        user_uuid=user_uuid,
        client_secret=client_secret,
    )
    updated_json = {"db_params": db_params, "schemas": {"schema_list": ["account"]}}
    response = requests.post(
        url=base_url + "/connection/database/",
        json=updated_json,
        headers=headers,
        params={"data_source_desc": "pytest_database"},
    )

    db_conn_uuid = response.json()["conn_uuid"]

    response = requests.post(
        url=base_url + f"/connection/{db_conn_uuid}/team/{team_uuid}/", headers=headers
    )

    response = requests.post(url=base_url + f"/chat/team/{team_uuid}/", headers=headers)
    chat_uuid = response.json()["chat_uuid"]

    # Ask the AI a question
    response = requests.post(
        url=base_url + f"/chat/{chat_uuid}/message/",
        headers=headers,
        # params={"prompt": "How many albums are there in the database?"},
        params={
            "prompt": "How many teams are there in the database?",
            "webhook_url": "http://0.0.0.0:8000/webhook/",
            "stream": True,
        },
    )

    msg_uuid = response.headers["msg_uuid"]
    if response.headers["report"]:
        report = requests.get(
            url=base_url + f"/chat/{chat_uuid}/message/{msg_uuid}/report/",
            headers=headers,
            stream=True,
        )
        print(report.text)


# @app.post("/webhook")
# def receive_webhook(text: str):
#     print(text)
