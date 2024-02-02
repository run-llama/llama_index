import requests

url = "http://127.0.0.1:8080"
response = requests.post(
    url=url + "/access_token/",
    params={
        "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJjbGllbnRfaWQiOiIxIiwiY2xpZW50X3V1aWQiOiJiNzRjNTUyMC1jNTNkLTRhY2MtOTcxNi1hZDM1ODA2MWUyZGIiLCJ1c2VyX2lkIjoiMSIsInVzZXJfdXVpZCI6ImM4NzcyMWNlLWY0ZDgtNDdhNy1hNTBjLTQ5MmVlNTIwNmE5NSIsImV4cCI6MTcxNDE1MDMxNX0.2zvdapsnZnl5JYJxiTYd5mAsn3RqnYeObw4ofRRJClw"
    },
)
access_token = response.json()["access_token"]

headers = {
    "Authorization": f"Bearer {access_token}",
    "Content-Type": "application/json",
}


response = requests.post(
    url=url + "/chat/",
    headers=headers,
    params={"team_uuid": "9e921db1-ed80-46f1-a017-fb23ddc7741e"},
)

chat_uuid = response.json()["chat_uuid"]

response = requests.post(
    url=url + f"/chat/{chat_uuid}/message/",
    headers=headers,
    params={"prompt": "How many albums are there in the database?"},
    json={"url": "http://0.0.0.0:8000/webhook", "parameters": {}, "options": {}},
    stream=True,
)
for i in response.iter_content():
    print(i.decode("utf-8"))
