# LlamaIndex Chat_Store Integration: DynamoDB Chat Store

This enables AWS DynamoDB to be used as a chat store.

## Installation

```bash
pip install llama-index-storage-chat-store-dynamodb
```

## Usage

### Assumptions

- `SessionID`, a string, is used as the partition key.
- The table used for the chat store already exists. Here is an example for that:

  ```python
  import boto3

  dynamodb = boto3.resource("dynamodb")

  # Create the DynamoDB table.
  table = dynamodb.create_table(
      TableName="EXAMPLE_TABLE",
      KeySchema=[{"AttributeName": "SessionId", "KeyType": "HASH"}],
      AttributeDefinitions=[
          {"AttributeName": "SessionId", "AttributeType": "S"}
      ],
      BillingMode="PAY_PER_REQUEST",
  )
  ```

### Using an AWS IAM Role

You can use any of the following AWS arguments to setup the required `boto3` resource connection:

- `profile_name`
- `aws_access_key_id`
- `aws_secret_access_key`
- `aws_session_token`
- `botocore_session` - A pre-existing existing Botocore session.
- `botocore_config`

As an example, if you have already assumed an AWS profile in your local environment or within an AWS compute
environment, you can simply do the following:

```python
import os
from llama_index.storage.chat_store.dynamodb.base import DynamoDBChatStore

store = DynamoDBChatStore(
    profile_name=os.getenv("AWS_PROFILE"),
    table_name="EXAMPLE_TABLE",
    session_id="123",
)
```
