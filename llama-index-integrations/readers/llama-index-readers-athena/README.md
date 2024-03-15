# Athena reader.

```bash
pip install llama-index-readers-athena

pip install llama-index-llms-openai
```

Athena reader allow execute SQL with AWS Athena. We using SQLAlchemy and PyAthena under the hood.

## Permissions

WE HIGHLY RECOMMEND USING THIS LOADER WITH AWS EC2 IAM ROLE.

## Usage

Here's an example usage of the AthenaReader.

```
import os
import dotenv
from llama_index.core import SQLDatabase,ServiceContext
from llama_index.core.query_engine import NLSQLTableQueryEngine
from llama_index.llms.openai import OpenAI
from llama_index.readers.athena import AthenaReader

dotenv.load_dotenv()

AWS_REGION = os.environ['AWS_REGION']
S3_STAGING_DIR = os.environ['S3_STAGING_DIR']
DATABASE = os.environ['DATABASE']
WORKGROUP = os.environ['WORKGROUP']
TABLE = os.environ['TABLE']

llm = OpenAI(model="gpt-4",temperature=0, max_tokens=1024)

engine = AthenaReader.create_athena_engine(
    aws_region=AWS_REGION,
    s3_staging_dir=S3_STAGING_DIR,
    database=DATABASE,
    workgroup=WORKGROUP
)

service_context = ServiceContext.from_defaults(
  llm=llm
)

sql_database = SQLDatabase(engine, include_tables=[TABLE])

query_engine = NLSQLTableQueryEngine(
    sql_database=sql_database,
    tables=[TABLE],
    service_context=service_context
)
query_str = (
    "Which blocknumber has the most transactions?"
)
response = query_engine.query(query_str)
```

## Screeshot

![image](https://vultureprime-research-center.s3.ap-southeast-1.amazonaws.com/Screenshot+2566-10-07+at+17.58.45.png)
