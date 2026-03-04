# LlamaIndex Vector_Stores Integration: MongoDB

## Setting up MongoDB Atlas as the Datastore Provider

MongoDB Atlas is a multi-cloud database service made by the same people that build MongoDB.
Atlas simplifies deploying and managing your databases while offering the versatility you need
to build resilient and performant global applications on the cloud providers of your choice.

You can perform semantic search on data in your Atlas cluster running MongoDB v6.0.11, v7.0.2,
or later using Atlas Vector Search. You can store vector embeddings for any kind of data along
with other data in your collection on the Atlas cluster.

In the section, we provide detailed instructions to run the tests.

### Deploy a Cluster

Follow the [Getting-Started](https://www.mongodb.com/basics/mongodb-atlas-tutorial) documentation
to create an account, deploy an Atlas cluster, and connect to a database.

### Retrieve the URI used by Python to connect to the Cluster

Once deployed, you will need a URI (connection string) to access the cluster.
This you should store as the environment variable: `MONGODB_URI`.
It will look something like the following. The username and password, if not provided,
can be configured in _Database Access_ under Security in the left panel.

```
export MONGODB_URI="mongodb+srv://<username>:<password>@cluster0.foo.mongodb.net/?retryWrites=true&w=majority"
```

Head to Atlas UI to find the connection string.

NOTE: There are a number of ways to navigate the Atlas UI. Keep your eye out for "Connect" and "driver".

On the left panel, find and click 'Database' under DEPLOYMENT.
Click the Connect button that appears, then Drivers. Select Python.
(Have no concern for the version. This is the PyMongo, not Python, version.)
Once you have the Connect Window open, you will see an instruction to `pip install pymongo`.
You will also see a **connection string**.
This is the `uri` that a `pymongo.MongoClient` uses to connect to the Database.

### Test the connection

Atlas provides a simple check. Once you have your `uri` and `pymongo` installed,
try the following in a python console.

```python
from pymongo.mongo_client import MongoClient

client = MongoClient(uri)  # Create a new client and connect to the server
try:
    client.admin.command(
        "ping"
    )  # Send a ping to confirm a successful connection
    print("Pinged your deployment. You successfully connected to MongoDB!")
except Exception as e:
    print(e)
```

**Troubleshooting**

- You can edit a Database's users and passwords on the 'Database Access' page, under Security.
- Remember to add your IP address. (Try `curl -4 ifconfig.co`)

### Create a Database and Collection

As mentioned, Vector Databases provide two functions. In addition to being the data store,
they provide very efficient search based on natural language queries.
With Vector Search, one will index and query data with a powerful vector search algorithm
using "Hierarchical Navigable Small World (HNSW) graphs to find vector similarity.

The indexing runs beside the data as a separate service asynchronously.
The Search index monitors changes to the Collection that it applies to.
Subsequently, one need not upload the data first.
We will create an empty collection now, which will simplify setup in the example notebook.

Back in the UI, navigate to the Database Deployments page by clicking Database on the left panel.
Click the "Browse Collections" and then "+ Create Database" buttons.
This will open a window where you choose Database and Collection names. (No additional preferences.)
Remember these values as they will be as the environment variables,
`MONGODB_DATABASE` and `MONGODB_COLLECTION`.

### Set Datastore Environment Variables

To establish a connection to the MongoDB Cluster, Database, and Collection, plus create a Vector Search Index,
define the following environment variables.
You can confirm that the required ones have been set like this: `assert "MONGODB_URI" in os.environ`

**IMPORTANT** It is crucial that the choices are consistent between setup in Atlas and Python environment(s).

| Name                 | Description       | Example                                                             |
| -------------------- | ----------------- | ------------------------------------------------------------------- |
| `MONGODB_URI`        | Connection String | mongodb+srv://`<user>`:`<password>`@llama-index.zeatahb.mongodb.net |
| `MONGODB_DATABASE`   | Database name     | llama_index_test_db                                                 |
| `MONGODB_COLLECTION` | Collection name   | llama_index_test_vectorstore                                        |
| `MONGODB_INDEX`      | Search index name | vector_index                                                        |

The following will be required to authenticate with OpenAI.

| Name             | Description                                                  |
| ---------------- | ------------------------------------------------------------ |
| `OPENAI_API_KEY` | OpenAI token created at https://platform.openai.com/api-keys |

### Create an Atlas Vector Search Index

The final step to configure MongoDB as the Datastore is to create a Vector Search Index.
The procedure is described [here](https://www.mongodb.com/docs/atlas/atlas-vector-search/create-index/#procedure).

Under Services on the left panel, choose Atlas Search > Create Search Index >
Atlas Vector Search JSON Editor.

The Plugin expects an index definition like the following.
To begin, choose `numDimensions: 1536` along with the suggested EMBEDDING variables above.
You can experiment with these later.

```json
{
  "fields": [
    {
      "numDimensions": 1536,
      "path": "embedding",
      "similarity": "cosine",
      "type": "vector"
    }
  ]
}
```

### Running MongoDB Integration Tests

In addition to the [Jupyter Notebook](https://docs.llamaindex.ai/en/stable/examples/vector_stores/MongoDBAtlasVectorSearch/) in the documentation,
a suite of integration tests is available to verify the MongoDB integration unders `./tests`.
This test suite needs the cluster up and running, and the environment variables defined above.
