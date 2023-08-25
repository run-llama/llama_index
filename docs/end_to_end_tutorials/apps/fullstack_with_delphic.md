# A Guide to Building a Full-Stack LlamaIndex Web App with Delphic

This guide seeks to walk you through using LlamaIndex with a production-ready web app starter template
called [Delphic](https://github.com/JSv4/Delphic). All code examples here are available from
the [Delphic](https://github.com/JSv4/Delphic) repo

## What We're Building

Here's a quick demo of the out-of-the-box functionality of Delphic:

https://user-images.githubusercontent.com/5049984/233236432-aa4980b6-a510-42f3-887a-81485c9644e6.mp4

## Architectural Overview

Delphic leverages the LlamaIndex python library to let users to create their own document collections they can then
query in a responsive frontend.

We chose a stack that provides a responsive, robust mix of technologies that can (1) orchestrate complex python
processing tasks while providing (2) a modern, responsive frontend and (3) a secure backend to build additional
functionality upon.

The core libraries are:

1. [Django](https://www.djangoproject.com/)
2. [Django Channels](https://channels.readthedocs.io/en/stable/)
3. [Django Ninja](https://django-ninja.rest-framework.com/)
4. [Redis](https://redis.io/)
5. [Celery](https://docs.celeryq.dev/en/stable/getting-started/introduction.html)
6. [LlamaIndex](https://gpt-index.readthedocs.io/en/latest/)
7. [Langchain](https://python.langchain.com/en/latest/index.html)
8. [React](https://github.com/facebook/react)
9. Docker & Docker Compose

Thanks to this modern stack built on the super stable Django web framework, the starter Delphic app boasts a streamlined
developer experience, built-in authentication and user management, asynchronous vector store processing, and
web-socket-based query connections for a responsive UI. In addition, our frontend is built with TypeScript and is based
on MUI React for a responsive and modern user interface.

## System Requirements

Celery doesn't work on Windows. It may be deployable with Windows Subsystem for Linux, but configuring that is beyond
the scope of this tutorial. For this reason, we recommend you only follow this tutorial if you're running Linux or OSX.
You will need Docker and Docker Compose installed to deploy the application. Local development will require node version
manager (nvm).

## Django Backend

### Project Directory Overview

The Delphic application has a structured backend directory organization that follows common Django project conventions.
From the repo root, in the `./delphic` subfolder, the main folders are:

1. `contrib`: This directory contains custom modifications or additions to Django's built-in `contrib` apps.
2. `indexes`: This directory contains the core functionality related to document indexing and LLM integration. It
   includes:

- `admin.py`: Django admin configuration for the app
- `apps.py`: Application configuration
- `models.py`: Contains the app's database models
- `migrations`: Directory containing database schema migrations for the app
- `signals.py`: Defines any signals for the app
- `tests.py`: Unit tests for the app

3. `tasks`: This directory contains tasks for asynchronous processing using Celery. The `index_tasks.py` file includes
   the tasks for creating vector indexes.
4. `users`: This directory is dedicated to user management, including:
5. `utils`: This directory contains utility modules and functions that are used across the application, such as custom
   storage backends, path helpers, and collection-related utilities.

### Database Models

The Delphic application has two core models: `Document` and `Collection`. These models represent the central entities
the application deals with when indexing and querying documents using LLMs. They're defined in
[`./delphic/indexes/models.py`](https://github.com/JSv4/Delphic/blob/main/delphic/indexes/models.py).

1. `Collection`:

- `api_key`: A foreign key that links a collection to an API key. This helps associate jobs with the source API key.
- `title`: A character field that provides a title for the collection.
- `description`: A text field that provides a description of the collection.
- `status`: A character field that stores the processing status of the collection, utilizing the `CollectionStatus`
  enumeration.
- `created`: A datetime field that records when the collection was created.
- `modified`: A datetime field that records the last modification time of the collection.
- `model`: A file field that stores the model associated with the collection.
- `processing`: A boolean field that indicates if the collection is currently being processed.

2. `Document`:

- `collection`: A foreign key that links a document to a collection. This represents the relationship between documents
  and collections.
- `file`: A file field that stores the uploaded document file.
- `description`: A text field that provides a description of the document.
- `created`: A datetime field that records when the document was created.
- `modified`: A datetime field that records the last modification time of the document.

These models provide a solid foundation for collections of documents and the indexes created from them with LlamaIndex.

### Django Ninja API

Django Ninja is a web framework for building APIs with Django and Python 3.7+ type hints. It provides a simple,
intuitive, and expressive way of defining API endpoints, leveraging Python’s type hints to automatically generate input
validation, serialization, and documentation.

In the Delphic repo,
the [`./config/api/endpoints.py`](https://github.com/JSv4/Delphic/blob/main/config/api/endpoints.py)
file contains the API routes and logic for the API endpoints. Now, let’s briefly address the purpose of each endpoint
in the `endpoints.py` file:

1. `/heartbeat`: A simple GET endpoint to check if the API is up and running. Returns `True` if the API is accessible.
   This is helpful for Kubernetes setups that expect to be able to query your container to ensure it's up and running.

2. `/collections/create`: A POST endpoint to create a new `Collection`. Accepts form parameters such
   as `title`, `description`, and a list of `files`. Creates a new `Collection` and `Document` instances for each file,
   and schedules a Celery task to create an index.

```python
@collections_router.post("/create")
async def create_collection(request,
                            title: str = Form(...),
                            description: str = Form(...),
                            files: list[UploadedFile] = File(...), ):
    key = None if getattr(request, "auth", None) is None else request.auth
    if key is not None:
        key = await key

    collection_instance = Collection(
        api_key=key,
        title=title,
        description=description,
        status=CollectionStatusEnum.QUEUED,
    )

    await sync_to_async(collection_instance.save)()

    for uploaded_file in files:
        doc_data = uploaded_file.file.read()
        doc_file = ContentFile(doc_data, uploaded_file.name)
        document = Document(collection=collection_instance, file=doc_file)
        await sync_to_async(document.save)()

    create_index.si(collection_instance.id).apply_async()

    return await sync_to_async(CollectionModelSchema)(
        ...
    )
```

3. `/collections/query` — a POST endpoint to query a document collection using the LLM. Accepts a JSON payload
   containing `collection_id` and `query_str`, and returns a response generated by querying the collection. We don't
   actually use this endpoint in our chat GUI (We use a websocket - see below), but you could build an app to integrate
   to this REST endpoint to query a specific collection.

```python
@collections_router.post("/query",
                         response=CollectionQueryOutput,
                         summary="Ask a question of a document collection", )
def query_collection_view(request: HttpRequest, query_input: CollectionQueryInput):
    collection_id = query_input.collection_id
    query_str = query_input.query_str
    response = query_collection(collection_id, query_str)
    return {"response": response}
```

4. `/collections/available`: A GET endpoint that returns a list of all collections created with the user's API key. The
   output is serialized using the `CollectionModelSchema`.

```python
@collections_router.get("/available",
                        response=list[CollectionModelSchema],
                        summary="Get a list of all of the collections created with my api_key", )
async def get_my_collections_view(request: HttpRequest):
    key = None if getattr(request, "auth", None) is None else request.auth
    if key is not None:
        key = await key

    collections = Collection.objects.filter(api_key=key)

    return [
        {
            ...
        }
        async for collection in collections
    ]
```

5. `/collections/{collection_id}/add_file`: A POST endpoint to add a file to an existing collection. Accepts
   a `collection_id` path parameter, and form parameters such as `file` and `description`. Adds the file as a `Document`
   instance associated with the specified collection.

```python
@collections_router.post("/{collection_id}/add_file", summary="Add a file to a collection")
async def add_file_to_collection(request,
                                 collection_id: int,
                                 file: UploadedFile = File(...),
                                 description: str = Form(...), ):
    collection = await sync_to_async(Collection.objects.get)(id=collection_id
```

### Intro to Websockets

WebSockets are a communication protocol that enables bidirectional and full-duplex communication between a client and a
server over a single, long-lived connection. The WebSocket protocol is designed to work over the same ports as HTTP and
HTTPS (ports 80 and 443, respectively) and uses a similar handshake process to establish a connection. Once the
connection is established, data can be sent in both directions as “frames” without the need to reestablish the
connection each time, unlike traditional HTTP requests.

There are several reasons to use WebSockets, particularly when working with code that takes a long time to load into
memory but is quick to run once loaded:

1. **Performance**: WebSockets eliminate the overhead associated with opening and closing multiple connections for each
   request, reducing latency.
2. **Efficiency**: WebSockets allow for real-time communication without the need for polling, resulting in more
   efficient use of resources and better responsiveness.
3. **Scalability**: WebSockets can handle a large number of simultaneous connections, making it ideal for applications
   that require high concurrency.

In the case of the Delphic application, using WebSockets makes sense as the LLMs can be expensive to load into memory.
By establishing a WebSocket connection, the LLM can remain loaded in memory, allowing subsequent requests to be
processed quickly without the need to reload the model each time.

The ASGI configuration file [`./config/asgi.py`](https://github.com/JSv4/Delphic/blob/main/config/asgi.py) defines how
the application should handle incoming connections, using the Django Channels `ProtocolTypeRouter` to route connections
based on their protocol type. In this case, we have two protocol types: "http" and "websocket".

The “http” protocol type uses the standard Django ASGI application to handle HTTP requests, while the “websocket”
protocol type uses a custom `TokenAuthMiddleware` to authenticate WebSocket connections. The `URLRouter` within
the `TokenAuthMiddleware` defines a URL pattern for the `CollectionQueryConsumer`, which is responsible for handling
WebSocket connections related to querying document collections.

```python
application = ProtocolTypeRouter(
    {
        "http": get_asgi_application(),
        "websocket": TokenAuthMiddleware(
            URLRouter(
                [
                    re_path(
                        r"ws/collections/(?P<collection_id>\w+)/query/$",
                        CollectionQueryConsumer.as_asgi(),
                    ),
                ]
            )
        ),
    }
)
```

This configuration allows clients to establish WebSocket connections with the Delphic application to efficiently query
document collections using the LLMs, without the need to reload the models for each request.

### Websocket Handler

The `CollectionQueryConsumer` class
in [`config/api/websockets/queries.py`](https://github.com/JSv4/Delphic/blob/main/config/api/websockets/queries.py) is
responsible for handling WebSocket connections related to querying document collections. It inherits from
the `AsyncWebsocketConsumer` class provided by Django Channels.

The `CollectionQueryConsumer` class has three main methods:

1. `connect`: Called when a WebSocket is handshaking as part of the connection process.
2. `disconnect`: Called when a WebSocket closes for any reason.
3. `receive`: Called when the server receives a message from the WebSocket.

#### Websocket connect listener

The `connect` method is responsible for establishing the connection, extracting the collection ID from the connection
path, loading the collection model, and accepting the connection.

```python
async def connect(self):
    try:
        self.collection_id = extract_connection_id(self.scope["path"])
        self.index = await load_collection_model(self.collection_id)
        await self.accept()

except ValueError as e:
await self.accept()
await self.close(code=4000)
except Exception as e:
pass
```

#### Websocket disconnect listener

The `disconnect` method is empty in this case, as there are no additional actions to be taken when the WebSocket is
closed.

#### Websocket receive listener

The `receive` method is responsible for processing incoming messages from the WebSocket. It takes the incoming message,
decodes it, and then queries the loaded collection model using the provided query. The response is then formatted as a
markdown string and sent back to the client over the WebSocket connection.

```python
async def receive(self, text_data):
    text_data_json = json.loads(text_data)

    if self.index is not None:
        query_str = text_data_json["query"]
        modified_query_str = f"Please return a nicely formatted markdown string to this request:\n\n{query_str}"
        query_engine = self.index.as_query_engine()
        response = query_engine.query(modified_query_str)

        markdown_response = f"## Response\n\n{response}\n\n"
        if response.source_nodes:
            markdown_sources = f"## Sources\n\n{response.get_formatted_sources()}"
        else:
            markdown_sources = ""

        formatted_response = f"{markdown_response}{markdown_sources}"

        await self.send(json.dumps({"response": formatted_response}, indent=4))
    else:
        await self.send(json.dumps({"error": "No index loaded for this connection."}, indent=4))
```

To load the collection model, the `load_collection_model` function is used, which can be found
in [`delphic/utils/collections.py`](https://github.com/JSv4/Delphic/blob/main/delphic/utils/collections.py). This
function retrieves the collection object with the given collection ID, checks if a JSON file for the collection model
exists, and if not, creates one. Then, it sets up the `LLMPredictor` and `ServiceContext` before loading
the `VectorStoreIndex` using the cache file.

```python
async def load_collection_model(collection_id: str | int) -> VectorStoreIndex:
    """
    Load the Collection model from cache or the database, and return the index.

    Args:
        collection_id (Union[str, int]): The ID of the Collection model instance.

    Returns:
        VectorStoreIndex: The loaded index.

    This function performs the following steps:
    1. Retrieve the Collection object with the given collection_id.
    2. Check if a JSON file with the name '/cache/model_{collection_id}.json' exists.
    3. If the JSON file doesn't exist, load the JSON from the Collection.model FileField and save it to
       '/cache/model_{collection_id}.json'.
    4. Call VectorStoreIndex.load_from_disk with the cache_file_path.
    """
    # Retrieve the Collection object
    collection = await Collection.objects.aget(id=collection_id)
    logger.info(f"load_collection_model() - loaded collection {collection_id}")

    # Make sure there's a model
    if collection.model.name:
        logger.info("load_collection_model() - Setup local json index file")

        # Check if the JSON file exists
        cache_dir = Path(settings.BASE_DIR) / "cache"
        cache_file_path = cache_dir / f"model_{collection_id}.json"
        if not cache_file_path.exists():
            cache_dir.mkdir(parents=True, exist_ok=True)
            with collection.model.open("rb") as model_file:
                with cache_file_path.open("w+", encoding="utf-8") as cache_file:
                    cache_file.write(model_file.read().decode("utf-8"))

        # define LLM
        logger.info(
            f"load_collection_model() - Setup service context with tokens {settings.MAX_TOKENS} and "
            f"model {settings.MODEL_NAME}"
        )
        llm = OpenAI(temperature=0, model="text-davinci-003", max_tokens=512)
        service_context = ServiceContext.from_defaults(llm=llm)

        # Call VectorStoreIndex.load_from_disk
        logger.info("load_collection_model() - Load llama index")
        index = VectorStoreIndex.load_from_disk(
            cache_file_path, service_context=service_context
        )
        logger.info(
            "load_collection_model() - Llamaindex loaded and ready for query..."
        )

    else:
        logger.error(
            f"load_collection_model() - collection {collection_id} has no model!"
        )
        raise ValueError("No model exists for this collection!")

    return index
```

## React Frontend

### Overview

We chose to use TypeScript, React and Material-UI (MUI) for the Delphic project’s frontend for a couple reasons. First,
as the most popular component library (MUI) for the most popular frontend framework (React), this choice makes this
project accessible to a huge community of developers. Second, React is, at this point, a stable and generally well-liked
framework that delivers valuable abstractions in the form of its virtual DOM while still being relatively stable and, in
our opinion, pretty easy to learn, again making it accessible.

### Frontend Project Structure

The frontend can be found in the [`/frontend`](https://github.com/JSv4/Delphic/tree/main/frontend) directory of the
repo, with the React-related components being in `/frontend/src` . You’ll notice there is a DockerFile in the `frontend`
directory and several folders and files related to configuring our frontend web
server — [nginx](https://www.nginx.com/).

The `/frontend/src/App.tsx` file serves as the entry point of the application. It defines the main components, such as
the login form, the drawer layout, and the collection create modal. The main components are conditionally rendered based
on whether the user is logged in and has an authentication token.

The DrawerLayout2 component is defined in the`DrawerLayour2.tsx` file. This component manages the layout of the
application and provides the navigation and main content areas.

Since the application is relatively simple, we can get away with not using a complex state management solution like
Redux and just use React’s useState hooks.

### Grabbing Collections from the Backend

The collections available to the logged-in user are retrieved and displayed in the DrawerLayout2 component. The process
can be broken down into the following steps:

1. Initializing state variables:

```tsx
const[collections, setCollections] = useState < CollectionModelSchema[] > ([]);
const[loading, setLoading] = useState(true);
```

Here, we initialize two state variables: `collections` to store the list of collections and `loading` to track whether
the collections are being fetched.

2. Collections are fetched for the logged-in user with the `fetchCollections()` function:

```tsx
const
fetchCollections = async () = > {
try {
const accessToken = localStorage.getItem("accessToken");
if (accessToken) {
const response = await getMyCollections(accessToken);
setCollections(response.data);
}
} catch (error) {
console.error(error);
} finally {
setLoading(false);
}
};
```

The `fetchCollections` function retrieves the collections for the logged-in user by calling the `getMyCollections` API
function with the user's access token. It then updates the `collections` state with the retrieved data and sets
the `loading` state to `false` to indicate that fetching is complete.

### Displaying Collections

The latest collectios are displayed in the drawer like this:

```tsx
< List >
{collections.map((collection) = > (
    < div key={collection.id} >
    < ListItem disablePadding >
    < ListItemButton
    disabled={
    collection.status != = CollectionStatus.COMPLETE | |
    !collection.has_model
    }
    onClick={() = > handleCollectionClick(collection)}
selected = {
    selectedCollection & &
    selectedCollection.id == = collection.id
}
>
< ListItemText
primary = {collection.title} / >
          {collection.status == = CollectionStatus.RUNNING ? (
    < CircularProgress
    size={24}
    style={{position: "absolute", right: 16}}
    / >
): null}
< / ListItemButton >
    < / ListItem >
        < / div >
))}
< / List >
```

You’ll notice that the `disabled` property of a collection’s `ListItemButton` is set based on whether the collection's
status is not `CollectionStatus.COMPLETE` or the collection does not have a model (`!collection.has_model`). If either
of these conditions is true, the button is disabled, preventing users from selecting an incomplete or model-less
collection. Where the CollectionStatus is RUNNING, we also show a loading wheel over the button.

In a separate `useEffect` hook, we check if any collection in the `collections` state has a status
of `CollectionStatus.RUNNING` or `CollectionStatus.QUEUED`. If so, we set up an interval to repeatedly call
the `fetchCollections` function every 15 seconds (15,000 milliseconds) to update the collection statuses. This way, the
application periodically checks for completed collections, and the UI is updated accordingly when the processing is
done.

```tsx
useEffect(() = > {
    let
interval: NodeJS.Timeout;
if (
    collections.some(
        (collection) = >
collection.status == = CollectionStatus.RUNNING | |
collection.status == = CollectionStatus.QUEUED
)
) {
    interval = setInterval(() = > {
    fetchCollections();
}, 15000);
}
return () = > clearInterval(interval);
}, [collections]);
```

### Chat View Component

The `ChatView` component in `frontend/src/chat/ChatView.tsx` is responsible for handling and displaying a chat interface
for a user to interact with a collection. The component establishes a WebSocket connection to communicate in real-time
with the server, sending and receiving messages.

Key features of the `ChatView` component include:

1. Establishing and managing the WebSocket connection with the server.
2. Displaying messages from the user and the server in a chat-like format.
3. Handling user input to send messages to the server.
4. Updating the messages state and UI based on received messages from the server.
5. Displaying connection status and errors, such as loading messages, connecting to the server, or encountering errors
   while loading a collection.

Together, all of this allows users to interact with their selected collection with a very smooth, low-latency
experience.

#### Chat Websocket Client

The WebSocket connection in the `ChatView` component is used to establish real-time communication between the client and
the server. The WebSocket connection is set up and managed in the `ChatView` component as follows:

First, we want to initialize the the WebSocket reference:

const websocket = useRef<WebSocket | null>(null);

A `websocket` reference is created using `useRef`, which holds the WebSocket object that will be used for
communication. `useRef` is a hook in React that allows you to create a mutable reference object that persists across
renders. It is particularly useful when you need to hold a reference to a mutable object, such as a WebSocket
connection, without causing unnecessary re-renders.

In the `ChatView` component, the WebSocket connection needs to be established and maintained throughout the lifetime of
the component, and it should not trigger a re-render when the connection state changes. By using `useRef`, you ensure
that the WebSocket connection is kept as a reference, and the component only re-renders when there are actual state
changes, such as updating messages or displaying errors.

The `setupWebsocket` function is responsible for establishing the WebSocket connection and setting up event handlers to
handle different WebSocket events.

Overall, the setupWebsocket function looks like this:

```tsx
const setupWebsocket = () => {  
  setConnecting(true);  
  // Here, a new WebSocket object is created using the specified URL, which includes the   
  // selected collection's ID and the user's authentication token.  
    
  websocket.current = new WebSocket(  
    `ws://localhost:8000/ws/collections/${selectedCollection.id}/query/?token=${authToken}`  
  );  
  
  websocket.current.onopen = (event) => {  
    //...  
  };  
  
  websocket.current.onmessage = (event) => {  
    //...  
  };  
  
  websocket.current.onclose = (event) => {  
    //...  
  };  
  
  websocket.current.onerror = (event) => {  
    //...  
  };  
  
  return () => {  
    websocket.current?.close();  
  };  
};
```

Notice in a bunch of places we trigger updates to the GUI based on the information from the web socket client.

When the component first opens and we try to establish a connection, the `onopen` listener is triggered. In the
callback, the component updates the states to reflect that the connection is established, any previous errors are
cleared, and no messages are awaiting responses:

```tsx
websocket.current.onopen = (event) => {  
  setError(false);  
  setConnecting(false);  
  setAwaitingMessage(false);  
  
  console.log("WebSocket connected:", event);  
};
```

`onmessage`is triggered when a new message is received from the server through the WebSocket connection. In the
callback, the received data is parsed and the `messages` state is updated with the new message from the server:

```
websocket.current.onmessage = (event) => {  
  const data = JSON.parse(event.data);  
  console.log("WebSocket message received:", data);  
  setAwaitingMessage(false);  
  
  if (data.response) {  
    // Update the messages state with the new message from the server  
    setMessages((prevMessages) => [  
      ...prevMessages,  
      {  
        sender_id: "server",  
        message: data.response,  
        timestamp: new Date().toLocaleTimeString(),  
      },  
    ]);  
  }  
};
```

`onclose`is triggered when the WebSocket connection is closed. In the callback, the component checks for a specific
close code (`4000`) to display a warning toast and update the component states accordingly. It also logs the close
event:

```tsx
websocket.current.onclose = (event) => {  
  if (event.code === 4000) {  
    toast.warning(  
      "Selected collection's model is unavailable. Was it created properly?"  
    );  
    setError(true);  
    setConnecting(false);  
    setAwaitingMessage(false);  
  }  
  console.log("WebSocket closed:", event);  
};
```

Finally, `onerror` is triggered when an error occurs with the WebSocket connection. In the callback, the component
updates the states to reflect the error and logs the error event:

```tsx
    websocket.current.onerror = (event) => {
      setError(true);
      setConnecting(false);
      setAwaitingMessage(false);

      console.error("WebSocket error:", event);
    };
  ```

#### Rendering our Chat Messages

In the `ChatView` component, the layout is determined using CSS styling and Material-UI components. The main layout
consists of a container with a `flex` display and a column-oriented `flexDirection`. This ensures that the content
within the container is arranged vertically.

There are three primary sections within the layout:

1. The chat messages area: This section takes up most of the available space and displays a list of messages exchanged
   between the user and the server. It has an overflow-y set to ‘auto’, which allows scrolling when the content
   overflows the available space. The messages are rendered using the `ChatMessage` component for each message and
   a `ChatMessageLoading` component to show the loading state while waiting for a server response.
2. The divider: A Material-UI `Divider` component is used to separate the chat messages area from the input area,
   creating a clear visual distinction between the two sections.
3. The input area: This section is located at the bottom and allows the user to type and send messages. It contains
   a `TextField` component from Material-UI, which is set to accept multiline input with a maximum of 2 rows. The input
   area also includes a `Button` component to send the message. The user can either click the "Send" button or press "
   Enter" on their keyboard to send the message.

The user inputs accepted in the `ChatView` component are text messages that the user types in the `TextField`. The
component processes these text inputs and sends them to the server through the WebSocket connection.

## Deployment

### Prerequisites

To deploy the app, you're going to need Docker and Docker Compose installed. If you're on Ubuntu or another, common
Linux distribution, DigitalOcean has
a [great Docker tutorial](https://www.digitalocean.com/community/tutorial_collections/how-to-install-and-use-docker) and
another great tutorial
for [Docker Compose](https://www.digitalocean.com/community/tutorials/how-to-install-and-use-docker-compose-on-ubuntu-20-04)
you can follow. If those don't work for you, try
the [official docker documentation.](https://docs.docker.com/engine/install/)

### Build and Deploy

The project is based on django-cookiecutter, and it’s pretty easy to get it deployed on a VM and configured to serve
HTTPs traffic for a specific domain. The configuration is somewhat involved, however — not because of this project, but
it’s just a fairly involved topic to configure your certificates, DNS, etc.

For the purposes of this guide, let’s just get running locally. Perhaps we’ll release a guide on production deployment.
In the meantime, check out
the [Django Cookiecutter project docs](https://cookiecutter-django.readthedocs.io/en/latest/deployment-with-docker.html)
for starters.

This guide assumes your goal is to get the application up and running for use. If you want to develop, most likely you
won’t want to launch the compose stack with the — profiles fullstack flag and will instead want to launch the react
frontend using the node development server.

To deploy, first clone the repo:

```commandline
git clone https://github.com/yourusername/delphic.git
```

Change into the project directory:

```commandline
cd delphic
```

Copy the sample environment files:

```commandline
mkdir -p ./.envs/.local/  
cp -a ./docs/sample_envs/local/.frontend ./frontend  
cp -a ./docs/sample_envs/local/.django ./.envs/.local  
cp -a ./docs/sample_envs/local/.postgres ./.envs/.local
```

Edit the `.django` and `.postgres` configuration files to include your OpenAI API key and set a unique password for your
database user. You can also set the response token limit in the .django file or switch which OpenAI model you want to
use. GPT4 is supported, assuming you’re authorized to access it.

Build the docker compose stack with the `--profiles fullstack` flag:

```commandline
sudo docker-compose --profiles fullstack -f local.yml build
```

The fullstack flag instructs compose to build a docker container from the frontend folder and this will be launched
along with all of the needed, backend containers. It takes a long time to build a production React container, however,
so we don’t recommend you develop this way. Follow
the [instructions in the project readme.md](https://github.com/JSv4/Delphic#development) for development environment
setup instructions.

Finally, bring up the application:

```commandline
sudo docker-compose -f local.yml up
```

Now, visit `localhost:3000` in your browser to see the frontend, and use the Delphic application locally.

## Using the Application

### Setup Users

In order to actually use the application (at the moment, we intend to make it possible to share certain models with
unauthenticated users), you need a login. You can use either a superuser or non-superuser. In either case, someone needs
to first create a superuser using the console:

**Why set up a Django superuser?** A Django superuser has all the permissions in the application and can manage all
aspects of the system, including creating, modifying, and deleting users, collections, and other data. Setting up a
superuser allows you to fully control and manage the application.

**How to create a Django superuser:**

1 Run the following command to create a superuser:

sudo docker-compose -f local.yml run django python manage.py createsuperuser

2 You will be prompted to provide a username, email address, and password for the superuser. Enter the required
information.

**How to create additional users using Django admin:**

1. Start your Delphic application locally following the deployment instructions.
2. Visit the Django admin interface by navigating to `http://localhost:8000/admin` in your browser.
3. Log in with the superuser credentials you created earlier.
4. Click on “Users” under the “Authentication and Authorization” section.
5. Click on the “Add user +” button in the top right corner.
6. Enter the required information for the new user, such as username and password. Click “Save” to create the user.
7. To grant the new user additional permissions or make them a superuser, click on their username in the user list,
   scroll down to the “Permissions” section, and configure their permissions accordingly. Save your changes.
