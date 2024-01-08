from flask import Flask, request, Response
from flask_cors import CORS
from llama_index import SimpleDirectoryReader, ServiceContext, VectorStoreIndex
import os
import time

index = None
chat_engine = None

def initialize_index():
    global index
    global chat_engine
    documents = SimpleDirectoryReader("./examples/ncs/faq_data/").load_data()
    index = VectorStoreIndex.from_documents(documents)
    chat_engine = index.as_chat_engine(chat_mode="context")

app = Flask(__name__)
CORS(app)

@app.route("/")
def home():
    return "Hello World!"

@app.route("/query", methods=["GET"])
def query_index():
  global chat_engine
  query_text = request.args.get("text", None)
  if query_text is None:
    return "No text found, please include a ?text=blah parameter in the URL", 400
  if not chat_engine:
    print("chat engine None")
    return Response('', content_type='text/plain')
  chat_response = chat_engine.chat(query_text)
  ans = str(chat_response)
  flask_response = Response(ans, content_type='text/plain')
  return flask_response

if __name__ == "__main__":
    initialize_index()
    app.run(host="localhost", port=5601)

