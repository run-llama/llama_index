import json
import sys


class Message:
    def __init__(
        self,
        message_id,
        message_text,
        author,
        timestamp,
        parent_message=None,
        child_message=None,
    ):
        self.message_id = message_id
        self.message_text = message_text
        self.author = author
        self.parent_message = parent_message
        self.child_message = child_message
        self.timestamp = timestamp

    def set_child(self, message):
        self.child_message = message

    def set_parent(self, message):
        self.parent_message = message


data_file = sys.argv[1]
with open(data_file, "r") as f:
    data = json.load(f)

messages = {}
for msg in data["messages"]:
    _id = msg["id"]
    text = msg["content"]
    msg_type = msg["type"]
    author = msg["author"]["name"]
    timestamp = msg["timestamp"]

    if msg_type in ("ThreadCreated", "ChannelPinnedMessage"):
        continue

    messages[_id] = Message(_id, text, author, timestamp)
    if msg_type == "Reply":
        parent_id = msg["reference"]["messageId"]
        try:
            messages[_id].set_parent(messages[parent_id])
        except:
            continue  # deleted message reference?
        messages[parent_id].set_child(messages[_id])

convo_docs = []
for msg in messages.values():
    # only check top-level messages
    if msg.parent_message is None:
        metadata = {"timestamp": msg.timestamp, "id": msg.message_id}
        convo = ""
        convo += msg.author + ":\n"
        convo += msg.message_text + "\n"

        cur_msg = msg
        is_thread = False
        while cur_msg.child_message is not None:
            is_thread = True
            cur_msg = cur_msg.child_message
            convo += cur_msg.author + ":\n"
            convo += cur_msg.message_text + "\n"

        if is_thread:
            convo_docs.append({"thread": convo, "metadata": metadata})

with open("conversation_docs.json", "w") as f:
    json.dump(convo_docs, f)

print("Done! Written to conversation_docs.json")
