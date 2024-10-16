import os
from llama_index.multi_modal_llms.reka import RekaMultiModalLLM
from llama_index.core.schema import ImageDocument
from llama_index.core.base.llms.types import ChatMessage, MessageRole

# Initialize the RekaMultiModalLLM
api_key = os.getenv("REKA_API_KEY")
if not api_key:
    raise ValueError("Please set the REKA_API_KEY environment variable.")

reka_mm_llm = RekaMultiModalLLM(api_key=api_key)

# Create an ImageDocument with the provided URL
image_doc_cat_img_from_url = ImageDocument(
    image_url="https://v0.docs.reka.ai/_images/000000245576.jpg"
)
# local folder and
current_folder = os.path.dirname(os.path.abspath(__file__))
image_doc_dog_img_local = ImageDocument(
    image_path=current_folder + "/abbie_the_corgi.jpeg"
)
# Create a chat message asking about the image
messages = [ChatMessage(role=MessageRole.USER, content="What do you see?")]

# Call the chat method with the image
response = reka_mm_llm.chat(
    messages=messages,
    image_documents=[image_doc_cat_img_from_url, image_doc_dog_img_local],
)

# Print the response
print("Reka's response:")
print(response.message.content)

# Ask a follow-up question
follow_up_message = ChatMessage(
    role=MessageRole.USER, content="What breed of pet is it? And what is it doing?"
)
messages.append(response.message)  # Add the assistant's previous response
messages.append(follow_up_message)

# Call the chat method again with the follow-up question
follow_up_response = reka_mm_llm.chat(
    messages=messages,
    image_documents=[image_doc_cat_img_from_url, image_doc_dog_img_local],
)

# Print the follow-up response
print("\nReka's follow-up response:")
print(follow_up_response.message.content)
