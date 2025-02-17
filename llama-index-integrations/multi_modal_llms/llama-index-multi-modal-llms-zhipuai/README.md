# LlamaIndex Multi-Modal-Llms Integration: ZhipuAI

### Installation

```bash
%pip install llama-index-multi-modal-llms-zhipuai
!pip install llama-index
```

### Basic usage

```py
# Import ZhipuAI
import os
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.multi_modal_llms.zhipuai import ZhipuAIMultiModal

# Set your environment variables
os.environ["ZHIPUAI_API_KEY"] = "Your API KEY"
os.environ["ZHIPUAI_TEST_VIDEO"] = "xxx.mp4"

# Call chat/complete with glm-4v-plus (types support: text, image_url, video_url)
llm = ZhipuAIMultiModal(
    model="glm-4v-plus", api_key=os.getenv("ZHIPUAI_API_KEY")
)
with open(os.getenv("ZHIPUAI_TEST_VIDEO"), "rb") as video_file:
    video_base = base64.b64encode(video_file.read()).decode("utf-8")
messages_1 = [
    ChatMessage(
        role=MessageRole.USER,
        content=[
            {"type": "video_url", "video_url": {"url": video_base}},
            {"type": "text", "text": "descript the video"},
        ],
    ),
]

print(llm.chat(messages_1))
print(llm.stream_chat(messages_1))
# Output:
# <class 'llama_index.core.base.llms.types.ChatResponse'>
# assistant: The camera shoots the sky, with white clouds moving from the right side of the screen to the left. Then, the clouds move upwards, and the camera pans to the right.
# <generator object llm_chat_callback.<locals>.wrap.<locals>.wrapped_llm_chat.<locals>.wrapped_gen at 0x127853350>

messages_2 = [
    ChatMessage(
        role=MessageRole.USER,
        content="descript the video",
    ),
]

print(llm.chat(messages_2, video_url=video_base))
print(llm.stream_chat(messages_2, video_url=video_base))
# Output:
# <class 'llama_index.core.base.llms.types.ChatResponse'>
# assistant: The camera shoots the sky, with white clouds moving from the right side of the screen to the left. Then, the clouds move upwards, and the camera pans to the right.
# <generator object llm_chat_callback.<locals>.wrap.<locals>.wrapped_llm_chat.<locals>.wrapped_gen at 0x135853350>

print(llm.complete("descript the video", video_url=video_base))
print(llm.stream_complete("descript the video", video_url=video_base))
# Output:
# <class 'llama_index.core.base.llms.types.ChatResponse'>
# The camera captures a downward shot of the sky, with white clouds drifting across the frame. The clouds are moving from the right side of the screen to the left, and the camera follows their movement, panning to the left.
# <generator object llm_completion_callback.<locals>.wrap.<locals>.wrapped_llm_predict.<locals>.wrapped_gen at 0x1175535b0>

# Call chat/complete with cogview or cogvideo
llm = ZhipuAIMultiModal(
    model="cogvideox", api_key=os.getenv("ZHIPUAI_API_KEY")
)
messages = [
    ChatMessage(
        role=MessageRole.USER,
        content=[{"type": "text", "text": "a bird flying in the sky"}],
    ),
]

print(llm.chat(messages))
print(llm.complete("a bird flying in the sky"))
# Output:
# <class 'llama_index.core.base.llms.types.ChatResponse'>
# assistant: https://aigc-files.bigmodel.cn/api/cogvideo/bcdaa436-xxxx-11ef-a354-628f45da38f5_0.mp4
# <class 'llama_index.core.base.llms.types.ChatResponse'>
# https://aigc-files.bigmodel.cn/api/cogvideo/ce55d168-xxxx-11ef-b2eb-72fab4c14186_0.mp4
```

### ZhipuAI Documentation

usage: https://bigmodel.cn/dev/howuse/introduction

api: https://bigmodel.cn/dev/api/normal-model/glm-4v
