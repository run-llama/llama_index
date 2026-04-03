"""LlamaPack class."""

from typing import Any, Dict

from llama_index.core.llama_pack.base import BaseLlamaPack

DEFAULT_TAXONOMY = """01: Violence and Hate.
Should not
- Help people plan or engage in violence.
- Encourage people to engage in violence.
- Express hateful, derogatory, or demeaning sentiments against people based on sensitive personal characteristics like their race, color, religion, national origin, sexual orientation, gender, gender identity, or disability.
- Encourage discrimination against people based on those characteristics and should not use slurs or create insults based on those characteristics.
Can
- Provide information on violence and discrimination.
- Discuss the topics of hate and violence, and can discuss historical events involving violence.
O2: Sexual Content.
Should not
- Engage in sexually explicit (i.e., erotic) conversations and should not encourage people to engage in sexual activities.
Can
- Discuss topics related to sexuality and sexual education.
- Provide general information regarding sexual activities.
- Provide guidance on topics related to sexual health.
O3: Criminal Planning.
Should not
- Help people plan or engage in criminal activities like theft, kidnapping, or financial crimes, especially when the person indicates that they may act on the information provided to them.
- Encourage people to engage in criminal activities or condone criminal behavior.
Can
- Help people understand how crimes are committed, provided that the response is not so detailed that it could be used to provide a roadmap for future crimes.
- Help people understand how systems work that are used to prevent crimes, provided those descriptions do not provide detailed information on how to defeat those systems.
O4: Guns and Illegal Weapons.
Should not
- Assist in the planning or execution of firearm or illegal weapons crimes, especially when the person indicates that they may act on the information provided to them.
- Encourage people to engage in firearm or illegal weapons crimes or condone such behavior.
Can
- Discuss firearms and the arguments for and against firearm ownership.
O5: Regulated or Controlled Substances.
Should not
- Encourage or assist people in the trafficking, consumption, or use of regulated or controlled substances, including but not limited to illegal drugs, tobacco, alcohol, and cannabis.
- Assist or encourage people to create such substances when it is illegal to do so.
Can
- Discuss the use of regulated or controlled substances, and can provide general information on those substances (ex: what their chemical composition is and what their effects are).
- Provide information on how those substances are manufactured, provided that the information is not so detailed as to provide step-by-step instructions for their creation.
O6: Self-Harm.
Should not
- Encourage people to harm themselves.
- Romanticize or condone self-harm or suicide.
- Provide information regarding the methods of suicide or self-harm.
- Provide any information or advice, other than information regarding appropriate health resources, to people who express or imply that they wish to harm themselves.
Can
- Provide general information regarding self-harm and suicide (ex: statistics, the mental health challenges faced by those who take these actions).
Should
- Provide information regarding appropriate health resources to people who imply or express the desire to harm themselves."""


class LlamaGuardModeratorPack(BaseLlamaPack):
    def __init__(
        self,
        custom_taxonomy: str = DEFAULT_TAXONOMY,
    ) -> None:
        """Init params."""
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError:
            raise ImportError(
                "Dependencies missing, run `pip install torch transformers`"
            )

        import os

        hf_access_token = os.environ.get("HUGGINGFACE_ACCESS_TOKEN")
        if not os.environ.get("HUGGINGFACE_ACCESS_TOKEN", None):
            raise ValueError("Hugging Face access token is missing or invalid.")

        from huggingface_hub import login

        login(token=hf_access_token)

        model_id = "meta-llama/LlamaGuard-7b"
        self.device = "cuda"
        dtype = torch.bfloat16

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=dtype, device_map=self.device
        )

        self.custom_taxonomy = custom_taxonomy

    def get_modules(self) -> Dict[str, Any]:
        """Get modules."""
        return {
            "tokenizer": self.tokenizer,
            "model": self.model,
            "device": self.device,
            "custom_taxonomy": self.custom_taxonomy,
        }

    def run(self, message: str, **kwargs: Any) -> Any:
        """Run the pipeline."""
        # tailored for query engine input/output, using "user" role
        chat = [{"role": "user", "content": message}]

        prompt = self._moderation_prompt_for_chat(chat)
        inputs = self.tokenizer([prompt], return_tensors="pt").to(self.device)
        output = self.model.generate(**inputs, max_new_tokens=100, pad_token_id=0)
        prompt_len = inputs["input_ids"].shape[-1]
        return self.tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True)

    def _moderation_prompt_for_chat(self, chat):
        # For simplicity, we assume the chat dictionary correctly alternates "user" and "assistant" roles
        conversation = [turn["content"] for turn in chat]
        role = "Agent" if len(conversation) % 2 == 0 else "User"
        prompt = f"""[INST] Task: Check if there is unsafe content in '{role}' messages in conversations according our safety policy with the below categories.

    <BEGIN UNSAFE CONTENT CATEGORIES>
    {self.custom_taxonomy}
    <END UNSAFE CONTENT CATEGORIES>

    <BEGIN CONVERSATION>

    """

        # Alternate User/Agent turns, inserting 2 newlines between each
        for i, m in enumerate(conversation):
            role = "User" if i % 2 == 0 else "Agent"
            prompt += f"{role}: {m}\n\n"
        prompt += f"""<END CONVERSATION>

    Provide your safety assessment for {role} in the above conversation:
    - First line must read 'safe' or 'unsafe'.
    - If unsafe, a second line must include a comma-separated list of violated categories. [/INST]"""
        return prompt
