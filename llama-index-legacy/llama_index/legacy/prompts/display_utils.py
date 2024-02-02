"""Prompt display utils."""

from llama_index.legacy.prompts.mixin import PromptDictType


# define prompt viewing function
def display_prompt_dict(prompts_dict: PromptDictType) -> None:
    """Display prompt dict.

    Args:
        prompts_dict: prompt dict

    """
    from IPython.display import Markdown, display

    for k, p in prompts_dict.items():
        text_md = f"**Prompt Key**: {k}<br>" f"**Text:** <br>"
        display(Markdown(text_md))
        print(p.get_template())
        display(Markdown("<br><br>"))
