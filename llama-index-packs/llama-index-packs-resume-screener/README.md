# Resumer Screener Pack

This LlamaPack loads a resume file, and review it against a user specified job description and screening criteria.

## CLI Usage

You can download llamapacks directly using `llamaindex-cli`, which comes installed with the `llama-index` python package:

```bash
llamaindex-cli download-llamapack ResumeScreenerPack --download-dir ./resume_screener_pack
```

You can then inspect the files at `./resume_screener_pack` and use them as a template for your own project!

## Code Usage

You can download the pack to a `./resume_screener_pack` directory:

```python
from llama_index.core.llama_pack import download_llama_pack

# download and install dependencies
ResumeScreenerPack = download_llama_pack(
    "ResumeScreenerPack", "./resume_screener_pack"
)
```

From here, you can use the pack, or inspect and modify the pack in `./resume_screener_pack`.

Then, you can set up the pack like so:

```python
# create the pack
resume_screener = ResumeScreenerPack(
    job_description="<general job description>",
    criteria=["<job criterion>", "<another job criterion>"],
)
```

```python
response = resume_screener.run(resume_path="resume.pdf")
print(response.overall_decision)
```

The `response` will be a pydantic model with the following schema

```python
class CriteriaDecision(BaseModel):
    """The decision made based on a single criteria"""

    decision: Field(
        type=bool, description="The decision made based on the criteria"
    )
    reasoning: Field(type=str, description="The reasoning behind the decision")


class ResumeScreenerDecision(BaseModel):
    """The decision made by the resume screener"""

    criteria_decisions: Field(
        type=List[CriteriaDecision],
        description="The decisions made based on the criteria",
    )
    overall_reasoning: Field(
        type=str, description="The reasoning behind the overall decision"
    )
    overall_decision: Field(
        type=bool,
        description="The overall decision made based on the criteria",
    )
```
