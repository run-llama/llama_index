# LlamaIndex Packs Integration: Resume Coach Pack

This LLamaIndex pack takes a resume, a job description, a LLM, an embedding model, and optionally a Reader and a list of queries and returns the answer for each query by using the job description, job criterions and job keywords to evaluate the resume.

The purpose of this pack is to help job candidates match and improve their resumes to reach the job requirements with the default queries or anything else by passing custom queries.

## Code Usage

You can download the pack to a `./resume_coach_pack` directory:

```python
from llama_index.core.llama_pack import download_llama_pack

# download and install dependencies
ResumeCoachPack = download_llama_pack(
    "ResumeCoachPack", "./resume_coach_pack"
)
```

From here, you can use the pack, or inspect and modify the pack in `./resume_coach_pack`.

Then, you can set up the pack like so:

```python
# create the pack
resume_coach = ResumeCoachPack(
    job_description_path='job_description.txt', 
    resume_path='CV.pdf', 
    embed_model=embed_model, 
    llm=llm
)
```

In this example, `embed_model` has the type BaseEmbedding and `llm` has the type LLM.

Two additional parameters can be passed to the pack:
* reader: a Reader object that has the 'load_data' method
* queries: a Dictionary containing a key that describes each query and a value that contains the query itself



You can get the response from the pack by running the following code:

```python
response = resume_match_pack.run()
```


```python
response = resume_screener.run(resume_path="resume.pdf")
print(response.overall_decision)
```

The `response` will be dictionary that can be displayed as follows:

```python
for key, value in response.items():
    display(Markdown(f"<b>{key}</b>"))
    display(Markdown(f"{value}"))
    display(Markdown(f"---"))
```