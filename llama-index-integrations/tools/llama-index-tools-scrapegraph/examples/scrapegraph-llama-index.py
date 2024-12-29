from pydantic import BaseModel, Field 
from llama_index.tools.scrapegraph.base import ScrapegraphToolSpec 
 
scrapegraph_tool = ScrapegraphToolSpec() 
 
 
class FounderSchema(BaseModel): 
    name: str = Field(description="Name of the founder") 
    role: str = Field(description="Role of the founder") 
    social_media: str = Field(description="Social media URL of the founder") 
 
class ListFoundersSchema(BaseModel): 
    founders: list[FounderSchema] = Field(description="List of founders") 
 
response = scrapegraph_tool.scrapegraph_smartscraper( 
    prompt="Extract product information", 
    url="https://scrapegraphai.com/", 
    api_key="sgai-***", 
    schema=ListFoundersSchema, 
) 
 
result = response["result"] 
 
for founder in result["founders"]: 
    print(founder)