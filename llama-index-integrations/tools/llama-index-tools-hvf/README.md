# HVF Tool (Hudson Valley Forestry API)

```bash
pip install llama-index-tools-hvf
```

This tool gives LLM agents native access to the
[Hudson Valley Forestry](https://www.hudsonvalleyforestry.com) agent API.
It covers three service divisions: **HVF Residential** (forestry mulching, selective
thinning, clearcut & grub, LiDAR mapping), **HVG Goat Grazing** (targeted grazing for
invasive species and inaccessible terrain), and **Commercial O&G** (pipeline ROW clearing,
vegetation management, site prep, LiDAR corridor mapping).

**OpenAPI spec**: https://app.hudsonvalleyforestry.com/openapi.json  
**Interactive docs**: https://app.hudsonvalleyforestry.com/api/docs

## Available Tools (9 total)

### HVF Residential

| Tool | Endpoint | Description |
|------|----------|-------------|
| `hvf_get_services()` | `GET /api/agent/services` | Get residential service catalog with pricing |
| `hvf_assess_property(lat, lng, acreage, service_type, vegetation_density)` | `POST /api/agent/assess` | Check eligibility and get price estimate |
| `hvf_submit_quote(email, name, acreage, service_type, property_description, ...)` | `POST /api/agent/quote` | Submit a residential quote request |

Service area: Hudson Valley NY, Berkshires MA, Western CT (~110-mile radius).  
Service types: `forestry_mulching`, `selective_thinning`, `clearcut_grub`, `lidar_mapping`.

### HVG Goat Grazing

| Tool | Endpoint | Description |
|------|----------|-------------|
| `hvg_get_services()` | `GET /api/agent/goat/services` | Get goat grazing service catalog |
| `hvg_assess_property(lat, lng, acreage, vegetation_type)` | `POST /api/agent/goat/assess` | Check eligibility (min 0.5 acres) |
| `hvg_submit_quote(email, name, acreage, service_type, property_description, ...)` | `POST /api/agent/goat/quote` | Submit a goat grazing quote request |

Service type: `goat_grazing`. Vegetation types: `invasive_brush`, `mixed_brush`, `light_grass`, `unknown`.

### Commercial O&G

| Tool | Endpoint | Description |
|------|----------|-------------|
| `og_get_services()` | `GET /api/agent/commercial/services` | Get commercial O&G service catalog |
| `og_assess_project(lat, lng, service_type, project_description, acreage, corridor_miles)` | `POST /api/agent/commercial/assess` | Check project location eligibility |
| `og_submit_quote(email, name, service_type, project_description, ...)` | `POST /api/agent/commercial/quote` | Submit a commercial quote request |

Service area: Northeast US (NY, NJ, CT, MA, VT, NH, ME, PA, RI, DE, MD).  
Service types: `row_clearing`, `vegetation_management`, `site_prep`, `lidar_corridor_mapping`.  
All commercial pricing is custom quoted.

## Usage

```python
from llama_index.tools.hvf import HVFToolSpec
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.llms.openai import OpenAI

tools = HVFToolSpec().to_tool_list()
agent = FunctionAgent(tools=tools, llm=OpenAI(model="gpt-4o"))

# Check if a residential property is eligible and get a price estimate
result = await agent.run(
    "Check if a 5-acre property at lat=41.8, lng=-73.9 "
    "is eligible for HVF forestry mulching and get a price estimate."
)

# Submit a residential quote
result = await agent.run(
    "Submit a forestry mulching quote for Jane Smith (jane@example.com, 845-555-0001). "
    "Her 4-acre wooded lot in Woodstock NY has heavy brush overgrowth."
)

# Check goat grazing eligibility
result = await agent.run(
    "Is a 2-acre hillside property at lat=41.9, lng=-73.8 with invasive knotweed "
    "eligible for HVG goat grazing? What would it cost?"
)

# Submit a commercial O&G quote
result = await agent.run(
    "Submit a commercial pipeline ROW clearing quote for Bob Smith at Northeast Pipeline LLC "
    "(bob@pipeline.com, 845-555-9999). They need 12 miles of corridor cleared by Q3 2026 "
    "near lat=41.5, lng=-74.0."
)
```

## Configuration

```python
# Custom base URL or timeout
hvf_tool = HVFToolSpec(
    base_url="https://app.hudsonvalleyforestry.com",
    timeout=60,
)
```

This loader is designed to be used as a way to interact with the
Hudson Valley Forestry agent API in an LLM Agent context.
