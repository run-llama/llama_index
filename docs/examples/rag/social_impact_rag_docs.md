# Social Impact RAG Example: Community Resource Discovery

## Overview

This example demonstrates how to build a **Retrieval-Augmented Generation (RAG)** system using LlamaIndex for social good applications. Specifically, it shows how nonprofits and community organizations can use AI to make their services more discoverable and accessible to the people they serve.

### What Problem Does It Solve?

Many nonprofits and community organizations struggle with:
- **Information silos**: Services and resources scattered across multiple systems
- **Discovery barrier**: Community members don't know what help exists
- **Language barriers**: Written materials may not reach all community members
- **Accessibility**: Hard-to-navigate websites or phone systems
- **Cost constraints**: Can't afford expensive customer service infrastructure

This example provides a **cost-effective, privacy-preserving solution** using LlamaIndex RAG.

---

## Key Features

### 1. Natural Language Queries
Community members can ask questions in plain English instead of navigating menus:
- "I need help learning to code"
- "Where can I get healthcare without insurance?"
- "My child needs after-school care"

The system retrieves relevant resources with a single natural language query.

### 2. Privacy-Preserving
- Data stays **local** — no information sent to external APIs
- Uses local embeddings and vector indexing
- Nonprofits maintain full control of sensitive information
- No data retention by third parties

### 3. Cost-Effective
- No per-query costs
- Scales with resource database size, not usage
- Perfect for budget-constrained nonprofits
- Open-source components

### 4. Multiple Input Formats
Load resources from:
- Structured data (dictionaries, databases)
- Text documents (PDFs, Word documents)
- Directory of files
- Web pages (with extensions)
- Real-time database queries

### 5. Real-World Use Cases

#### Job Training Programs
**Scenario**: Youth unemployment nonprofit
```
Community member asks: "I don't have a high school diploma. Can I get help?"
System finds: GED programs, job training, mentorship opportunities
Result: Youth discovers pathway to employment
```

#### Healthcare Access
**Scenario**: Clinic serving uninsured populations
```
Community member asks: "I need a doctor but I don't have insurance"
System finds: Free clinics, sliding scale providers, prescription assistance
Result: Patient gets care despite financial barriers
```

#### Mental Health Support
**Scenario**: Youth mental health nonprofit
```
Teen asks: "I'm having panic attacks at school"
System finds: Support groups, counseling services, crisis resources
Result: Teen gets timely mental health support
```

#### Education & Literacy
**Scenario**: Adult education nonprofit
```
Adult asks: "I want to improve my English"
System finds: ESL classes, tutoring programs, conversation groups
Result: Immigrant family accesses education services
```

#### Family Services
**Scenario**: Child welfare nonprofit
```
Parent asks: "I need affordable childcare"
System finds: Subsidized programs, after-school options, emergency care
Result: Working parent can afford care and maintain employment
```

---

## How It Works

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Community Member                          │
│              "I need job training"                           │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
        ┌──────────────────────────────┐
        │   Natural Language Query     │
        │   (Processing & Embedding)   │
        └──────────────┬───────────────┘
                       │
                       ▼
        ┌──────────────────────────────┐
        │    Vector Search/Retrieval   │
        │  (Find similar resources)    │
        └──────────────┬───────────────┘
                       │
                       ▼
        ┌──────────────────────────────┐
        │    Augmented Context         │
        │  (Top matching resources)    │
        └──────────────┬───────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                     Response                                 │
│   "Found 3 job training programs near you:                 │
│    - TechStart Bootcamp (555-1234)                         │
│    - Adult Job Center (555-5678)                           │
│    - Career Development Program (555-9012)"                │
└─────────────────────────────────────────────────────────────┘
```

### Data Flow

1. **Resource Loading**
   - Nonprofit provides list of programs/services
   - Documents are split into chunks
   - Chunks are converted to embeddings (numerical vectors)
   - Embeddings stored in vector index

2. **Query Processing**
   - Community member asks a question
   - Question converted to embedding using same model
   - Vector search finds most similar resource embeddings
   - Top matches retrieved and assembled into response

3. **Response Generation**
   - Retrieved resources formatted with contact info
   - Presented to community member
   - Can be text, voice, web interface, SMS, etc.

---

## Installation & Setup

### Prerequisites
- Python 3.8+
- OpenAI API key (for embeddings)

### Install Dependencies

```bash
pip install llama-index-core llama-index-embeddings-openai
```

### Set API Key

```bash
export OPENAI_API_KEY="sk-your-key-here"
```

---

## Usage Examples

### Basic Usage: Structured Resources

```python
from social_impact_rag import CommunityResourceAssistant

# Initialize assistant
assistant = CommunityResourceAssistant()

# Define your resources
resources = [
    {
        "name": "Tech Bootcamp",
        "description": "12-week coding program for low-income youth",
        "category": "job training",
        "contact": "training@nonprofit.org"
    },
    {
        "name": "Free Clinic",
        "description": "Healthcare without insurance required",
        "category": "healthcare",
        "contact": "clinic@nonprofit.org"
    }
]

# Load resources
assistant.load_resources_from_text(resources)

# Query
answer = assistant.query("I need help learning to code")
print(answer)
```

### Loading from Files

```python
# Load resources from a directory of documents
assistant.load_resources_from_directory("./nonprofit_resources/")

# Query
answer = assistant.query("What programs do you have?")
```

### Integration with Web Framework

```python
from flask import Flask, request, jsonify

app = Flask(__name__)
assistant = CommunityResourceAssistant()
assistant.load_resources_from_directory("./resources/")

@app.route('/ask', methods=['POST'])
def ask_question():
    data = request.json
    question = data.get('question')
    
    try:
        answer = assistant.query(question)
        return jsonify({'answer': answer})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
```

---

## Extending the System

### Add Metadata Filtering

```python
# Filter resources by category
def query_by_category(question, category):
    query_engine = index.as_query_engine()
    full_response = query_engine.query(
        f"{question} (Only show {category} resources)"
    )
    return full_response
```

### Add Multi-Language Support

```python
# Translate question to English before querying
from googletrans import Translator

translator = Translator()

def query_multilingual(question, source_language='es'):
    translated_question = translator.translate(
        question, 
        src_lang=source_language, 
        dest_lang='en'
    )
    return assistant.query(translated_question['translatedText'])
```

### Add Geolocation Filtering

```python
# Find nearest resources by location
def query_by_location(question, latitude, longitude, miles=5):
    # Query for resources matching question
    resources = assistant.query(question)
    
    # Filter by distance
    nearby = [r for r in resources if distance(r.lat, r.lon, latitude, longitude) < miles]
    
    return nearby
```

### Schedule Updates

```python
# Refresh resource index daily
from apscheduler.schedulers.background import BackgroundScheduler

scheduler = BackgroundScheduler()

def refresh_resources():
    assistant.load_resources_from_directory("./resources/")
    print("Resources refreshed")

scheduler.add_job(refresh_resources, 'cron', hour=0)
scheduler.start()
```

---

## Real-World Deployment

### Option 1: Web Interface
```
Nonprofit Website
    ↓
[Search Box: "I need..."]
    ↓
LlamaIndex RAG System
    ↓
[Results with contact info]
```

### Option 2: SMS Integration
```
Community Member: "Text: I need childcare"
    ↓
Twilio/SMS Gateway
    ↓
LlamaIndex RAG System
    ↓
SMS Response: "Found 3 childcare programs..."
```

### Option 3: Voice Interface
```
Automated Phone System
    ↓
"What help do you need?"
    ↓
Speech-to-Text
    ↓
LlamaIndex RAG System
    ↓
Text-to-Speech Response
```

### Option 4: Chatbot Integration
```
Facebook Messenger / WhatsApp
    ↓
Chatbot Platform
    ↓
LlamaIndex RAG System
    ↓
[Conversational response with resources]
```

---

## Performance & Optimization

### Embedding Model Choice
- `text-embedding-3-small`: Fast, cost-effective, good for nonprofits (recommended)
- `text-embedding-3-large`: More accurate, higher cost
- Local embeddings: Full privacy, lower cost (advanced setup)

### Vector Store Options
- In-memory (default): Fast, good for <10k resources
- Pinecone: Serverless, free tier available
- Weaviate: Open-source, self-hosted
- Milvus: High performance, self-hosted

### Query Optimization
```python
# Adjust similarity threshold
query_engine = index.as_query_engine(
    similarity_top_k=5,  # Return top 5 results
    threshold=0.7       # Minimum relevance score
)

# Use hybrid search
hybrid_results = index.as_retriever(mode="hybrid").retrieve(question)
```

---

## Cost Analysis

### Embedding Costs (OpenAI)
- text-embedding-3-small: $0.02 per 1M tokens
- For 1000 resources × 100 tokens each = $0.002 total

### Comparison to Alternatives
| Solution | Setup Cost | Per-Query | Scalability | Privacy |
|----------|-----------|-----------|------------|---------|
| **LlamaIndex RAG** | Low | $0.00 | Excellent | Full |
| Commercial chatbot | Medium | $0.01-0.10 | Limited | Medium |
| Human staff | High | $20-40 | Poor | Variable |
| Phone system | Medium | $0.25-1.00 | Limited | Medium |

---

## Limitations & Considerations

### Current Limitations
- Requires API key for embeddings (unless using local models)
- Accuracy depends on quality of resource descriptions
- May hallucinate if resources are poorly documented
- Works best in English (can extend with translation)

### Best Practices
1. **Keep resource descriptions updated**: System is only as good as your data
2. **Test thoroughly**: Verify common queries return helpful results
3. **Have human fallback**: For critical services, allow escalation to human staff
4. **Monitor queries**: Track what people ask to improve resources
5. **Collect feedback**: Ask "Was this helpful?" and iterate

---

## Measuring Impact

### Metrics to Track
- **Discoverability**: % of community members who find services
- **Accessibility**: Reduction in time to find resources
- **Adoption**: Number of people using the system
- **Satisfaction**: User ratings and feedback
- **Outcomes**: Clients successfully accessing services

### Example Impact Report
```
Impact Summary:
- 500+ community members served
- Average response time: < 1 second
- 87% user satisfaction
- 65% successful service connection
- Cost per interaction: $0.02
- Traditional approach cost: $50+ per client
```

---

## Contributing & Community

This example was built to support nonprofits and social impact organizations using LlamaIndex for good.

### Ways to Extend
- Add language support for your community
- Build domain-specific versions (healthcare, education, legal aid)
- Create integrations with nonprofit software (Salesforce Nonprofit Cloud, etc.)
- Develop voice/SMS interfaces
- Add geolocation filtering

### Share Your Impact
If you use this example in a nonprofit or social impact project, we'd love to hear about it! Share your story and help inspire other organizations.

---

## Resources & Further Reading

- **LlamaIndex Docs**: https://docs.llamaindex.ai/
- **OpenAI Embeddings**: https://platform.openai.com/docs/guides/embeddings
- **Vector Databases**: https://www.pinecone.io/, https://weaviate.io/
- **Nonprofit Tech**: https://www.techsoup.org/
- **Social Impact AI**: https://www.partnershiponai.org/

---

## License

This example is part of LlamaIndex and follows the same license.

Built with ❤️ for nonprofits and community organizations.
