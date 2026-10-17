"""
Social Impact RAG Example - Community Resource Discovery System

This example demonstrates how to use LlamaIndex to build a Retrieval-Augmented
Generation (RAG) system for social good applications, specifically enabling
nonprofits and community organizations to make their resources more accessible.

Features:
- Natural language querying of community resources
- Privacy-preserving local indexing (no data sent to external services)
- Cost-effective solution suitable for resource-constrained nonprofits
- Support for multiple resource formats (text, documents, structured data)
- Real-world use cases: job training, healthcare, education, mental health

Use Case: A nonprofit serving low-income communities can index their programs,
services, and resources, then allow community members to ask natural language
questions to discover what help is available.
"""

from typing import List, Optional
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Document
from llama_index.core.schema import Document as LlamaDocument
from llama_index.embeddings.openai import OpenAIEmbedding
import os


class CommunityResourceAssistant:
    """
    A RAG-based assistant for discovering community resources.
    
    This system allows nonprofits to:
    - Index their resources and services
    - Answer natural language questions from community members
    - Maintain privacy by using local indexing
    - Scale efficiently without high computational costs
    """
    
    def __init__(self, embed_model: Optional[str] = None):
        """
        Initialize the Community Resource Assistant.
        
        Args:
            embed_model: Embedding model to use (default: OpenAI's text-embedding-3-small)
        """
        self.embed_model = embed_model or "text-embedding-3-small"
        self.index = None
        self.resources_loaded = False
        
    def load_resources_from_text(self, resources: List[dict]) -> None:
        """
        Load community resources from structured data.
        
        Args:
            resources: List of resource dictionaries with 'name', 'description', 'category', 'contact'
        
        Example:
            resources = [
                {
                    "name": "Tech Skills Training",
                    "description": "12-week coding bootcamp for underserved youth...",
                    "category": "job training",
                    "contact": "training@nonprofit.org"
                },
                ...
            ]
        """
        documents = []
        for resource in resources:
            # Create a document combining all resource information
            content = f"""
Resource: {resource['name']}
Category: {resource['category']}
Description: {resource['description']}
Contact: {resource['contact']}
"""
            doc = LlamaDocument(text=content, metadata={
                "resource_name": resource['name'],
                "category": resource['category']
            })
            documents.append(doc)
        
        # Create index from documents
        self.index = VectorStoreIndex.from_documents(
            documents,
            embed_model=OpenAIEmbedding(model_name=self.embed_model)
        )
        self.resources_loaded = True
        print(f"✓ Loaded {len(documents)} community resources into index")
        
    def load_resources_from_directory(self, directory_path: str) -> None:
        """
        Load resources from a directory of documents (PDFs, TXT, etc).
        
        Args:
            directory_path: Path to directory containing resource documents
        """
        reader = SimpleDirectoryReader(directory_path)
        documents = reader.load_data()
        
        self.index = VectorStoreIndex.from_documents(
            documents,
            embed_model=OpenAIEmbedding(model_name=self.embed_model)
        )
        self.resources_loaded = True
        print(f"✓ Loaded {len(documents)} documents from {directory_path}")
        
    def query(self, question: str, similarity_top_k: int = 3) -> str:
        """
        Query the resource index with a natural language question.
        
        Args:
            question: Natural language question about available resources
            similarity_top_k: Number of top results to return
            
        Returns:
            Answer with relevant resources and contact information
            
        Raises:
            ValueError: If resources haven't been loaded yet
        """
        if not self.resources_loaded:
            raise ValueError("Resources not loaded. Call load_resources_from_text() or load_resources_from_directory() first.")
        
        query_engine = self.index.as_query_engine(similarity_top_k=similarity_top_k)
        response = query_engine.query(question)
        return str(response)
    
    def get_resources_by_category(self, category: str) -> List[str]:
        """
        Get all resources in a specific category.
        
        Args:
            category: Resource category to filter by
            
        Returns:
            List of resource names in that category
        """
        if not self.resources_loaded:
            return []
        
        # Query for resources in this category
        query_engine = self.index.as_query_engine()
        response = query_engine.query(f"List all resources in the {category} category")
        return str(response)


def demo_basic_usage():
    """
    Basic demo: Load sample resources and answer questions.
    """
    print("=" * 70)
    print("COMMUNITY RESOURCE ASSISTANT - BASIC DEMO")
    print("=" * 70)
    
    # Sample resources from a nonprofit
    sample_resources = [
        {
            "name": "TechStart Bootcamp",
            "description": "Intensive 12-week coding bootcamp for low-income youth ages 18-24. Covers full-stack web development, JavaScript, Python, and project-based learning. Includes job placement assistance and mentorship.",
            "category": "job training",
            "contact": "bootcamp@techstart.org | (555) 123-4567"
        },
        {
            "name": "Community Health Clinic",
            "description": "Free and low-cost healthcare services. No insurance required. Offers preventive care, dental services, mental health counseling, and prescription assistance programs.",
            "category": "healthcare",
            "contact": "clinic@community-health.org | Walk-ins welcome"
        },
        {
            "name": "Youth Mental Health Support",
            "description": "Peer support groups and counseling for teenagers dealing with anxiety, depression, or trauma. Led by trained facilitators and mental health professionals. Completely confidential.",
            "category": "mental health",
            "contact": "youth-support@nonprofitname.org | (555) 987-6543"
        },
        {
            "name": "Adult Literacy Program",
            "description": "Free English language classes and GED prep courses for adults. Small classes, flexible evening and weekend schedules. One-on-one tutoring also available.",
            "category": "education",
            "contact": "literacy@nonprofit.org | (555) 234-5678"
        },
        {
            "name": "Job Search Assistance",
            "description": "Resume writing workshops, interview coaching, and job placement services. Career counselors help match skills to local job opportunities.",
            "category": "job training",
            "contact": "jobs@nonprofit.org | By appointment"
        },
        {
            "name": "Childcare Support Program",
            "description": "Subsidized childcare for working parents. Includes preschool education, after-school programs, and emergency childcare services.",
            "category": "family services",
            "contact": "childcare@nonprofit.org | (555) 345-6789"
        }
    ]
    
    # Initialize assistant
    assistant = CommunityResourceAssistant()
    
    # Load resources
    print("\n1. Loading community resources...")
    assistant.load_resources_from_text(sample_resources)
    
    # Example queries
    print("\n2. Answering community member questions:\n")
    
    sample_questions = [
        "I need help learning to code. What programs are available?",
        "I'm looking for healthcare services and don't have insurance",
        "My teenager is struggling with anxiety. Where can we get help?",
        "I want to improve my English skills"
    ]
    
    for question in sample_questions:
        print(f"\n📋 Question: {question}")
        print("-" * 70)
        try:
            answer = assistant.query(question)
            print(f"📝 Answer:\n{answer}\n")
        except Exception as e:
            print(f"Note: Requires OpenAI API key. Error: {e}")
            print("To run this example, set OPENAI_API_KEY environment variable\n")


def demo_nonprofit_workflow():
    """
    Nonprofit workflow demo: shows how a real nonprofit would use this system.
    """
    print("=" * 70)
    print("NONPROFIT WORKFLOW EXAMPLE")
    print("=" * 70)
    print("""
Typical nonprofit setup workflow:

1. RESOURCE INVENTORY
   └─ Create document/database of all programs, services, locations, hours
   
2. LOAD INTO SYSTEM
   └─ assistant.load_resources_from_directory("./our_resources/")
   
3. DEPLOY TO COMMUNITY
   └─ Web interface: "Ask about services"
   └─ Phone system: "Press 1 for job training"
   └─ SMS: "Text your question to 555-HELP-NOW"
   
4. CONTINUOUS IMPROVEMENT
   └─ Track popular questions
   └─ Add new resources as programs expand
   └─ Update descriptions based on feedback

BENEFITS FOR NONPROFITS:
✓ Privacy: Data stays local, no cloud exposure
✓ Cost: No per-query fees like commercial APIs
✓ Accessibility: Works offline, in low-bandwidth areas
✓ Scalability: Add more resources without new infrastructure
✓ Community impact: Members find help faster
""")


if __name__ == "__main__":
    # Run basic demo (requires OPENAI_API_KEY environment variable)
    demo_basic_usage()
    
    # Show nonprofit workflow
    print("\n" + "=" * 70)
    demo_nonprofit_workflow()
    
    print("\n" + "=" * 70)
    print("NEXT STEPS:")
    print("=" * 70)
    print("""
1. Set your OpenAI API key:
   export OPENAI_API_KEY="sk-..."

2. Install dependencies:
   pip install llama-index-core llama-index-embeddings-openai

3. Customize for your nonprofit:
   - Replace sample_resources with your actual programs
   - Load from files: assistant.load_resources_from_directory("path")
   - Deploy with web framework (Flask, FastAPI, etc)

4. For more examples and documentation:
   - See social_impact_rag_docs.md for detailed guide
   - Check LlamaIndex docs: https://docs.llamaindex.ai/
""")
