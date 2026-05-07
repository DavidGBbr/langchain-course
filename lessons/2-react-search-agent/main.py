from typing import List
from pydantic import BaseModel, Field

from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain.agents import create_agent
from tavily import TavilyClient

load_dotenv()

tavily = TavilyClient()

class Source(BaseModel):
    """Schema for a source used by the agent"""
    company_name: str = Field(description="The name of the company")
    tech_stack: List[str] = Field(description="List of technologies used by the company")
    seniority: str = Field(description="The seniority of the position")
    location: str = Field(description="The location of the position")
    salary_range: str = Field(description="The salary range of the position")
    is_remote: bool = Field(description="Whether the position is remote")
    job_description: str = Field(description="A brief description of the job and its responsibilities")
    url: str = Field(description="The URL of the source")

class AgentResponse(BaseModel):
    """Schema for agent response with answer and sources"""
    answer: str = Field(description="The agent's answer to the query")
    sources: List[Source] = Field(default_factory=list, description="List of sources used to answer the query")

@tool
def search(query: str) -> str:
    """
    Tool that searches over internet
    Args:
        query: The query to search for
    Returns:
        The search result
    """
    print(f"Searching for {query}...")
    return tavily.search(query=query)
    
llm = ChatOpenAI()
tools = [search]
# Fix: Removed List[] from response_format as it's not supported as a top-level generic alias
agent = create_agent(model=llm, tools=tools, response_format=AgentResponse)

def main():
    print("Starting job search...")
    # Agent Call with a specific prompt
    result = agent.invoke({"messages": [HumanMessage(content="Search for 3 job postings for AI Engineer using Langchain and summarize the responsibilities for each.")]})
    
    response = result['structured_response']
    
    print("\n" + "="*60)
    print("SEARCH RESULTS")
    print("="*60)
    print(f"Summary: {response.answer}")
    
    print("\nJOB POSTING DETAILS:")
    for i, job in enumerate(response.sources, 1):
        print(f"\n{i}. {job.company_name} - {job.seniority}")
        print(f"   Location: {job.location} ({'Remote' if job.is_remote else 'On-site'})")
        print(f"   Tech Stack: {', '.join(job.tech_stack)}")
        print(f"   Job Description: {job.job_description}")
        print(f"   Source URL: {job.url}")

if __name__ == "__main__":
    main()
