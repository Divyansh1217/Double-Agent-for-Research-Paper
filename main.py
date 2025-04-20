import os
from typing import Optional
from pydantic import BaseModel
from langchain.agents import Tool, initialize_agent, AgentType
from langchain_community.llms import HuggingFaceHub
from langchain_community.tools.tavily_search.tool import TavilySearchResults
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI

# Set API keys (it's better to use environment variables or .env files in production)
os.environ["HUGGINGFACEHUB_API_TOKEN"] = ""
os.environ["TAVILY_API_KEY"] = ""

# Initialize the LLM (Choose one model, either HuggingFaceHub or ChatOpenAI)
llm = HuggingFaceHub(
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),  # Use the token from environment
    repo_id="mistralai/Mistral-7B-Instruct-v0.1",  # Example HuggingFace model, replace as needed
    task="text-generation",
    model_kwargs={"temperature": 0.3, "max_length": 1024}
)

# Define the search tool (Tavily)
tavily_tool = TavilySearchResults(max_results=3)

# Dummy function to simulate a tool
def deckreader(query: str) -> str:
    return f"Web content for: {query}"

# Define the web reader tool
web_reader = Tool(
    name="web_reader",
    func=deckreader,
    description="Reads and extracts useful information from URLs or topics."
)

# Define state schema
class MyState(BaseModel):
    input: str
    query: Optional[str] = None
    research: Optional[str] = None
    final_answer: Optional[str] = None

# First agent: gathers research
def research_agent(state: MyState):
    query = state.input
    agent = initialize_agent(
        tools=[web_reader, tavily_tool],
        llm=llm,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,  # Use the enum instead of string
        verbose=True,
        handle_parsing_errors=True  # ✅ This will retry if the LLM returns malformed output
    )
    research_output = agent.run(f"Find and summarize detailed, trustworthy information on: {query}")
    return {"query": query, "research": research_output}

# Second agent: writes the final answer
def answer_agent(state: MyState):
    research_notes = state.research
    query = state.query
    response = llm.predict(f"""You are an expert writer. Using the following research notes:
{research_notes}
Write a detailed and well-organized response to the original query:
"{query}"
Include sources if available and ensure clarity.
""")
    return {"final_answer": response}

# Create workflow graph
workflow = StateGraph(state_schema=MyState)
workflow.add_node("research_agent", research_agent)
workflow.add_node("answer_agent", answer_agent)
workflow.set_entry_point("research_agent")
workflow.add_edge("research_agent", "answer_agent")
workflow.add_edge("answer_agent", END)

# Compile graph
graph_executor = workflow.compile()

# Run query
query = input("Enter your query: ")
results = graph_executor.invoke({"input": query})
print("Final Answer:\n", results["final_answer"])
