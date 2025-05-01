import os
from typing import Optional
from pydantic import BaseModel
from langchain.agents import Tool, initialize_agent, AgentType
from langchain_huggingface import HuggingFaceEndpoint
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from tavily import TavilyClient

# Set API keys (it's better to use environment variables or .env files in production)
os.environ["HUGGINGFACEHUB_API_TOKEN"] = ""
os.environ["TAVILY_API_KEY"] = ""

# Initialize the LLM (Choose one model, either HuggingFaceHub or ChatOpenAI)
llm = HuggingFaceEndpoint(
    repo_id="tiiuae/falcon-rw-1b",  # Example HuggingFace model, replace as needed
)

# Define Tavily client
tavily_client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])

# Define the search tool (Tavily)
def tavily_search(query: str) -> str:
    # Perform the search query using TavilyClient
    response = tavily_client.search(query,search_depth="advanced",max_results=3)
    
    # If the response contains results, extract and format the relevant content
    if 'results' in response:
        results_summary = ""
        for result in response['results']:
            title = result.get('title', 'No Title Available')
            content = result.get('content', 'No Content Available')
            results_summary += f"Title: {title}\nContent: {content}\n\n"
        return results_summary
    else:
        return "No results found."

# Wrap the Tavily search function in a Tool object
tavily_search_tool = Tool(
    name="tavily_search_tool",
    func=tavily_search,
    description="Searches the web using Tavily API for relevant results and summarizes them."
)

# Dummy function to simulate a tool (Web content extraction)
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
        tools=[web_reader, tavily_search_tool],
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
    response = llm.invoke(f"""You are an expert writer. Using the following research notes:
{research_notes}
Write a detailed and well-organized response to the original query:
"{query}"
Include sources if available and ensure clarity.
""")
    return {"final_answer": response}

# Define the state graph and the flow between agents
workflow = StateGraph(state_schema=MyState)
workflow.add_node("research_agent", research_agent)
workflow.add_node("answer_agent", answer_agent)  # Add the second agent
workflow.set_entry_point("research_agent")
workflow.add_edge("research_agent", "answer_agent")  # Link research_agent to answer_agent
workflow.add_edge("answer_agent", END)  # Link the final agent to END

# Compile the graph
graph_executor = workflow.compile()

# Execute the workflow
query = input("Enter your query: ")
results = graph_executor.invoke({"input": query})

# Check for final answer and print the results
if "final_answer" in results:
    print("Final Answer:\n", results["final_answer"])
else:
    print("No final answer found. Here is the full result:\n", results)
