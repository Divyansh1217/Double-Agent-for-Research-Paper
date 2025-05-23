﻿# Double-Agent-for-Research-Paper
#Multi-Agent Query Processing System
This project is a multi-agent system that processes and answers user queries by gathering research from the web, summarizing it, and generating a detailed response using a series of automated agents.

Features
Web Research: Automatically gathers relevant information from the web using external APIs like Tavily.

Intelligent Answer Generation: Uses advanced LLMs (Language Learning Models) like Hugging Face and OpenAI to generate detailed, well-organized answers to queries.

State-based Workflow: Manages tasks using a state graph to ensure each agent operates in sequence for efficient query processing.

Trustworthy Information: Focuses on gathering trustworthy and detailed information for precise, high-quality answers.

Components
LLMs (Language Models):

HuggingFaceHub: Utilized for performing the main query processing tasks.

ChatOpenAI: Used for generating detailed responses based on gathered research.

Tools:

TavilySearchResults: Fetches web data related to the query.

Web Reader: A custom tool used to simulate web content extraction.

Agents:

Research Agent: Gathers information based on a user query using the web search tool and summarizes it.

Answer Agent: Takes the research gathered by the Research Agent and writes a detailed, well-organized answer.

StateGraph:

Manages the flow of tasks, ensuring agents are executed in the correct order to generate a final output.
