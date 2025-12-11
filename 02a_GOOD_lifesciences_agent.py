# Databricks notebook source
# MAGIC %md
# MAGIC # Life Sciences Orchestrator-Synthesizer Agent (`alpha`)
# MAGIC
# MAGIC
# MAGIC >`alpha_Agent` is a multi‑worker, orchestrated agent that routes between SQL aggregations and parallel vector‑search workers over four life‑science knowledge bases, then synthesizes the vector workers’ findings into a single, high‑level scientific answer.   
# MAGIC
# MAGIC ---    
# MAGIC
# MAGIC This notebook implements a LangGraph agent with an orchestrator-synthesizer architecture:
# MAGIC
# MAGIC **Architecture:**
# MAGIC - **Orchestrator**: Routes queries to appropriate workers
# MAGIC - **SQL Worker**: Handles simple aggregation queries using UC functions
# MAGIC - **Vector Search Workers (2 parallel)**: Search different vector indexes for relevant information
# MAGIC - **Synthesizer/Judge**: Combines results from workers and provides final answer
# MAGIC
# MAGIC **Features:**
# MAGIC - Parallel execution of vector search workers
# MAGIC - Confidence-based result ranking
# MAGIC - Models as code pattern
# MAGIC - Production-ready deployment configuration

# COMMAND ----------

# MAGIC %pip install -U -qqqq langgraph uv databricks-agents databricks-langchain mlflow-skinny[databricks] databricks-vectorsearch pyyaml
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import os
import yaml

# Function to read and parse the config.yaml file
def load_config():
    config_path = os.path.join(os.getcwd(), 'config.yaml')
    catalog = schema = model_base_name = vs_endpoint_name = None

    if os.path.exists(config_path):
        with open(config_path, 'r') as file:
            try:
                config = yaml.safe_load(file)
                catalog = config.get('catalog', 'your_catalog')
                schema = config.get('schema', 'your_schema')
                model_base_name = config.get('model_base_name', 'your_model_base_name')
                vs_endpoint_name = config.get('vs_endpoint_name', 'your_vectorsearch_endpoint_name')
            except yaml.YAMLError as e:
                print(f"Error parsing YAML file: {e}")

    return catalog, schema, model_base_name, vs_endpoint_name

# Load the configuration
catalog, schema, model_base_name, vs_endpoint_name = load_config()

# COMMAND ----------

# Widgets for catalog, schema, and model base name

dbutils.widgets.text("catalog", catalog, "Catalog")
dbutils.widgets.text("schema", schema, "Schema")
dbutils.widgets.text("model_base_name", model_base_name, "Model Base Name")
dbutils.widgets.text("vs_endpoint_name", vs_endpoint_name, "VectorSearch_endpoint") #lifesciences_vector_search

import os

# Set environment variables from widget values
os.environ["CATALOG"] = dbutils.widgets.get("catalog")
os.environ["SCHEMA"] = dbutils.widgets.get("schema")
os.environ["VECTOR_SEARCH_ENDPOINT"] = dbutils.widgets.get("vs_endpoint_name")
os.environ["MODEL_BASE_NAME"] = dbutils.widgets.get("model_base_name")

# Assign Python variables from environment variables
CATALOG = os.environ["CATALOG"]
SCHEMA = os.environ["SCHEMA"]
MODEL_BASE_NAME = os.environ["MODEL_BASE_NAME"]
VS_ENDPOINT_NAME = os.environ["VECTOR_SEARCH_ENDPOINT"]

# Construct Fully Qualified Unity Catalog model name
UC_MODEL_NAME = f"{CATALOG}.{SCHEMA}.alpha_{MODEL_BASE_NAME}"

print(f"Using catalog: {CATALOG}, schema: {SCHEMA}")
print(f"Model base name: {MODEL_BASE_NAME}")
print(f"FQ UC Model Name: {UC_MODEL_NAME}")
print(f"VectorSearch_endpoint: {VS_ENDPOINT_NAME}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Define the Agent in Code
# MAGIC
# MAGIC This agent uses an orchestrator-synthesizer pattern with true parallel execution using LangGraph's Send API.
# MAGIC
# MAGIC ### Architecture Flow
# MAGIC
# MAGIC ```mermaid
# MAGIC graph TD
# MAGIC     Start([User Query]) --> Orchestrator[Orchestrator<br/>Route Decision]
# MAGIC
# MAGIC     Orchestrator -->|SQL Query| SQLWorker[SQL Worker<br/>UC Functions]
# MAGIC     Orchestrator -->|Semantic Query| VectorWorker1[Vector Worker 1<br/>Genes & Proteins]
# MAGIC     Orchestrator -->|Semantic Query| VectorWorker2[Vector Worker 2<br/>Pathways & Compounds]
# MAGIC
# MAGIC     SQLWorker --> End1([Final Answer])
# MAGIC
# MAGIC     VectorWorker1 --> Synthesizer[Synthesizer/Judge<br/>Confidence-Based Synthesis]
# MAGIC     VectorWorker2 --> Synthesizer
# MAGIC
# MAGIC     Synthesizer --> End2([Final Answer])
# MAGIC
# MAGIC     style Orchestrator fill:#e1f5ff
# MAGIC     style SQLWorker fill:#fff4e1
# MAGIC     style VectorWorker1 fill:#e8f5e9
# MAGIC     style VectorWorker2 fill:#e8f5e9
# MAGIC     style Synthesizer fill:#f3e5f5
# MAGIC     style Start fill:#fce4ec
# MAGIC     style End1 fill:#fce4ec
# MAGIC     style End2 fill:#fce4ec
# MAGIC ```
# MAGIC
# MAGIC ### Execution Patterns
# MAGIC
# MAGIC 1. **SQL Path**: Orchestrator → SQL Worker → END (direct answer, no synthesis)
# MAGIC 2. **Vector Path**: Orchestrator → [Vector Worker 1 + Vector Worker 2 in parallel] → Synthesizer → END

# COMMAND ----------

# DBTITLE 1,%%writefile alpha_agent.py
# MAGIC %%writefile alpha_agent.py
# MAGIC from typing import Annotated, Any, Generator, Literal, Optional, Sequence, TypedDict, Union
# MAGIC import json
# MAGIC from operator import add
# MAGIC import os
# MAGIC
# MAGIC import mlflow
# MAGIC from databricks_langchain import ChatDatabricks, UCFunctionToolkit, VectorSearchRetrieverTool
# MAGIC from langchain.messages import AIMessage, AIMessageChunk, AnyMessage, HumanMessage, SystemMessage
# MAGIC from langchain_core.runnables import RunnableConfig, RunnableLambda
# MAGIC from langchain_core.tools import BaseTool
# MAGIC from langgraph.graph import END, StateGraph
# MAGIC from langgraph.graph.message import add_messages
# MAGIC from langgraph.prebuilt.tool_node import ToolNode
# MAGIC from langgraph.types import Send
# MAGIC from mlflow.pyfunc import ResponsesAgent
# MAGIC from mlflow.types.responses import (
# MAGIC     ResponsesAgentRequest,
# MAGIC     ResponsesAgentResponse,
# MAGIC     ResponsesAgentStreamEvent,
# MAGIC     output_to_responses_items_stream,
# MAGIC     to_chat_completions_input,
# MAGIC )
# MAGIC
# MAGIC ############################################
# MAGIC # Configuration
# MAGIC ############################################
# MAGIC LLM_ENDPOINT_NAME = "databricks-claude-3-7-sonnet"
# MAGIC
# MAGIC # CATALOG = os.environ.get("CATALOG", "<your_catalog>") ## user to update 
# MAGIC # SCHEMA = os.environ.get("SCHEMA", "<your_schema>") ## user to update
# MAGIC # VECTOR_SEARCH_ENDPOINT = os.environ.get("VECTOR_SEARCH_ENDPOINT", "<your_vs_endpoint>") ## user to update
# MAGIC
# MAGIC ## hardcoding for the example here | please update
# MAGIC CATALOG = "mmt"
# MAGIC SCHEMA = "LS_agent"
# MAGIC VECTOR_SEARCH_ENDPOINT = "ls_vs_mmt"
# MAGIC
# MAGIC llm = ChatDatabricks(endpoint=LLM_ENDPOINT_NAME)
# MAGIC
# MAGIC ############################################
# MAGIC # Define Tools
# MAGIC ############################################
# MAGIC
# MAGIC # SQL Worker Tools - UC Functions for aggregations
# MAGIC # We have specific functions for each table since UC SQL functions can't use dynamic table names
# MAGIC UC_TOOL_NAMES = [
# MAGIC     # Count functions
# MAGIC     f"{CATALOG}.{SCHEMA}.count_genes_knowledge_rows",
# MAGIC     f"{CATALOG}.{SCHEMA}.count_proteins_knowledge_rows",
# MAGIC     f"{CATALOG}.{SCHEMA}.count_pathways_knowledge_rows",
# MAGIC     f"{CATALOG}.{SCHEMA}.count_compounds_knowledge_rows",
# MAGIC     # High confidence count functions
# MAGIC     f"{CATALOG}.{SCHEMA}.count_high_confidence_genes_knowledge",
# MAGIC     f"{CATALOG}.{SCHEMA}.count_high_confidence_proteins_knowledge",
# MAGIC     f"{CATALOG}.{SCHEMA}.count_high_confidence_pathways_knowledge",
# MAGIC     f"{CATALOG}.{SCHEMA}.count_high_confidence_compounds_knowledge",
# MAGIC     # Average confidence functions
# MAGIC     f"{CATALOG}.{SCHEMA}.avg_confidence_genes_knowledge",
# MAGIC     f"{CATALOG}.{SCHEMA}.avg_confidence_proteins_knowledge",
# MAGIC     f"{CATALOG}.{SCHEMA}.avg_confidence_pathways_knowledge",
# MAGIC     f"{CATALOG}.{SCHEMA}.avg_confidence_compounds_knowledge",
# MAGIC ]
# MAGIC uc_toolkit = UCFunctionToolkit(function_names=UC_TOOL_NAMES)
# MAGIC sql_tools = uc_toolkit.tools
# MAGIC
# MAGIC # Vector Search Tools - Four knowledge bases
# MAGIC VECTOR_SEARCH_TOOLS = [
# MAGIC     VectorSearchRetrieverTool(
# MAGIC         index_name=f"{CATALOG}.{SCHEMA}.genes_knowledge_vs_index",
# MAGIC         columns=["id", "name", "description", "confidence"],
# MAGIC         num_results=3,
# MAGIC         name="search_genes",
# MAGIC         description="Search the genes knowledge base for information about genes, their functions, and associated diseases."
# MAGIC     ),
# MAGIC     VectorSearchRetrieverTool(
# MAGIC         index_name=f"{CATALOG}.{SCHEMA}.proteins_knowledge_vs_index",
# MAGIC         columns=["id", "name", "description", "confidence"],
# MAGIC         num_results=3,
# MAGIC         name="search_proteins",
# MAGIC         description="Search the proteins knowledge base for information about protein structures, functions, and modifications."
# MAGIC     ),
# MAGIC     VectorSearchRetrieverTool(
# MAGIC         index_name=f"{CATALOG}.{SCHEMA}.pathways_knowledge_vs_index",
# MAGIC         columns=["id", "name", "description", "confidence"],
# MAGIC         num_results=3,
# MAGIC         name="search_pathways",
# MAGIC         description="Search the pathways knowledge base for information about biological pathways, metabolic processes, and signaling cascades."
# MAGIC     ),
# MAGIC     VectorSearchRetrieverTool(
# MAGIC         index_name=f"{CATALOG}.{SCHEMA}.compounds_knowledge_vs_index",
# MAGIC         columns=["id", "name", "description", "confidence"],
# MAGIC         num_results=3,
# MAGIC         name="search_compounds",
# MAGIC         description="Search the compounds knowledge base for information about drugs, small molecules, and therapeutic agents."
# MAGIC     ),
# MAGIC ]
# MAGIC
# MAGIC ############################################
# MAGIC # Agent State
# MAGIC ############################################
# MAGIC
# MAGIC def merge_worker_results(left: Optional[dict], right: Optional[dict]) -> dict:
# MAGIC     """Merge worker results from parallel execution."""
# MAGIC     if left is None:
# MAGIC         return right or {}
# MAGIC     if right is None:
# MAGIC         return left
# MAGIC     # Merge dictionaries - each worker adds its own key
# MAGIC     return {**left, **right}
# MAGIC
# MAGIC class AgentState(TypedDict):
# MAGIC     messages: Annotated[Sequence[AnyMessage], add_messages]
# MAGIC     custom_inputs: Optional[dict[str, Any]]
# MAGIC     custom_outputs: Optional[dict[str, Any]]
# MAGIC     worker_results: Annotated[Optional[dict[str, Any]], merge_worker_results]
# MAGIC     route_decision: Optional[str]
# MAGIC
# MAGIC ############################################
# MAGIC # Orchestrator Node
# MAGIC ############################################
# MAGIC
# MAGIC def orchestrator_node(state: AgentState, config: RunnableConfig):
# MAGIC     """
# MAGIC     Orchestrator decides whether to route to SQL worker or vector workers.
# MAGIC     """
# MAGIC     messages = state["messages"]
# MAGIC     last_message = messages[-1]
# MAGIC
# MAGIC     orchestrator_prompt = """You are an orchestrator for a life sciences knowledge base system.
# MAGIC
# MAGIC Analyze the user's question and decide:
# MAGIC - Route to "sql" if the question is asking for simple counts, aggregations, or statistics about tables (e.g., "How many genes?", "What's the average confidence?", "Count of high confidence proteins")
# MAGIC - Route to "vector" for all other questions that require semantic search of knowledge content
# MAGIC
# MAGIC Respond with ONLY one word: either "sql" or "vector"
# MAGIC """
# MAGIC
# MAGIC     decision_messages = [
# MAGIC         SystemMessage(content=orchestrator_prompt),
# MAGIC         last_message
# MAGIC     ]
# MAGIC
# MAGIC     response = llm.invoke(decision_messages)
# MAGIC     decision = response.content.strip().lower()
# MAGIC
# MAGIC     # Default to vector if unclear
# MAGIC     if "sql" not in decision and "vector" not in decision:
# MAGIC         decision = "vector"
# MAGIC     elif "sql" in decision:
# MAGIC         decision = "sql"
# MAGIC     else:
# MAGIC         decision = "vector"
# MAGIC
# MAGIC     return {
# MAGIC         "route_decision": decision,
# MAGIC         "messages": [AIMessage(content=f"[Orchestrator: Routing to {decision} worker]")]
# MAGIC     }
# MAGIC
# MAGIC ############################################
# MAGIC # SQL Worker Node
# MAGIC ############################################
# MAGIC
# MAGIC def sql_worker_node(state: AgentState, config: RunnableConfig):
# MAGIC     """
# MAGIC     SQL worker handles aggregation queries using UC functions.
# MAGIC     """
# MAGIC     messages = state["messages"]
# MAGIC     user_message = [m for m in messages if isinstance(m, HumanMessage)][-1]
# MAGIC
# MAGIC     sql_prompt = f"""You are a SQL worker for a life sciences knowledge base.
# MAGIC
# MAGIC Available tables and UC functions:
# MAGIC
# MAGIC For genes_knowledge table:
# MAGIC - count_genes_knowledge_rows(): Count total rows
# MAGIC - count_high_confidence_genes_knowledge(min_confidence): Count entries above confidence threshold
# MAGIC - avg_confidence_genes_knowledge(): Get average confidence
# MAGIC
# MAGIC For proteins_knowledge table:
# MAGIC - count_proteins_knowledge_rows(): Count total rows
# MAGIC - count_high_confidence_proteins_knowledge(min_confidence): Count entries above confidence threshold
# MAGIC - avg_confidence_proteins_knowledge(): Get average confidence
# MAGIC
# MAGIC For pathways_knowledge table:
# MAGIC - count_pathways_knowledge_rows(): Count total rows
# MAGIC - count_high_confidence_pathways_knowledge(min_confidence): Count entries above confidence threshold
# MAGIC - avg_confidence_pathways_knowledge(): Get average confidence
# MAGIC
# MAGIC For compounds_knowledge table:
# MAGIC - count_compounds_knowledge_rows(): Count total rows
# MAGIC - count_high_confidence_compounds_knowledge(min_confidence): Count entries above confidence threshold
# MAGIC - avg_confidence_compounds_knowledge(): Get average confidence
# MAGIC
# MAGIC Use these functions to answer the user's question about table statistics and aggregations.
# MAGIC """
# MAGIC
# MAGIC     sql_llm = llm.bind_tools(sql_tools)
# MAGIC     sql_messages = [SystemMessage(content=sql_prompt), user_message]
# MAGIC
# MAGIC     response = sql_llm.invoke(sql_messages)
# MAGIC     result_messages = [response]
# MAGIC
# MAGIC     # Execute tool calls if present
# MAGIC     if response.tool_calls:
# MAGIC         tool_node = ToolNode(sql_tools)
# MAGIC         tool_results = tool_node.invoke({"messages": [response]})
# MAGIC         result_messages.extend(tool_results["messages"])
# MAGIC
# MAGIC         # Get final answer from LLM
# MAGIC         final_response = llm.invoke(sql_messages + result_messages)
# MAGIC         result_messages.append(final_response)
# MAGIC
# MAGIC     # SQL worker returns final answer directly (no synthesis needed)
# MAGIC     final_answer = result_messages[-1].content if result_messages else "No results available."
# MAGIC
# MAGIC     return {
# MAGIC         "messages": [AIMessage(content=final_answer)]
# MAGIC     }
# MAGIC
# MAGIC ############################################
# MAGIC # Vector Worker Nodes
# MAGIC ############################################
# MAGIC
# MAGIC def vector_worker_1_node(state: AgentState, config: RunnableConfig):
# MAGIC     """
# MAGIC     Vector worker 1: Searches genes and proteins indexes.
# MAGIC     """
# MAGIC     messages = state["messages"]
# MAGIC     user_message = [m for m in messages if isinstance(m, HumanMessage)][-1]
# MAGIC
# MAGIC     worker_tools = [VECTOR_SEARCH_TOOLS[0], VECTOR_SEARCH_TOOLS[1]]  # genes, proteins
# MAGIC
# MAGIC     worker_prompt = """You are Vector Worker 1, specializing in genes and proteins knowledge.
# MAGIC
# MAGIC Search the genes and proteins knowledge bases to find relevant information for the user's question.
# MAGIC Use BOTH search tools to gather relevant information. Return the results with their confidence scores.
# MAGIC """
# MAGIC
# MAGIC     worker_llm = llm.bind_tools(worker_tools)
# MAGIC     worker_messages = [SystemMessage(content=worker_prompt), user_message]
# MAGIC
# MAGIC     response = worker_llm.invoke(worker_messages)
# MAGIC     result_messages = [response]
# MAGIC
# MAGIC     # Execute tool calls and capture results
# MAGIC     search_results = []
# MAGIC     if response.tool_calls:
# MAGIC         tool_node = ToolNode(worker_tools)
# MAGIC         tool_results = tool_node.invoke({"messages": [response]})
# MAGIC         result_messages.extend(tool_results["messages"])
# MAGIC
# MAGIC         # Extract tool result content
# MAGIC         for msg in tool_results["messages"]:
# MAGIC             if hasattr(msg, 'content') and msg.content:
# MAGIC                 search_results.append(str(msg.content))
# MAGIC
# MAGIC         # Synthesize findings if we have results
# MAGIC         if search_results:
# MAGIC             final_response = llm.invoke(worker_messages + result_messages + [
# MAGIC                 SystemMessage(content="Summarize the search results concisely, highlighting key findings and confidence scores.")
# MAGIC             ])
# MAGIC             result_messages.append(final_response)
# MAGIC
# MAGIC     # Compile results for synthesizer
# MAGIC     final_result = result_messages[-1].content if result_messages else "No results found"
# MAGIC     if search_results:
# MAGIC         final_result = f"{final_result}\n\nRaw search results:\n" + "\n---\n".join(search_results[:500])  # Limit size
# MAGIC
# MAGIC     worker_results = state.get("worker_results", {})
# MAGIC     worker_results["vector_worker_1"] = {
# MAGIC         "result": final_result,
# MAGIC         "confidence": 0.80,
# MAGIC         "source": "Vector Worker 1 (Genes & Proteins)",
# MAGIC         "tool_calls_made": len(response.tool_calls) if response.tool_calls else 0
# MAGIC     }
# MAGIC
# MAGIC     return {
# MAGIC         "worker_results": worker_results,
# MAGIC         "messages": [AIMessage(content=f"[Vector Worker 1 completed with {len(search_results)} results]")]
# MAGIC     }
# MAGIC
# MAGIC def vector_worker_2_node(state: AgentState, config: RunnableConfig):
# MAGIC     """
# MAGIC     Vector worker 2: Searches pathways and compounds indexes.
# MAGIC     """
# MAGIC     messages = state["messages"]
# MAGIC     user_message = [m for m in messages if isinstance(m, HumanMessage)][-1]
# MAGIC
# MAGIC     worker_tools = [VECTOR_SEARCH_TOOLS[2], VECTOR_SEARCH_TOOLS[3]]  # pathways, compounds
# MAGIC
# MAGIC     worker_prompt = """You are Vector Worker 2, specializing in pathways and compounds knowledge.
# MAGIC
# MAGIC Search the pathways and compounds knowledge bases to find relevant information for the user's question.
# MAGIC Use BOTH search tools to gather relevant information. Return the results with their confidence scores.
# MAGIC """
# MAGIC
# MAGIC     worker_llm = llm.bind_tools(worker_tools)
# MAGIC     worker_messages = [SystemMessage(content=worker_prompt), user_message]
# MAGIC
# MAGIC     response = worker_llm.invoke(worker_messages)
# MAGIC     result_messages = [response]
# MAGIC
# MAGIC     # Execute tool calls and capture results
# MAGIC     search_results = []
# MAGIC     if response.tool_calls:
# MAGIC         tool_node = ToolNode(worker_tools)
# MAGIC         tool_results = tool_node.invoke({"messages": [response]})
# MAGIC         result_messages.extend(tool_results["messages"])
# MAGIC
# MAGIC         # Extract tool result content
# MAGIC         for msg in tool_results["messages"]:
# MAGIC             if hasattr(msg, 'content') and msg.content:
# MAGIC                 search_results.append(str(msg.content))
# MAGIC
# MAGIC         # Synthesize findings if we have results
# MAGIC         if search_results:
# MAGIC             final_response = llm.invoke(worker_messages + result_messages + [
# MAGIC                 SystemMessage(content="Summarize the search results concisely, highlighting key findings and confidence scores.")
# MAGIC             ])
# MAGIC             result_messages.append(final_response)
# MAGIC
# MAGIC     # Compile results for synthesizer
# MAGIC     final_result = result_messages[-1].content if result_messages else "No results found"
# MAGIC     if search_results:
# MAGIC         final_result = f"{final_result}\n\nRaw search results:\n" + "\n---\n".join(search_results[:500])  # Limit size
# MAGIC
# MAGIC     worker_results = state.get("worker_results", {})
# MAGIC     worker_results["vector_worker_2"] = {
# MAGIC         "result": final_result,
# MAGIC         "confidence": 0.80,
# MAGIC         "source": "Vector Worker 2 (Pathways & Compounds)",
# MAGIC         "tool_calls_made": len(response.tool_calls) if response.tool_calls else 0
# MAGIC     }
# MAGIC
# MAGIC     return {
# MAGIC         "worker_results": worker_results,
# MAGIC         "messages": [AIMessage(content=f"[Vector Worker 2 completed with {len(search_results)} results]")]
# MAGIC     }
# MAGIC
# MAGIC ############################################
# MAGIC # Synthesizer/Judge Node
# MAGIC ############################################
# MAGIC
# MAGIC def synthesizer_node(state: AgentState, config: RunnableConfig):
# MAGIC     """
# MAGIC     Synthesizer/judge combines all worker results and produces final answer.
# MAGIC     """
# MAGIC     messages = state["messages"]
# MAGIC     user_message = [m for m in messages if isinstance(m, HumanMessage)][-1]
# MAGIC     worker_results = state.get("worker_results", {})
# MAGIC
# MAGIC     # Compile worker results
# MAGIC     results_summary = "\n\n".join([
# MAGIC         f"**{key}**:\n- Confidence: {val['confidence']}\n- Result: {val['result']}"
# MAGIC         for key, val in worker_results.items()
# MAGIC     ])
# MAGIC
# MAGIC     system_prompt = """You are a synthesizer/judge agent for a life sciences knowledge system.
# MAGIC
# MAGIC You have received results from multiple specialized workers. Your job is to:
# MAGIC 1. Evaluate the confidence and relevance of each result
# MAGIC 2. Synthesize the information into a coherent, accurate answer
# MAGIC 3. Cite which sources you used and their confidence levels
# MAGIC 4. If results conflict, prefer higher confidence sources
# MAGIC
# MAGIC Provide an answer based on the worker results. Be scientific and precise.
# MAGIC """
# MAGIC
# MAGIC     user_request = f"""Original Question: {user_message.content}
# MAGIC
# MAGIC Worker Results:
# MAGIC {results_summary}
# MAGIC
# MAGIC Synthesize these results to answer the original question."""
# MAGIC
# MAGIC     final_messages = [
# MAGIC         SystemMessage(content=system_prompt),
# MAGIC         HumanMessage(content=user_request)
# MAGIC     ]
# MAGIC     response = llm.invoke(final_messages)
# MAGIC
# MAGIC     return {
# MAGIC         "messages": [response]
# MAGIC     }
# MAGIC
# MAGIC ############################################
# MAGIC # Routing Logic
# MAGIC ############################################
# MAGIC
# MAGIC def route_after_orchestrator(state: AgentState):
# MAGIC     """Route based on orchestrator decision using Send for parallel execution."""
# MAGIC     decision = state.get("route_decision", "vector")
# MAGIC
# MAGIC     if decision == "sql":
# MAGIC         # SQL worker goes directly to END
# MAGIC         return Send("sql_worker", state)
# MAGIC     else:
# MAGIC         # Send to both vector workers in parallel
# MAGIC         return [
# MAGIC             Send("vector_worker_1", state),
# MAGIC             Send("vector_worker_2", state),
# MAGIC         ]
# MAGIC
# MAGIC ############################################
# MAGIC # Build the Graph
# MAGIC ############################################
# MAGIC
# MAGIC def create_orchestrator_agent():
# MAGIC     """Create the orchestrator-synthesizer agent graph with true parallel execution using Send."""
# MAGIC     workflow = StateGraph(AgentState)
# MAGIC
# MAGIC     # Add nodes
# MAGIC     workflow.add_node("orchestrator", orchestrator_node)
# MAGIC     workflow.add_node("sql_worker", sql_worker_node)
# MAGIC     workflow.add_node("vector_worker_1", vector_worker_1_node)
# MAGIC     workflow.add_node("vector_worker_2", vector_worker_2_node)
# MAGIC     workflow.add_node("synthesizer", synthesizer_node)
# MAGIC
# MAGIC     # Define flow
# MAGIC     workflow.set_entry_point("orchestrator")
# MAGIC
# MAGIC     # Orchestrator uses Send for dynamic parallel routing
# MAGIC     workflow.add_conditional_edges("orchestrator", route_after_orchestrator)
# MAGIC
# MAGIC     # SQL worker goes directly to END (no synthesis needed)
# MAGIC     workflow.add_edge("sql_worker", END)
# MAGIC
# MAGIC     # Both vector workers go to synthesizer
# MAGIC     workflow.add_edge("vector_worker_1", "synthesizer")
# MAGIC     workflow.add_edge("vector_worker_2", "synthesizer")
# MAGIC
# MAGIC     # Synthesizer goes to END
# MAGIC     workflow.add_edge("synthesizer", END)
# MAGIC
# MAGIC     return workflow.compile()
# MAGIC
# MAGIC ############################################
# MAGIC # ResponsesAgent Wrapper
# MAGIC ############################################
# MAGIC
# MAGIC class LangGraphResponsesAgent(ResponsesAgent):
# MAGIC     def __init__(self, agent):
# MAGIC         self.agent = agent
# MAGIC
# MAGIC     def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
# MAGIC         outputs = [
# MAGIC             event.item
# MAGIC             for event in self.predict_stream(request)
# MAGIC             if event.type == "response.output_item.done"
# MAGIC         ]
# MAGIC         return ResponsesAgentResponse(output=outputs, custom_outputs=request.custom_inputs)
# MAGIC
# MAGIC     def predict_stream(
# MAGIC         self,
# MAGIC         request: ResponsesAgentRequest,
# MAGIC     ) -> Generator[ResponsesAgentStreamEvent, None, None]:
# MAGIC         cc_msgs = to_chat_completions_input([i.model_dump() for i in request.input])
# MAGIC
# MAGIC         for event in self.agent.stream({"messages": cc_msgs}, stream_mode=["updates", "messages"]):
# MAGIC             if event[0] == "updates":
# MAGIC                 for node_data in event[1].values():
# MAGIC                     if len(node_data.get("messages", [])) > 0:
# MAGIC                         yield from output_to_responses_items_stream(node_data["messages"])
# MAGIC             elif event[0] == "messages":
# MAGIC                 try:
# MAGIC                     chunk = event[1][0]
# MAGIC                     if isinstance(chunk, AIMessageChunk) and (content := chunk.content):
# MAGIC                         yield ResponsesAgentStreamEvent(
# MAGIC                             **self.create_text_delta(delta=content, item_id=chunk.id),
# MAGIC                         )
# MAGIC                 except Exception as e:
# MAGIC                     print(f"Stream error: {e}")
# MAGIC
# MAGIC ############################################
# MAGIC # Initialize Agent
# MAGIC ############################################
# MAGIC
# MAGIC mlflow.langchain.autolog()
# MAGIC agent = create_orchestrator_agent()
# MAGIC AGENT = LangGraphResponsesAgent(agent)
# MAGIC mlflow.models.set_model(AGENT)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Test `alpha` Agent

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,checking the tools
# from alpha_agent import uc_toolkit

# for tool in uc_toolkit.tools:
#     print("Testing tool:", tool.name)
#     try:
#         # try calling it with minimal params
#         res = tool.invoke({"min_confidence": 0.9}) if "high_confidence" in tool.name else tool.invoke({})
#         print("  OK:", res)
#     except Exception as e:
#         print("  ERROR:", e)

# COMMAND ----------

# Re-fetch widget values after Python restart
catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
model_base_name = dbutils.widgets.get("model_base_name")
vs_endpoint_name = dbutils.widgets.get("vs_endpoint_name")

# Construct Fully Qualified Unity Catalog model name
UC_MODEL_NAME = f"{catalog}.{schema}.alpha_{model_base_name}"

print(f"Using catalog: {catalog}, schema: {schema}")
print(f"Model base name: {model_base_name}")
print(f"FQ UC Model Name: {UC_MODEL_NAME}")
print(f"VectorSearch_endpoint: {vs_endpoint_name}")

# Import the agent
from alpha_agent import AGENT

# COMMAND ----------

# Test 1: SQL routing - aggregation query
print("=" * 80)
print("Test 1: SQL Worker (Aggregation Query)")
print("=" * 80)
result1 = AGENT.predict(
    {
        "input": [
            {
                "role": "user",
                "content": f"How many rows are in the {catalog}.{schema}.genes_knowledge table?",
            }
        ]
    }
)
print(result1.model_dump(exclude_none=True))

# COMMAND ----------


# Test 2: Vector routing - semantic query
print("=" * 80)
print("Test 2: Vector Workers (Semantic Query)")
print("=" * 80)
result2 = AGENT.predict(
    {
        "input": [
            {
                "role": "user",
                "content": "What proteins are involved in cell signaling and what pathways do they participate in?",
            }
        ]
    }
)
print(result2.model_dump(exclude_none=True))

# COMMAND ----------

# Test 3: Complex scientific query
print("=" * 80)
print("Test 3: Complex Scientific Query")
print("=" * 80)
result3 = AGENT.predict(
    {
        "input": [
            {
                "role": "user",
                "content": "Tell me about kinase proteins and any compounds that target them for cancer treatment.",
            }
        ]
    }
)
print(result3.model_dump(exclude_none=True))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Log `alpha` Agent as an MLflow Model

# COMMAND ----------

from alpha_agent import UC_TOOL_NAMES, VECTOR_SEARCH_TOOLS
import mlflow
from mlflow.models.resources import DatabricksFunction
from pkg_resources import get_distribution

# Collect all resources for automatic auth passthrough
resources = []

# Add vector search tools
for tool in VECTOR_SEARCH_TOOLS:
    resources.extend(tool.resources)

# Add UC functions
for tool_name in UC_TOOL_NAMES:
    resources.append(DatabricksFunction(function_name=tool_name))

# Log the model
with mlflow.start_run():
    logged_agent_info = mlflow.pyfunc.log_model(        
        name="alpha_agent",
        python_model="alpha_agent.py",
        pip_requirements=[
            "databricks-langchain",
            "databricks-vectorsearch",
            "psycopg[binary]",  # Required by databricks-langchain for checkpoint functionality
            f"langgraph=={get_distribution('langgraph').version}",
            f"databricks-connect=={get_distribution('databricks-connect').version}",
        ],
        resources=resources,
    )

print(f"Agent logged: {logged_agent_info.model_uri}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## TEST Evaluate `alpha` Agent

# COMMAND ----------

import mlflow
from mlflow.genai.scorers import RelevanceToQuery, Safety

eval_dataset = [
    {
        "inputs": {
            "input": [
                {
                    "role": "user",
                    "content": f"How many genes are in the {catalog}.{schema}.genes_knowledge table?",
                }
            ]
        },
        "expected_response": "There are 100 genes in the table.",
    },
    {
        "inputs": {
            "input": [
                {"role": "user", "content": "What do you know about kinase proteins?"}
            ]
        },
        "expected_response": "Kinases are proteins that catalyze phosphorylation reactions and play critical roles in cell signaling.",
    },
    {
        "inputs": {
            "input": [{"role": "user", "content": "Tell me about metabolic pathways."}]
        },
        "expected_response": "Metabolic pathways involve multiple enzymes and metabolites that work together to convert substrates to products.",
    },
]

eval_results = mlflow.genai.evaluate(
    data=eval_dataset,
    predict_fn=lambda input: AGENT.predict({"input": input}),
    scorers=[RelevanceToQuery(), Safety()],
)

print("Evaluation complete. Check MLflow UI for results.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Pre-deployment Validation

# COMMAND ----------

import mlflow

mlflow.models.predict(
    model_uri=logged_agent_info.model_uri, 
    input_data={
        "input": [
            {
                "role": "user",
                "content": "What are the high confidence proteins in the knowledge base?",
            }
        ]
    },
    env_manager="uv",
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Register to Unity Catalog [Optional]

# COMMAND ----------

mlflow.set_registry_uri("databricks-uc")

# model_base_name = "lifesciences_agent"
# UC_MODEL_NAME = f"{catalog}.{schema}.alpha_{model_base_name}"

# Check existing versions before registering
from mlflow.tracking import MlflowClient

client = MlflowClient()

try:
    existing_versions = client.search_model_versions(f"name='{UC_MODEL_NAME}'")
    print(f"Existing model versions: {[v.version for v in existing_versions]}")
except Exception as e:
    print(f"No existing versions found (this is fine for first run): {e}")

# Register the model - this creates a new version
print(f"Registering model from run: {logged_agent_info.run_id}")
uc_registered_model_info = mlflow.register_model(
    model_uri=logged_agent_info.model_uri, name=UC_MODEL_NAME
)

print(f"Model registered: {UC_MODEL_NAME} (version {uc_registered_model_info.version})")
print(f"  Run ID: {logged_agent_info.run_id}")
print(f"  Model URI: {logged_agent_info.model_uri}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## `NEXT`: Evaluate before Deploying the Agent
# MAGIC Evaluate development updates with some evaluation metric to promote model version for deployment

# COMMAND ----------

# MAGIC %md
# MAGIC ## Architecture Summary
# MAGIC
# MAGIC This agent implements an orchestrator-synthesizer pattern with parallel execution using LangGraph's Send API:
# MAGIC
# MAGIC ### Routing Logic
# MAGIC
# MAGIC 1. **SQL Path**: Orchestrator → SQL Worker → END (direct answer, no synthesis)
# MAGIC 2. **Vector Path**: Orchestrator → [Vector Worker 1 + Vector Worker 2] → Synthesizer → END (parallel execution)
