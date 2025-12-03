# Databricks notebook source
# MAGIC %md
# MAGIC # Life Sciences Structured Retrieval Agent with Genie
# MAGIC
# MAGIC This notebook implements an orchestrator-synthesizer-Genie architecture with multiple routing paths:
# MAGIC
# MAGIC **Architecture:**
# MAGIC - **Orchestrator/Supervisor**: 3-way routing (SQL, Direct Genie, or Vector Search)
# MAGIC - **Direct Genie Path**: Well-formed questions go directly to Genie with quality check
# MAGIC - **Vector + Genie Path**: Complex questions get context from vector search before Genie query
# MAGIC - **SQL Worker**: Simple aggregations bypass other processing

# COMMAND ----------

# MAGIC %pip install -U -qqqq langgraph uv databricks-agents databricks-langchain mlflow-skinny[databricks] databricks-vectorsearch databricks-sdk
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# Widgets for configuration
dbutils.widgets.text("catalog", "dbxmetagen", "Catalog")
dbutils.widgets.text("schema", "default", "Schema")
dbutils.widgets.text("sql_warehouse_id", "9f4807aa9999e7f4", "SQL Warehouse ID")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
sql_warehouse_id = dbutils.widgets.get("sql_warehouse_id")

print(f"Using catalog: {catalog}, schema: {schema}")
if sql_warehouse_id:
    print(f"SQL Warehouse ID: {sql_warehouse_id}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Agent Architecture
# MAGIC
# MAGIC ```mermaid
# MAGIC graph TD
# MAGIC     Start([User Query]) --> Orch{Orchestrator}
# MAGIC
# MAGIC     Orch -->|Simple count/avg| SQL[SQL Worker]
# MAGIC     Orch -->|Clear structured query| DirectGenie[Direct Genie]
# MAGIC     Orch -->|Vague + needs data| ContextFetch[Context Fetcher]
# MAGIC     Orch -->|Semantic only| VectorOnly[Vector Search]
# MAGIC
# MAGIC     SQL -->|Can answer| End1([Answer])
# MAGIC     SQL -->|Cannot answer| Orch
# MAGIC
# MAGIC     DirectGenie --> QC{Quality Check}
# MAGIC     QC -->|Pass| End2([Answer])
# MAGIC     QC -->|Fail| ContextFetch
# MAGIC
# MAGIC     ContextFetch -->|Parallel| VW1[Vector Worker 1]
# MAGIC     ContextFetch -->|Parallel| VW2[Vector Worker 2]
# MAGIC     VW1 --> Rewrite[Rewrite with Context]
# MAGIC     VW2 --> Rewrite
# MAGIC     Rewrite --> GenieCtx[Genie with Context]
# MAGIC     GenieCtx --> End3([Answer])
# MAGIC
# MAGIC     VectorOnly -->|Parallel| VW3[Vector Worker 1]
# MAGIC     VectorOnly -->|Parallel| VW4[Vector Worker 2]
# MAGIC     VW3 --> Synth[Synthesizer]
# MAGIC     VW4 --> Synth
# MAGIC     Synth --> End4([Answer])
# MAGIC
# MAGIC     style Orch fill:#e1f5ff
# MAGIC     style SQL fill:#fff4e1
# MAGIC     style DirectGenie fill:#ffe0b2
# MAGIC     style ContextFetch fill:#f8bbd0
# MAGIC     style GenieCtx fill:#ffe0b2
# MAGIC     style VW1 fill:#c8e6c9
# MAGIC     style VW2 fill:#c8e6c9
# MAGIC     style VW3 fill:#c8e6c9
# MAGIC     style VW4 fill:#c8e6c9
# MAGIC     style Synth fill:#f3e5f5
# MAGIC ```
# MAGIC
# MAGIC ## Architecture Flow with Genie Integration
# MAGIC
# MAGIC ```mermaid
# MAGIC graph TD
# MAGIC     Start([User Query]) --> Orchestrator[Orchestrator/Supervisor<br/>3-Way Routing]
# MAGIC
# MAGIC     Orchestrator -->|Well-formed query| DirectGenie[Direct Genie]
# MAGIC     Orchestrator -->|Needs context| VectorWorker1[Vector Worker 1<br/>Genes & Proteins]
# MAGIC     Orchestrator -->|Needs context| VectorWorker2[Vector Worker 2<br/>Pathways & Compounds]
# MAGIC     Orchestrator -->|SQL aggregation| SQLWorker[SQL Worker<br/>UC Functions]
# MAGIC
# MAGIC     DirectGenie --> QualityCheck{Quality Check}
# MAGIC     QualityCheck -->|Good answer| End1([Return Answer])
# MAGIC     QualityCheck -->|Poor answer| Orchestrator
# MAGIC
# MAGIC     VectorWorker1 --> Synthesizer[Synthesizer/Judge<br/>Question Writer]
# MAGIC     VectorWorker2 --> Synthesizer
# MAGIC
# MAGIC     Synthesizer --> SynthDecision{Decision}
# MAGIC     SynthDecision -->|Complete answer| End2([Return Answer])
# MAGIC     SynthDecision -->|Needs Genie| RewriteQuestion[Rewrite with Context]
# MAGIC
# MAGIC     RewriteQuestion --> GenieEnriched[Genie with Context]
# MAGIC     GenieEnriched --> End3([Return Answer])
# MAGIC
# MAGIC     SQLWorker --> End4([Return Answer])
# MAGIC
# MAGIC     style Orchestrator fill:#e1f5ff
# MAGIC     style DirectGenie fill:#fff3e0
# MAGIC     style GenieEnriched fill:#fff3e0
# MAGIC     style VectorWorker1 fill:#e8f5e9
# MAGIC     style VectorWorker2 fill:#e8f5e9
# MAGIC     style Synthesizer fill:#f3e5f5
# MAGIC     style SQLWorker fill:#fff4e1
# MAGIC     style QualityCheck fill:#e0f2f1
# MAGIC     style SynthDecision fill:#f3e5f5
# MAGIC ```
# MAGIC
# MAGIC ### Routing Paths
# MAGIC
# MAGIC 1. **SQL Path**: Orchestrator → SQL Worker → END (or back to Orchestrator if can't answer)
# MAGIC 2. **Direct Genie Path**: Orchestrator → Direct Genie → Quality Check → END (or to Context Fetcher if fails)
# MAGIC 3. **Genie with Context Path**: Orchestrator → Context Fetcher → [Vector Workers (parallel)] → Context Rewriter → Genie → END
# MAGIC 4. **Vector Only Path**: Orchestrator → [Vector Workers (parallel)] → Synthesizer → END

# COMMAND ----------

# MAGIC %%writefile agent_genie.py
# MAGIC from typing import Annotated, Any, Generator, Literal, Optional, Sequence, TypedDict, Union
# MAGIC import json
# MAGIC from operator import add
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
# MAGIC import os
# MAGIC
# MAGIC LLM_ENDPOINT_NAME = "databricks-claude-3-7-sonnet"
# MAGIC CATALOG = os.getenv("CATALOG", "dbxmetagen")
# MAGIC SCHEMA = os.getenv("SCHEMA", "default")
# MAGIC VECTOR_SEARCH_ENDPOINT = "lifesciences_vector_search"
# MAGIC GENIE_SPACE_ID = os.getenv("GENIE_SPACE_ID", "01f0c64ba4c61bd49b1aa03af847407a")
# MAGIC SQL_WAREHOUSE_ID = os.getenv("SQL_WAREHOUSE_ID", "9f4807aa9999e7f4")
# MAGIC
# MAGIC llm = ChatDatabricks(endpoint=LLM_ENDPOINT_NAME)
# MAGIC
# MAGIC ############################################
# MAGIC # Define Tools
# MAGIC ############################################
# MAGIC
# MAGIC # SQL Worker Tools
# MAGIC UC_TOOL_NAMES = [
# MAGIC     f"{CATALOG}.{SCHEMA}.count_genes_knowledge_rows",
# MAGIC     f"{CATALOG}.{SCHEMA}.count_proteins_knowledge_rows",
# MAGIC     f"{CATALOG}.{SCHEMA}.count_pathways_knowledge_rows",
# MAGIC     f"{CATALOG}.{SCHEMA}.count_compounds_knowledge_rows",
# MAGIC     f"{CATALOG}.{SCHEMA}.count_high_confidence_genes_knowledge",
# MAGIC     f"{CATALOG}.{SCHEMA}.count_high_confidence_proteins_knowledge",
# MAGIC     f"{CATALOG}.{SCHEMA}.count_high_confidence_pathways_knowledge",
# MAGIC     f"{CATALOG}.{SCHEMA}.count_high_confidence_compounds_knowledge",
# MAGIC     f"{CATALOG}.{SCHEMA}.avg_confidence_genes_knowledge",
# MAGIC     f"{CATALOG}.{SCHEMA}.avg_confidence_proteins_knowledge",
# MAGIC     f"{CATALOG}.{SCHEMA}.avg_confidence_pathways_knowledge",
# MAGIC     f"{CATALOG}.{SCHEMA}.avg_confidence_compounds_knowledge",
# MAGIC ]
# MAGIC uc_toolkit = UCFunctionToolkit(function_names=UC_TOOL_NAMES)
# MAGIC sql_tools = uc_toolkit.tools
# MAGIC
# MAGIC # Vector Search Tools
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
# MAGIC # Genie Agent
# MAGIC # GenieAgent is invoked directly in the direct_genie_node and genie_with_context_node
# MAGIC from databricks_langchain.genie import GenieAgent
# MAGIC
# MAGIC genie_agent = GenieAgent(
# MAGIC     genie_space_id=GENIE_SPACE_ID,
# MAGIC     genie_agent_name="life_sciences_genie",
# MAGIC     description="Query the Genie AI assistant for structured data analysis questions about life sciences data"
# MAGIC )
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
# MAGIC     return {**left, **right}
# MAGIC
# MAGIC class AgentState(TypedDict):
# MAGIC     messages: Annotated[Sequence[AnyMessage], add_messages]
# MAGIC     custom_inputs: Optional[dict[str, Any]]
# MAGIC     custom_outputs: Optional[dict[str, Any]]
# MAGIC     worker_results: Annotated[Optional[dict[str, Any]], merge_worker_results]
# MAGIC     route_decision: Optional[str]
# MAGIC     genie_attempt_count: Optional[int]
# MAGIC     context_added: Optional[bool]
# MAGIC     synthesizer_decision: Optional[str]
# MAGIC     sql_can_answer: Optional[bool]
# MAGIC
# MAGIC ############################################
# MAGIC # Orchestrator Node
# MAGIC ############################################
# MAGIC
# MAGIC def orchestrator_node(state: AgentState, config: RunnableConfig):
# MAGIC     """
# MAGIC     Orchestrator decides 3-way routing: SQL, Direct Genie, or Vector Search.
# MAGIC     """
# MAGIC     messages = state["messages"]
# MAGIC     last_message = messages[-1]
# MAGIC     genie_attempts = state.get("genie_attempt_count", 0)
# MAGIC
# MAGIC     orchestrator_prompt = """You are an orchestrator for a life sciences knowledge base system with Genie AI integration.
# MAGIC
# MAGIC Analyze the user's question and decide on the best route:
# MAGIC
# MAGIC 1. Route to "sql" if: Simple counts or averages about ONE table (e.g., "How many genes?", "Average confidence for proteins?")
# MAGIC
# MAGIC 2. Route to "genie" if:
# MAGIC    - Question is well-formed, specific, and has enough context for direct Genie query
# MAGIC    - Clear structured queries: "Compare X and Y", "Show me top 10 Z", "What is the total across all tables?"
# MAGIC    - All information needed is in the question itself
# MAGIC
# MAGIC 3. Route to "genie_context" if:
# MAGIC    - Question needs structured data/aggregation BUT is vague or lacks context
# MAGIC    - Questions like "What compounds do we have for kinases?" (needs to know what kinases are first)
# MAGIC    - Asks for data/aggregation about something that needs semantic lookup first
# MAGIC
# MAGIC 4. Route to "vector" if:
# MAGIC    - Question is purely semantic/conceptual (no aggregation needed)
# MAGIC    - Questions like "What are kinases?", "Explain cell signaling", "Tell me about proteins"
# MAGIC
# MAGIC Respond with ONLY one word: "sql", "genie", "genie_context", or "vector"
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
# MAGIC     if decision not in ["sql", "genie", "genie_context", "vector"]:
# MAGIC         decision = "vector"
# MAGIC
# MAGIC     print(f"[Orchestrator Decision: {decision.upper()}]")
# MAGIC
# MAGIC     return {
# MAGIC         "route_decision": decision,
# MAGIC         "genie_attempt_count": genie_attempts,
# MAGIC         "messages": [AIMessage(content=f"[Orchestrator: Routing to {decision}]")]
# MAGIC     }
# MAGIC
# MAGIC ############################################
# MAGIC # SQL Worker Node
# MAGIC ############################################
# MAGIC
# MAGIC def sql_worker_node(state: AgentState, config: RunnableConfig):
# MAGIC     """SQL worker handles aggregation queries using UC functions."""
# MAGIC     messages = state["messages"]
# MAGIC     user_message = [m for m in messages if isinstance(m, HumanMessage)][-1]
# MAGIC
# MAGIC     sql_prompt = f"""You are a SQL worker for a life sciences knowledge base.
# MAGIC
# MAGIC Available UC functions for each table (genes/proteins/pathways/compounds):
# MAGIC - count_<table>_rows(): Count total rows
# MAGIC - count_high_confidence_<table>(min_confidence): Count entries above confidence threshold
# MAGIC - avg_confidence_<table>(): Get average confidence
# MAGIC
# MAGIC If you can answer using these functions, use them. If the question is beyond these simple functions, respond with: "CANNOT_ANSWER"
# MAGIC """
# MAGIC
# MAGIC     sql_llm = llm.bind_tools(sql_tools)
# MAGIC     sql_messages = [SystemMessage(content=sql_prompt), user_message]
# MAGIC
# MAGIC     response = sql_llm.invoke(sql_messages)
# MAGIC     result_messages = [response]
# MAGIC
# MAGIC     # Check if SQL worker can handle this
# MAGIC     can_answer = True
# MAGIC     if response.tool_calls:
# MAGIC         tool_node = ToolNode(sql_tools)
# MAGIC         tool_results = tool_node.invoke({"messages": [response]})
# MAGIC         result_messages.extend(tool_results["messages"])
# MAGIC
# MAGIC         final_response = llm.invoke(sql_messages + result_messages)
# MAGIC         result_messages.append(final_response)
# MAGIC         final_answer = final_response.content
# MAGIC     else:
# MAGIC         final_answer = response.content
# MAGIC
# MAGIC     # Check if SQL worker couldn't answer
# MAGIC     if "CANNOT_ANSWER" in final_answer:
# MAGIC         can_answer = False
# MAGIC         final_answer = "[SQL Worker: Cannot handle this query, routing back to orchestrator]"
# MAGIC
# MAGIC     return {
# MAGIC         "sql_can_answer": can_answer,
# MAGIC         "messages": [AIMessage(content=final_answer)]
# MAGIC     }
# MAGIC
# MAGIC ############################################
# MAGIC # Context Fetcher Node (triggers parallel vector search)
# MAGIC ############################################
# MAGIC
# MAGIC def context_fetcher_node(state: AgentState, config: RunnableConfig):
# MAGIC     """
# MAGIC     Context fetcher node that triggers parallel vector search.
# MAGIC     Sets a flag so vector workers know to route to context_rewriter instead of synthesizer.
# MAGIC     """
# MAGIC     print("[Context Fetcher: Triggering parallel vector search for context]")
# MAGIC     return {
# MAGIC         "context_added": True,  # Flag to indicate we're in context-fetching mode
# MAGIC         "messages": [AIMessage(content="[Fetching context from vector search...]")]
# MAGIC     }
# MAGIC
# MAGIC ############################################
# MAGIC # Context Rewriter Node (after vector workers complete)
# MAGIC ############################################
# MAGIC
# MAGIC def context_rewriter_node(state: AgentState, config: RunnableConfig):
# MAGIC     """
# MAGIC     Rewrites the original question with context from vector search workers.
# MAGIC     Extracts structured information from documents and passes to Genie.
# MAGIC     """
# MAGIC     messages = state["messages"]
# MAGIC     user_message = [m for m in messages if isinstance(m, HumanMessage)][-1]
# MAGIC     worker_results = state.get("worker_results", {})
# MAGIC
# MAGIC     # Collect full vector search results without manual parsing
# MAGIC     context_parts = []
# MAGIC     for key, val in worker_results.items():
# MAGIC         source = val.get('source', key)
# MAGIC         result = val.get('result', '')
# MAGIC         context_parts.append(f"{source}:\n{result}")
# MAGIC
# MAGIC     context_summary = "\n\n".join(context_parts)
# MAGIC
# MAGIC     rewriter_prompt = """You are a question rewriter for a Genie AI system. You have full vector search results about the user's question.
# MAGIC
# MAGIC Your task:
# MAGIC 1. Review the complete vector search context provided
# MAGIC 2. Extract relevant entity IDs, names, and attributes from the results
# MAGIC 3. Rewrite the question to be explicit and specific for Genie
# MAGIC 4. Include concrete entity identifiers found in the search results
# MAGIC 5. Make it a clear structured data query that Genie can answer directly
# MAGIC
# MAGIC Example:
# MAGIC - Original: "What compounds do we have for kinases?"
# MAGIC - Context includes: COMP_0048, COMP_0052 (compounds targeting kinases)
# MAGIC - Rewritten: "Show me data for compounds COMP_0048, COMP_0052, COMP_0012 that target kinases, including their confidence scores and bioavailability"
# MAGIC """
# MAGIC
# MAGIC     rewriter_input = f"""Original Question: {user_message.content}
# MAGIC
# MAGIC Complete Vector Search Results:
# MAGIC {context_summary}
# MAGIC
# MAGIC Rewrite the question to be specific, include entity IDs/names found, and make it answerable by Genie:"""
# MAGIC
# MAGIC     response = llm.invoke([
# MAGIC         SystemMessage(content=rewriter_prompt),
# MAGIC         HumanMessage(content=rewriter_input)
# MAGIC     ])
# MAGIC     enriched_question = response.content
# MAGIC
# MAGIC     print(f"[Context Rewriter: Creating enriched question for Genie]")
# MAGIC     print(f"  Original: {user_message.content[:100]}")
# MAGIC     print(f"  Context Summary Length: {len(context_summary)} chars")
# MAGIC     print(f"  Context Preview: {context_summary[:200]}...")
# MAGIC     print(f"  Enriched: {enriched_question[:150]}...")
# MAGIC
# MAGIC     return {
# MAGIC         "messages": [HumanMessage(content=enriched_question)],
# MAGIC         "context_added": True
# MAGIC     }
# MAGIC
# MAGIC ############################################
# MAGIC # Direct Genie Node
# MAGIC ############################################
# MAGIC
# MAGIC def direct_genie_node(state: AgentState, config: RunnableConfig):
# MAGIC     """
# MAGIC     Direct Genie call for well-formed questions.
# MAGIC     """
# MAGIC     messages = state["messages"]
# MAGIC     user_message = [m for m in messages if isinstance(m, HumanMessage)][-1]
# MAGIC     genie_attempts = state.get("genie_attempt_count", 0)
# MAGIC
# MAGIC     # Invoke Genie agent directly with the user's message
# MAGIC     genie_response = genie_agent.invoke({"messages": [user_message]})
# MAGIC
# MAGIC     # Extract result from Genie response
# MAGIC     if "messages" in genie_response:
# MAGIC         genie_result = genie_response["messages"][-1].content if genie_response["messages"] else "No results from Genie."
# MAGIC     else:
# MAGIC         genie_result = str(genie_response)
# MAGIC
# MAGIC     return {
# MAGIC         "messages": [AIMessage(content=f"[Direct Genie Result]\n{genie_result}")],
# MAGIC         "genie_attempt_count": genie_attempts + 1
# MAGIC     }
# MAGIC
# MAGIC ############################################
# MAGIC # Quality Check Node
# MAGIC ############################################
# MAGIC
# MAGIC def quality_check_node(state: AgentState, config: RunnableConfig):
# MAGIC     """
# MAGIC     Evaluate quality of Genie response and decide whether to return or retry.
# MAGIC     """
# MAGIC     messages = state["messages"]
# MAGIC     last_message = messages[-1]
# MAGIC     genie_attempts = state.get("genie_attempt_count", 0)
# MAGIC
# MAGIC     quality_prompt = """You are a quality checker for Genie AI responses.
# MAGIC
# MAGIC Evaluate the Genie result and determine if it adequately answers the user's question.
# MAGIC
# MAGIC Consider:
# MAGIC - Does it contain actual data, numbers, tables, or structured results?
# MAGIC - Does it directly address the question asked?
# MAGIC - Tables, comparisons, statistics, and data visualizations are GOOD responses
# MAGIC
# MAGIC Respond with ONLY: "pass" if the response contains relevant data/results, "fail" if it's just an error message or completely empty
# MAGIC """
# MAGIC
# MAGIC     quality_messages = [
# MAGIC         SystemMessage(content=quality_prompt),
# MAGIC         HumanMessage(content=f"Genie result to evaluate:\n{last_message.content}")
# MAGIC     ]
# MAGIC
# MAGIC     response = llm.invoke(quality_messages)
# MAGIC     quality = response.content.strip().lower()
# MAGIC
# MAGIC     # Only allow one retry to avoid loops
# MAGIC     if "fail" in quality and genie_attempts < 2:
# MAGIC         decision = "retry"
# MAGIC     else:
# MAGIC         decision = "pass"
# MAGIC
# MAGIC     return {
# MAGIC         "route_decision": decision,
# MAGIC         "messages": [AIMessage(content=f"[Quality Check: {decision}]")]
# MAGIC     }
# MAGIC
# MAGIC ############################################
# MAGIC # Vector Worker Nodes
# MAGIC ############################################
# MAGIC
# MAGIC def vector_worker_1_node(state: AgentState, config: RunnableConfig):
# MAGIC     """Vector worker 1: Searches genes and proteins indexes."""
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
# MAGIC     # Return worker results for merging
# MAGIC     return {
# MAGIC         "worker_results": {
# MAGIC             "vector_worker_1": {
# MAGIC                 "result": final_result,
# MAGIC                 "confidence": 0.80,
# MAGIC                 "source": "Vector Worker 1 (Genes & Proteins)",
# MAGIC                 "tool_calls_made": len(response.tool_calls) if response.tool_calls else 0
# MAGIC             }
# MAGIC         },
# MAGIC         "messages": [AIMessage(content=f"[Vector Worker 1 completed with {len(search_results)} results]")]
# MAGIC     }
# MAGIC
# MAGIC def vector_worker_2_node(state: AgentState, config: RunnableConfig):
# MAGIC     """Vector worker 2: Searches pathways and compounds indexes."""
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
# MAGIC     # Return worker results for merging
# MAGIC     return {
# MAGIC         "worker_results": {
# MAGIC             "vector_worker_2": {
# MAGIC                 "result": final_result,
# MAGIC                 "confidence": 0.80,
# MAGIC                 "source": "Vector Worker 2 (Pathways & Compounds)",
# MAGIC                 "tool_calls_made": len(response.tool_calls) if response.tool_calls else 0
# MAGIC             }
# MAGIC         },
# MAGIC         "messages": [AIMessage(content=f"[Vector Worker 2 completed with {len(search_results)} results]")]
# MAGIC     }
# MAGIC
# MAGIC ############################################
# MAGIC # Synthesizer/Judge/Question-Writer Node
# MAGIC ############################################
# MAGIC
# MAGIC def synthesizer_node(state: AgentState, config: RunnableConfig):
# MAGIC     """
# MAGIC     Synthesizer combines all worker results into a coherent synthesis.
# MAGIC     For vector-only path, provides final answer directly.
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
# MAGIC     system_prompt = """You are a synthesizer agent for a life sciences knowledge system.
# MAGIC
# MAGIC You have received results from multiple specialized workers. Your job is to:
# MAGIC 1. Combine the information from all workers into a coherent narrative
# MAGIC 2. Integrate findings across different knowledge bases
# MAGIC 3. Present a comprehensive synthesis
# MAGIC
# MAGIC If the vector search results are insufficient or empty, respond with:
# MAGIC "I was unable to find sufficient information to answer your question. Please try rephrasing with more specific details."
# MAGIC
# MAGIC Otherwise, be scientific and precise in your synthesis.
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
# MAGIC # Genie with Context Node
# MAGIC ############################################
# MAGIC
# MAGIC def genie_with_context_node(state: AgentState, config: RunnableConfig):
# MAGIC     """
# MAGIC     Call Genie with context-enriched question (already rewritten by context_rewriter_node).
# MAGIC     """
# MAGIC     print("[Genie with Context Node: Calling Genie AI]")
# MAGIC     messages = state["messages"]
# MAGIC     # The last message is the rewritten question from context_rewriter
# MAGIC     enriched_question = messages[-1]
# MAGIC
# MAGIC     # Invoke Genie agent with context-enriched question
# MAGIC     genie_response = genie_agent.invoke({"messages": [enriched_question]})
# MAGIC
# MAGIC     # Extract result from Genie response
# MAGIC     if "messages" in genie_response:
# MAGIC         genie_result = genie_response["messages"][-1].content if genie_response["messages"] else "No results from Genie."
# MAGIC     else:
# MAGIC         genie_result = str(genie_response)
# MAGIC
# MAGIC     return {
# MAGIC         "messages": [AIMessage(content=f"[Genie with Context Result]\n{genie_result}")]
# MAGIC     }
# MAGIC
# MAGIC ############################################
# MAGIC # Routing Logic
# MAGIC ############################################
# MAGIC
# MAGIC def route_from_orchestrator(state: AgentState):
# MAGIC     """Route from orchestrator: SQL, Direct Genie, Genie with Context, or Vector-only."""
# MAGIC     decision = state.get("route_decision", "vector")
# MAGIC
# MAGIC     if decision == "sql":
# MAGIC         return "sql_worker"
# MAGIC     elif decision == "genie":
# MAGIC         return "direct_genie"
# MAGIC     elif decision == "genie_context":
# MAGIC         # Route to context fetcher which will trigger parallel vector search
# MAGIC         return "context_fetcher"
# MAGIC     else:
# MAGIC         # Pure vector search (no Genie after)
# MAGIC         return [
# MAGIC             Send("vector_worker_1", state),
# MAGIC             Send("vector_worker_2", state),
# MAGIC         ]
# MAGIC
# MAGIC def route_after_quality_check(state: AgentState):
# MAGIC     """Route after quality check: END or to context fetcher."""
# MAGIC     decision = state.get("route_decision", "pass")
# MAGIC
# MAGIC     if decision == "retry":
# MAGIC         # Failed Genie attempt - get context and try again with enriched question
# MAGIC         print("[Quality Check: Failed, routing to context fetcher]")
# MAGIC         return "context_fetcher"
# MAGIC     else:
# MAGIC         return END
# MAGIC
# MAGIC def route_from_sql_worker(state: AgentState):
# MAGIC     """Route from SQL worker: END if answered, back to orchestrator if not."""
# MAGIC     can_answer = state.get("sql_can_answer", True)
# MAGIC
# MAGIC     if can_answer:
# MAGIC         return END
# MAGIC     else:
# MAGIC         print("[SQL Worker: Cannot answer, routing back to orchestrator]")
# MAGIC         return "orchestrator"
# MAGIC
# MAGIC def route_from_context_fetcher(state: AgentState):
# MAGIC     """Route from context fetcher to parallel vector workers."""
# MAGIC     return [
# MAGIC         Send("vector_worker_1", state),
# MAGIC         Send("vector_worker_2", state),
# MAGIC     ]
# MAGIC
# MAGIC def route_from_vector_workers(state: AgentState):
# MAGIC     """Route from vector workers: to context_rewriter if context mode, else synthesizer."""
# MAGIC     context_mode = state.get("context_added", False)
# MAGIC
# MAGIC     if context_mode:
# MAGIC         # We're in the genie_context path - go to context_rewriter
# MAGIC         return "context_rewriter"
# MAGIC     else:
# MAGIC         # We're in the vector-only path - go to synthesizer
# MAGIC         return "synthesizer"
# MAGIC
# MAGIC def route_after_synthesizer(state: AgentState):
# MAGIC     """Route after synthesizer: just END (synthesizer handles vector-only queries)."""
# MAGIC     return END
# MAGIC
# MAGIC ############################################
# MAGIC # Build the Graph
# MAGIC ############################################
# MAGIC
# MAGIC def create_genie_agent():
# MAGIC     """Create the orchestrator-synthesizer-Genie agent graph."""
# MAGIC     workflow = StateGraph(AgentState)
# MAGIC
# MAGIC     # Add all nodes
# MAGIC     workflow.add_node("orchestrator", orchestrator_node)
# MAGIC     workflow.add_node("sql_worker", sql_worker_node)
# MAGIC     workflow.add_node("direct_genie", direct_genie_node)
# MAGIC     workflow.add_node("quality_check", quality_check_node)
# MAGIC     workflow.add_node("context_fetcher", context_fetcher_node)
# MAGIC     workflow.add_node("context_rewriter", context_rewriter_node)
# MAGIC     workflow.add_node("vector_worker_1", vector_worker_1_node)
# MAGIC     workflow.add_node("vector_worker_2", vector_worker_2_node)
# MAGIC     workflow.add_node("synthesizer", synthesizer_node)
# MAGIC     workflow.add_node("genie_with_context", genie_with_context_node)
# MAGIC
# MAGIC     # Define flow
# MAGIC     workflow.set_entry_point("orchestrator")
# MAGIC
# MAGIC     # Orchestrator routes to: sql_worker, direct_genie, context_fetcher, or [vector_worker_1, vector_worker_2]
# MAGIC     workflow.add_conditional_edges("orchestrator", route_from_orchestrator)
# MAGIC
# MAGIC     # SQL worker can route back to orchestrator or END
# MAGIC     workflow.add_conditional_edges("sql_worker", route_from_sql_worker)
# MAGIC
# MAGIC     # Direct Genie goes to quality check
# MAGIC     workflow.add_edge("direct_genie", "quality_check")
# MAGIC
# MAGIC     # Quality check: END or to context_fetcher
# MAGIC     workflow.add_conditional_edges("quality_check", route_after_quality_check)
# MAGIC
# MAGIC     # Context fetcher fans out to vector workers (parallel)
# MAGIC     workflow.add_conditional_edges("context_fetcher", route_from_context_fetcher)
# MAGIC
# MAGIC     # Vector workers route based on context: to context_rewriter or synthesizer
# MAGIC     workflow.add_conditional_edges("vector_worker_1", route_from_vector_workers)
# MAGIC     workflow.add_conditional_edges("vector_worker_2", route_from_vector_workers)
# MAGIC
# MAGIC     # Context rewriter goes to Genie with context
# MAGIC     workflow.add_edge("context_rewriter", "genie_with_context")
# MAGIC
# MAGIC     # Genie with context goes to END
# MAGIC     workflow.add_edge("genie_with_context", END)
# MAGIC
# MAGIC     # Synthesizer (from vector-only path) goes to END
# MAGIC     workflow.add_conditional_edges("synthesizer", route_after_synthesizer)
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
# MAGIC agent = create_genie_agent()
# MAGIC AGENT = LangGraphResponsesAgent(agent)
# MAGIC mlflow.models.set_model(AGENT)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Test the Agent

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# Re-fetch widget values after Python restart
catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
sql_warehouse_id = dbutils.widgets.get("sql_warehouse_id")

print(f"Using catalog: {catalog}, schema: {schema}")
if sql_warehouse_id:
    print(f"SQL Warehouse ID: {sql_warehouse_id}")
    import os

    os.environ["SQL_WAREHOUSE_ID"] = sql_warehouse_id
    os.environ["CATALOG"] = catalog
    os.environ["SCHEMA"] = schema

# Import the agent
from agent_genie import AGENT

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

# Test 2: Direct Genie - well-formed question
print("=" * 80)
print("Test 2: Direct Genie (Well-formed Query)")
print("=" * 80)
result2 = AGENT.predict(
    {
        "input": [
            {
                "role": "user",
                "content": "What is the average confidence score across all tables in the life sciences database?",
            }
        ]
    }
)
print(result2.model_dump(exclude_none=True))

# COMMAND ----------

# Test 3: Genie with Context path - vague aggregation question
print("=" * 80)
print("Test 3: Genie with Context (Vague aggregation → Vector → Rewrite → Genie)")
print("=" * 80)
result3 = AGENT.predict(
    {
        "input": [
            {
                "role": "user",
                "content": "What compounds do we have for kinases? Show me the data with confidence scores.",
            }
        ]
    }
)
print(result3.model_dump(exclude_none=True))

# COMMAND ----------

# Test 4: Another Genie test - comparison query
print("=" * 80)
print("Test 4: Structured Comparison Query (Should Route to Genie)")
print("=" * 80)
result4 = AGENT.predict(
    {
        "input": [
            {
                "role": "user",
                "content": "Compare the average confidence scores between genes and proteins in the database. Which category has higher confidence?",
            }
        ]
    }
)
print(result4.model_dump(exclude_none=True))

# COMMAND ----------

# Test 5: Vector only path - semantic answer sufficient
print("=" * 80)
print("Test 5: Vector Search Only (Complete Answer)")
print("=" * 80)
result5 = AGENT.predict(
    {
        "input": [
            {
                "role": "user",
                "content": "What are the main functions of proteins involved in cell signaling?",
            }
        ]
    }
)
print(result5.model_dump(exclude_none=True))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Log the Agent as an MLflow Model

# COMMAND ----------

from agent_genie import (
    UC_TOOL_NAMES,
    VECTOR_SEARCH_TOOLS,
    GENIE_SPACE_ID,
    CATALOG,
    SCHEMA,
    SQL_WAREHOUSE_ID,
)
import mlflow
from mlflow.models.resources import (
    DatabricksFunction,
    DatabricksTable,
    DatabricksGenieSpace,
    DatabricksSQLWarehouse,
)
from pkg_resources import get_distribution

# Collect all resources for automatic auth passthrough
resources = []

# Add Genie space - enables the agent to query Genie for structured data analysis
# Genie space must have access to the base tables listed below
resources.append(DatabricksGenieSpace(genie_space_id=GENIE_SPACE_ID))

# Add SQL Warehouse - required for Genie to execute SQL queries
if SQL_WAREHOUSE_ID:
    resources.append(DatabricksSQLWarehouse(warehouse_id=SQL_WAREHOUSE_ID))
else:
    print(
        "WARNING: SQL_WAREHOUSE_ID not set. Genie agent may not function properly without a SQL warehouse."
    )

# Add base Delta tables (these are what Genie queries)
for table in [
    "genes_knowledge",
    "proteins_knowledge",
    "pathways_knowledge",
    "compounds_knowledge",
]:
    resources.append(DatabricksTable(table_name=f"{CATALOG}.{SCHEMA}.{table}"))

# Add vector search tools
for tool in VECTOR_SEARCH_TOOLS:
    resources.extend(tool.resources)

# Add UC functions
for tool_name in UC_TOOL_NAMES:
    resources.append(DatabricksFunction(function_name=tool_name))

# Log the model
with mlflow.start_run():
    logged_agent_info = mlflow.pyfunc.log_model(
        name="agent",
        python_model="agent_genie.py",
        pip_requirements=[
            "databricks-langchain",
            "databricks-vectorsearch",
            "databricks-sdk",
            "psycopg[binary]",
            f"langgraph=={get_distribution('langgraph').version}",
            f"databricks-connect=={get_distribution('databricks-connect').version}",
        ],
        resources=resources,
    )

print(f"Agent logged: {logged_agent_info.model_uri}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Evaluate the Agent

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
                {
                    "role": "user",
                    "content": "What is the average confidence for proteins?",
                }
            ]
        },
        "expected_response": "The average confidence can be calculated using the UC function.",
    },
    {
        "inputs": {
            "input": [{"role": "user", "content": "Tell me about kinase proteins."}]
        },
        "expected_response": "Kinases are proteins that catalyze phosphorylation reactions.",
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

mlflow.models.predict(
    model_uri=f"runs:/{logged_agent_info.run_id}/agent",
    input_data={
        "input": [
            {
                "role": "user",
                "content": "What are the pathways involved in metabolism?",
            }
        ]
    },
    env_manager="uv",
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Register to Unity Catalog

# COMMAND ----------

mlflow.set_registry_uri("databricks-uc")

model_name = "lifesciences_genie_agent"
UC_MODEL_NAME = f"{catalog}.{schema}.{model_name}"

# Check existing versions
from mlflow.tracking import MlflowClient

client = MlflowClient()

try:
    existing_versions = client.search_model_versions(f"name='{UC_MODEL_NAME}'")
    print(f"Existing model versions: {[v.version for v in existing_versions]}")
except Exception as e:
    print(f"No existing versions found: {e}")

# Register the model
print(f"Registering model from run: {logged_agent_info.run_id}")
uc_registered_model_info = mlflow.register_model(
    model_uri=logged_agent_info.model_uri, name=UC_MODEL_NAME
)

print(f"Model registered: {UC_MODEL_NAME} (version {uc_registered_model_info.version})")
print(f"  Run ID: {logged_agent_info.run_id}")
print(f"  Model URI: {logged_agent_info.model_uri}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Deploy the Agent

# COMMAND ----------

# DEPLOYMENT CODE TEMPORARILY COMMENTED OUT
# Use notebook 05_deploy_agent.py to deploy the model

# from databricks import agents
# from databricks.sdk import WorkspaceClient
# import time

# w = WorkspaceClient()

# # Check if endpoint already exists
# endpoint_name = f"agents_{UC_MODEL_NAME.replace('.', '-')}"
# print(f"\n=== Deployment Information ===")
# print(f"Model: {UC_MODEL_NAME}")
# print(f"Version to deploy: {uc_registered_model_info.version}")
# print(f"Endpoint name: {endpoint_name}")

# try:
#     existing_endpoint = w.serving_endpoints.get(endpoint_name)
#     print(f"\nExisting endpoint found:")
#     print(
#         f"  State: {existing_endpoint.state.ready if existing_endpoint.state else 'Unknown'}"
#     )

#     is_failed = False
#     endpoint_state = (
#         existing_endpoint.state.config_update if existing_endpoint.state else None
#     )

#     if (
#         endpoint_state == "UPDATE_FAILED"
#         or existing_endpoint.state.ready == "NOT_READY"
#     ):
#         is_failed = True

#     if existing_endpoint.config is None:
#         print(f"\nEndpoint config is None - this causes the auto_capture_config error")
#         is_failed = True

#     if is_failed:
#         print(
#             f"\nEndpoint {endpoint_name} is in failed/bad state. Deleting and recreating..."
#         )
#         w.serving_endpoints.delete(endpoint_name)
#         print(f"  Waiting 60 seconds for deletion to complete...")
#         time.sleep(60)
#         print("Endpoint deleted - fresh endpoint will be created")
# except Exception as e:
#     if "RESOURCE_DOES_NOT_EXIST" not in str(e):
#         print(f"Note checking endpoint: {e}")
#     else:
#         print("No existing endpoint found - will create new one")

# # Deploy with retry logic
# max_retries = 3
# retry_delay = 60

# print(f"\n=== Starting Deployment ===")
# for attempt in range(max_retries):
#     try:
#         print(f"\nDeploying agent (attempt {attempt + 1}/{max_retries})...")
#         print(f"  Model: {UC_MODEL_NAME}")
#         print(f"  Version: {uc_registered_model_info.version}")

#         deployment_info = agents.deploy(
#             UC_MODEL_NAME,
#             uc_registered_model_info.version,
#             tags={
#                 "architecture": "orchestrator-synthesizer-genie",
#                 "domain": "life-sciences",
#             },
#             deploy_feedback_model=False,
#         )

#         print(f"\nAgent deployed successfully.")
#         print(f"  Model: {UC_MODEL_NAME}")
#         print(f"  Version: {uc_registered_model_info.version}")
#         print(f"  Endpoint: {endpoint_name}")

#         deployed_endpoint = w.serving_endpoints.get(endpoint_name)
#         if deployed_endpoint.config and deployed_endpoint.config.served_entities:
#             print(f"\nCurrently serving:")
#             for entity in deployed_endpoint.config.served_entities:
#                 print(
#                     f"  - Version {entity.entity_version} (scale_to_zero: {entity.scale_to_zero_enabled})"
#                 )

#         break

#     except AttributeError as e:
#         if "auto_capture_config" in str(e) and attempt < max_retries - 1:
#             print(
#                 f"Deployment failed with config error. Retrying in {retry_delay} seconds..."
#             )
#             time.sleep(retry_delay)
#         else:
#             print(f"Deployment failed after {attempt + 1} attempts.")
#             print(f"Error: {e}")
#             print("\nTroubleshooting:")
#             print(
#                 f"1. Check endpoint status: w.serving_endpoints.get('{endpoint_name}')"
#             )
#             print(
#                 f"2. Try deleting endpoint manually: w.serving_endpoints.delete('{endpoint_name}')"
#             )
#             print(f"3. Wait a few minutes and re-run this cell")
#             raise
#     except Exception as e:
#         print(f"Deployment failed: {e}")
#         if attempt < max_retries - 1:
#             print(f"Retrying in {retry_delay} seconds...")
#             time.sleep(retry_delay)
#         else:
#             raise

print("\n=== Model Registered Successfully ===")
print(f"Model: {UC_MODEL_NAME}")
print(f"Version: {uc_registered_model_info.version}")
print(f"\nTo evaluate and deploy:")
print(f"  1. Run notebook 04_evaluate_and_promote.py")
print(f"  2. Run notebook 05_deploy_agent.py")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Architecture Summary
# MAGIC
# MAGIC This agent implements a flexible orchestrator pattern with 4 distinct routing paths:
# MAGIC
# MAGIC ### 1. SQL Path (Simple Aggregations)
# MAGIC - For simple counts/averages on single tables
# MAGIC - Orchestrator → SQL Worker → END (or back to Orchestrator if can't answer)
# MAGIC
# MAGIC ### 2. Direct Genie Path (Clear Structured Queries)
# MAGIC - For well-formed questions with all context included
# MAGIC - Orchestrator → Direct Genie → Quality Check → END (or to Context Fetcher if fails)
# MAGIC
# MAGIC ### 3. Genie with Context Path (Vague Aggregations)
# MAGIC - For questions needing both semantic lookup AND structured queries
# MAGIC - Orchestrator → Context Fetcher → [Vector Workers (parallel)] → Context Rewriter → Genie → END
# MAGIC - Example: "What compounds do we have for kinases? Show me the data."
# MAGIC
# MAGIC ### 4. Vector Only Path (Pure Semantic Questions)
# MAGIC - For conceptual/explanatory questions with no aggregation needed
# MAGIC - Orchestrator → [Vector Workers (parallel)] → Synthesizer → END
# MAGIC
# MAGIC ### Key Features
# MAGIC - Context Fetcher sets flag so vector workers know which path to take
# MAGIC - SQL worker can fail gracefully and re-route
# MAGIC - Quality checker accepts structured data (tables, comparisons)
# MAGIC - Fallback messages when unable to answer
