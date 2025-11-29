from typing import Annotated, Any, Generator, Literal, Optional, Sequence, TypedDict, Union
import json
from operator import add

import mlflow
from databricks_langchain import ChatDatabricks, UCFunctionToolkit, VectorSearchRetrieverTool
from langchain.messages import AIMessage, AIMessageChunk, AnyMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig, RunnableLambda
from langchain_core.tools import BaseTool
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt.tool_node import ToolNode
from langgraph.types import Send
from mlflow.pyfunc import ResponsesAgent
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
    ResponsesAgentStreamEvent,
    output_to_responses_items_stream,
    to_chat_completions_input,
)

############################################
# Configuration
############################################
LLM_ENDPOINT_NAME = "databricks-claude-3-7-sonnet"
# CATALOG = "dbxmetagen"
# SCHEMA = "default"
# VECTOR_SEARCH_ENDPOINT = "lifesciences_vector_search"

## update #cant' use widgets... ?
CATALOG = "mmt"
SCHEMA = "LS_agent"
VECTOR_SEARCH_ENDPOINT = "ls_vs_mmt"

llm = ChatDatabricks(endpoint=LLM_ENDPOINT_NAME)

############################################
# Define Tools
############################################

# SQL Worker Tools - UC Functions for aggregations
# We have specific functions for each table since UC SQL functions can't use dynamic table names
UC_TOOL_NAMES = [
    # Count functions
    f"{CATALOG}.{SCHEMA}.count_genes_knowledge_rows",
    f"{CATALOG}.{SCHEMA}.count_proteins_knowledge_rows",
    f"{CATALOG}.{SCHEMA}.count_pathways_knowledge_rows",
    f"{CATALOG}.{SCHEMA}.count_compounds_knowledge_rows",
    # High confidence count functions
    f"{CATALOG}.{SCHEMA}.count_high_confidence_genes_knowledge",
    f"{CATALOG}.{SCHEMA}.count_high_confidence_proteins_knowledge",
    f"{CATALOG}.{SCHEMA}.count_high_confidence_pathways_knowledge",
    f"{CATALOG}.{SCHEMA}.count_high_confidence_compounds_knowledge",
    # Average confidence functions
    f"{CATALOG}.{SCHEMA}.avg_confidence_genes_knowledge",
    f"{CATALOG}.{SCHEMA}.avg_confidence_proteins_knowledge",
    f"{CATALOG}.{SCHEMA}.avg_confidence_pathways_knowledge",
    f"{CATALOG}.{SCHEMA}.avg_confidence_compounds_knowledge",
]
uc_toolkit = UCFunctionToolkit(function_names=UC_TOOL_NAMES)
sql_tools = uc_toolkit.tools

# Vector Search Tools - Four knowledge bases
VECTOR_SEARCH_TOOLS = [
    VectorSearchRetrieverTool(
        index_name=f"{CATALOG}.{SCHEMA}.genes_knowledge_vs_index",
        columns=["id", "name", "description", "confidence"],
        num_results=3,
        name="search_genes",
        description="Search the genes knowledge base for information about genes, their functions, and associated diseases."
    ),
    VectorSearchRetrieverTool(
        index_name=f"{CATALOG}.{SCHEMA}.proteins_knowledge_vs_index",
        columns=["id", "name", "description", "confidence"],
        num_results=3,
        name="search_proteins",
        description="Search the proteins knowledge base for information about protein structures, functions, and modifications."
    ),
    VectorSearchRetrieverTool(
        index_name=f"{CATALOG}.{SCHEMA}.pathways_knowledge_vs_index",
        columns=["id", "name", "description", "confidence"],
        num_results=3,
        name="search_pathways",
        description="Search the pathways knowledge base for information about biological pathways, metabolic processes, and signaling cascades."
    ),
    VectorSearchRetrieverTool(
        index_name=f"{CATALOG}.{SCHEMA}.compounds_knowledge_vs_index",
        columns=["id", "name", "description", "confidence"],
        num_results=3,
        name="search_compounds",
        description="Search the compounds knowledge base for information about drugs, small molecules, and therapeutic agents."
    ),
]

############################################
# Agent State
############################################

def merge_worker_results(left: Optional[dict], right: Optional[dict]) -> dict:
    """Merge worker results from parallel execution."""
    if left is None:
        return right or {}
    if right is None:
        return left
    # Merge dictionaries - each worker adds its own key
    return {**left, **right}

class AgentState(TypedDict):
    messages: Annotated[Sequence[AnyMessage], add_messages]
    custom_inputs: Optional[dict[str, Any]]
    custom_outputs: Optional[dict[str, Any]]
    worker_results: Annotated[Optional[dict[str, Any]], merge_worker_results]
    route_decision: Optional[str]

############################################
# Orchestrator Node
############################################

def orchestrator_node(state: AgentState, config: RunnableConfig):
    """
    Orchestrator decides whether to route to SQL worker or vector workers.
    """
    messages = state["messages"]
    last_message = messages[-1]

    orchestrator_prompt = """You are an orchestrator for a life sciences knowledge base system.

Analyze the user's question and decide:
- Route to "sql" if the question is asking for simple counts, aggregations, or statistics about tables (e.g., "How many genes?", "What's the average confidence?", "Count of high confidence proteins")
- Route to "vector" for all other questions that require semantic search of knowledge content

Respond with ONLY one word: either "sql" or "vector"
"""

    decision_messages = [
        SystemMessage(content=orchestrator_prompt),
        last_message
    ]

    response = llm.invoke(decision_messages)
    decision = response.content.strip().lower()

    # Default to vector if unclear
    if "sql" not in decision and "vector" not in decision:
        decision = "vector"
    elif "sql" in decision:
        decision = "sql"
    else:
        decision = "vector"

    return {
        "route_decision": decision,
        "messages": [AIMessage(content=f"[Orchestrator: Routing to {decision} worker]")]
    }

############################################
# SQL Worker Node
############################################

def sql_worker_node(state: AgentState, config: RunnableConfig):
    """
    SQL worker handles aggregation queries using UC functions.
    """
    messages = state["messages"]
    user_message = [m for m in messages if isinstance(m, HumanMessage)][-1]

    sql_prompt = f"""You are a SQL worker for a life sciences knowledge base.

Available tables and UC functions:

For genes_knowledge table:
- count_genes_knowledge_rows(): Count total rows
- count_high_confidence_genes_knowledge(min_confidence): Count entries above confidence threshold
- avg_confidence_genes_knowledge(): Get average confidence

For proteins_knowledge table:
- count_proteins_knowledge_rows(): Count total rows
- count_high_confidence_proteins_knowledge(min_confidence): Count entries above confidence threshold
- avg_confidence_proteins_knowledge(): Get average confidence

For pathways_knowledge table:
- count_pathways_knowledge_rows(): Count total rows
- count_high_confidence_pathways_knowledge(min_confidence): Count entries above confidence threshold
- avg_confidence_pathways_knowledge(): Get average confidence

For compounds_knowledge table:
- count_compounds_knowledge_rows(): Count total rows
- count_high_confidence_compounds_knowledge(min_confidence): Count entries above confidence threshold
- avg_confidence_compounds_knowledge(): Get average confidence

Use these functions to answer the user's question about table statistics and aggregations.
"""

    sql_llm = llm.bind_tools(sql_tools)
    sql_messages = [SystemMessage(content=sql_prompt), user_message]

    response = sql_llm.invoke(sql_messages)
    result_messages = [response]

    # Execute tool calls if present
    if response.tool_calls:
        tool_node = ToolNode(sql_tools)
        tool_results = tool_node.invoke({"messages": [response]})
        result_messages.extend(tool_results["messages"])

        # Get final answer from LLM
        final_response = llm.invoke(sql_messages + result_messages)
        result_messages.append(final_response)

    # SQL worker returns final answer directly (no synthesis needed)
    final_answer = result_messages[-1].content if result_messages else "No results available."

    return {
        "messages": [AIMessage(content=final_answer)]
    }

############################################
# Vector Worker Nodes
############################################

def vector_worker_1_node(state: AgentState, config: RunnableConfig):
    """
    Vector worker 1: Searches genes and proteins indexes.
    """
    messages = state["messages"]
    user_message = [m for m in messages if isinstance(m, HumanMessage)][-1]

    worker_tools = [VECTOR_SEARCH_TOOLS[0], VECTOR_SEARCH_TOOLS[1]]  # genes, proteins

    worker_prompt = """You are Vector Worker 1, specializing in genes and proteins knowledge.

Search the genes and proteins knowledge bases to find relevant information for the user's question.
Use BOTH search tools to gather relevant information. Return the results with their confidence scores.
"""

    worker_llm = llm.bind_tools(worker_tools)
    worker_messages = [SystemMessage(content=worker_prompt), user_message]

    response = worker_llm.invoke(worker_messages)
    result_messages = [response]

    # Execute tool calls and capture results
    search_results = []
    if response.tool_calls:
        tool_node = ToolNode(worker_tools)
        tool_results = tool_node.invoke({"messages": [response]})
        result_messages.extend(tool_results["messages"])

        # Extract tool result content
        for msg in tool_results["messages"]:
            if hasattr(msg, 'content') and msg.content:
                search_results.append(str(msg.content))

        # Synthesize findings if we have results
        if search_results:
            final_response = llm.invoke(worker_messages + result_messages + [
                SystemMessage(content="Summarize the search results concisely, highlighting key findings and confidence scores.")
            ])
            result_messages.append(final_response)

    # Compile results for synthesizer
    final_result = result_messages[-1].content if result_messages else "No results found"
    if search_results:
        final_result = f"{final_result}\n\nRaw search results:\n" + "\n---\n".join(search_results[:500])  # Limit size

    worker_results = state.get("worker_results", {})
    worker_results["vector_worker_1"] = {
        "result": final_result,
        "confidence": 0.80,
        "source": "Vector Worker 1 (Genes & Proteins)",
        "tool_calls_made": len(response.tool_calls) if response.tool_calls else 0
    }

    return {
        "worker_results": worker_results,
        "messages": [AIMessage(content=f"[Vector Worker 1 completed with {len(search_results)} results]")]
    }

def vector_worker_2_node(state: AgentState, config: RunnableConfig):
    """
    Vector worker 2: Searches pathways and compounds indexes.
    """
    messages = state["messages"]
    user_message = [m for m in messages if isinstance(m, HumanMessage)][-1]

    worker_tools = [VECTOR_SEARCH_TOOLS[2], VECTOR_SEARCH_TOOLS[3]]  # pathways, compounds

    worker_prompt = """You are Vector Worker 2, specializing in pathways and compounds knowledge.

Search the pathways and compounds knowledge bases to find relevant information for the user's question.
Use BOTH search tools to gather relevant information. Return the results with their confidence scores.
"""

    worker_llm = llm.bind_tools(worker_tools)
    worker_messages = [SystemMessage(content=worker_prompt), user_message]

    response = worker_llm.invoke(worker_messages)
    result_messages = [response]

    # Execute tool calls and capture results
    search_results = []
    if response.tool_calls:
        tool_node = ToolNode(worker_tools)
        tool_results = tool_node.invoke({"messages": [response]})
        result_messages.extend(tool_results["messages"])

        # Extract tool result content
        for msg in tool_results["messages"]:
            if hasattr(msg, 'content') and msg.content:
                search_results.append(str(msg.content))

        # Synthesize findings if we have results
        if search_results:
            final_response = llm.invoke(worker_messages + result_messages + [
                SystemMessage(content="Summarize the search results concisely, highlighting key findings and confidence scores.")
            ])
            result_messages.append(final_response)

    # Compile results for synthesizer
    final_result = result_messages[-1].content if result_messages else "No results found"
    if search_results:
        final_result = f"{final_result}\n\nRaw search results:\n" + "\n---\n".join(search_results[:500])  # Limit size

    worker_results = state.get("worker_results", {})
    worker_results["vector_worker_2"] = {
        "result": final_result,
        "confidence": 0.80,
        "source": "Vector Worker 2 (Pathways & Compounds)",
        "tool_calls_made": len(response.tool_calls) if response.tool_calls else 0
    }

    return {
        "worker_results": worker_results,
        "messages": [AIMessage(content=f"[Vector Worker 2 completed with {len(search_results)} results]")]
    }

############################################
# Synthesizer/Judge Node
############################################

def synthesizer_node(state: AgentState, config: RunnableConfig):
    """
    Synthesizer/judge combines all worker results and produces final answer.
    """
    messages = state["messages"]
    user_message = [m for m in messages if isinstance(m, HumanMessage)][-1]
    worker_results = state.get("worker_results", {})

    # Compile worker results
    results_summary = "\n\n".join([
        f"**{key}**:\n- Confidence: {val['confidence']}\n- Result: {val['result']}"
        for key, val in worker_results.items()
    ])

    system_prompt = """You are a synthesizer/judge agent for a life sciences knowledge system.

You have received results from multiple specialized workers. Your job is to:
1. Evaluate the confidence and relevance of each result
2. Synthesize the information into a coherent, accurate answer
3. Cite which sources you used and their confidence levels
4. If results conflict, prefer higher confidence sources

Provide an answer based on the worker results. Be scientific and precise.
"""

    user_request = f"""Original Question: {user_message.content}

Worker Results:
{results_summary}

Synthesize these results to answer the original question."""

    final_messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_request)
    ]
    response = llm.invoke(final_messages)

    return {
        "messages": [response]
    }

############################################
# Routing Logic
############################################

def route_after_orchestrator(state: AgentState):
    """Route based on orchestrator decision using Send for parallel execution."""
    decision = state.get("route_decision", "vector")

    if decision == "sql":
        # SQL worker goes directly to END
        return Send("sql_worker", state)
    else:
        # Send to both vector workers in parallel
        return [
            Send("vector_worker_1", state),
            Send("vector_worker_2", state),
        ]

############################################
# Build the Graph
############################################

def create_orchestrator_agent():
    """Create the orchestrator-synthesizer agent graph with true parallel execution using Send."""
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("orchestrator", orchestrator_node)
    workflow.add_node("sql_worker", sql_worker_node)
    workflow.add_node("vector_worker_1", vector_worker_1_node)
    workflow.add_node("vector_worker_2", vector_worker_2_node)
    workflow.add_node("synthesizer", synthesizer_node)

    # Define flow
    workflow.set_entry_point("orchestrator")

    # Orchestrator uses Send for dynamic parallel routing
    workflow.add_conditional_edges("orchestrator", route_after_orchestrator)

    # SQL worker goes directly to END (no synthesis needed)
    workflow.add_edge("sql_worker", END)

    # Both vector workers go to synthesizer
    workflow.add_edge("vector_worker_1", "synthesizer")
    workflow.add_edge("vector_worker_2", "synthesizer")

    # Synthesizer goes to END
    workflow.add_edge("synthesizer", END)

    return workflow.compile()

############################################
# ResponsesAgent Wrapper
############################################

class LangGraphResponsesAgent(ResponsesAgent):
    def __init__(self, agent):
        self.agent = agent

    def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
        outputs = [
            event.item
            for event in self.predict_stream(request)
            if event.type == "response.output_item.done"
        ]
        return ResponsesAgentResponse(output=outputs, custom_outputs=request.custom_inputs)

    def predict_stream(
        self,
        request: ResponsesAgentRequest,
    ) -> Generator[ResponsesAgentStreamEvent, None, None]:
        cc_msgs = to_chat_completions_input([i.model_dump() for i in request.input])

        for event in self.agent.stream({"messages": cc_msgs}, stream_mode=["updates", "messages"]):
            if event[0] == "updates":
                for node_data in event[1].values():
                    if len(node_data.get("messages", [])) > 0:
                        yield from output_to_responses_items_stream(node_data["messages"])
            elif event[0] == "messages":
                try:
                    chunk = event[1][0]
                    if isinstance(chunk, AIMessageChunk) and (content := chunk.content):
                        yield ResponsesAgentStreamEvent(
                            **self.create_text_delta(delta=content, item_id=chunk.id),
                        )
                except Exception as e:
                    print(f"Stream error: {e}")

############################################
# Initialize Agent
############################################

mlflow.langchain.autolog()
agent = create_orchestrator_agent()
AGENT = LangGraphResponsesAgent(agent)
mlflow.models.set_model(AGENT)
