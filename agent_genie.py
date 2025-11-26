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
# GENIE_SPACE_ID = "01f0c64ba4c61bd49b1aa03af847407a"

## can't use widgets?
CATALOG = "mmt"
SCHEMA = "LS_agent"
VECTOR_SEARCH_ENDPOINT = "ls_vs_mmt"
GENIE_SPACE_ID = "01f0c96ac7941abc82c4d50100a66439" ### 

llm = ChatDatabricks(endpoint=LLM_ENDPOINT_NAME)

############################################
# Define Tools
############################################

# SQL Worker Tools
UC_TOOL_NAMES = [
    f"{CATALOG}.{SCHEMA}.count_genes_knowledge_rows",
    f"{CATALOG}.{SCHEMA}.count_proteins_knowledge_rows",
    f"{CATALOG}.{SCHEMA}.count_pathways_knowledge_rows",
    f"{CATALOG}.{SCHEMA}.count_compounds_knowledge_rows",
    f"{CATALOG}.{SCHEMA}.count_high_confidence_genes_knowledge",
    f"{CATALOG}.{SCHEMA}.count_high_confidence_proteins_knowledge",
    f"{CATALOG}.{SCHEMA}.count_high_confidence_pathways_knowledge",
    f"{CATALOG}.{SCHEMA}.count_high_confidence_compounds_knowledge",
    f"{CATALOG}.{SCHEMA}.avg_confidence_genes_knowledge",
    f"{CATALOG}.{SCHEMA}.avg_confidence_proteins_knowledge",
    f"{CATALOG}.{SCHEMA}.avg_confidence_pathways_knowledge",
    f"{CATALOG}.{SCHEMA}.avg_confidence_compounds_knowledge",
]
uc_toolkit = UCFunctionToolkit(function_names=UC_TOOL_NAMES)
sql_tools = uc_toolkit.tools

# Vector Search Tools
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

# Genie Agent
# GenieAgent is invoked directly in the direct_genie_node and genie_with_context_node
from databricks_langchain.genie import GenieAgent

genie_agent = GenieAgent(
    genie_space_id=GENIE_SPACE_ID,
    genie_agent_name="life_sciences_genie",
    description="Query the Genie AI assistant for structured data analysis questions about life sciences data"
)

############################################
# Agent State
############################################

def merge_worker_results(left: Optional[dict], right: Optional[dict]) -> dict:
    """Merge worker results from parallel execution."""
    if left is None:
        return right or {}
    if right is None:
        return left
    return {**left, **right}

class AgentState(TypedDict):
    messages: Annotated[Sequence[AnyMessage], add_messages]
    custom_inputs: Optional[dict[str, Any]]
    custom_outputs: Optional[dict[str, Any]]
    worker_results: Annotated[Optional[dict[str, Any]], merge_worker_results]
    route_decision: Optional[str]
    genie_attempt_count: Optional[int]
    context_added: Optional[bool]
    synthesizer_decision: Optional[str]
    sql_can_answer: Optional[bool]

############################################
# Orchestrator Node
############################################

def orchestrator_node(state: AgentState, config: RunnableConfig):
    """
    Orchestrator decides 3-way routing: SQL, Direct Genie, or Vector Search.
    """
    messages = state["messages"]
    last_message = messages[-1]
    genie_attempts = state.get("genie_attempt_count", 0)

    orchestrator_prompt = """You are an orchestrator for a life sciences knowledge base system with Genie AI integration.

Analyze the user's question and decide on the best route:

1. Route to "sql" if: Simple counts or averages about ONE table (e.g., "How many genes?", "Average confidence for proteins?")

2. Route to "genie" if:
   - Question is well-formed, specific, and has enough context for direct Genie query
   - Clear structured queries: "Compare X and Y", "Show me top 10 Z", "What is the total across all tables?"
   - All information needed is in the question itself

3. Route to "genie_context" if:
   - Question needs structured data/aggregation BUT is vague or lacks context
   - Questions like "What compounds do we have for kinases?" (needs to know what kinases are first)
   - Asks for data/aggregation about something that needs semantic lookup first

4. Route to "vector" if:
   - Question is purely semantic/conceptual (no aggregation needed)
   - Questions like "What are kinases?", "Explain cell signaling", "Tell me about proteins"

Respond with ONLY one word: "sql", "genie", "genie_context", or "vector"
"""

    decision_messages = [
        SystemMessage(content=orchestrator_prompt),
        last_message
    ]

    response = llm.invoke(decision_messages)
    decision = response.content.strip().lower()

    # Default to vector if unclear
    if decision not in ["sql", "genie", "genie_context", "vector"]:
        decision = "vector"

    print(f"[Orchestrator Decision: {decision.upper()}]")

    return {
        "route_decision": decision,
        "genie_attempt_count": genie_attempts,
        "messages": [AIMessage(content=f"[Orchestrator: Routing to {decision}]")]
    }

############################################
# SQL Worker Node
############################################

def sql_worker_node(state: AgentState, config: RunnableConfig):
    """SQL worker handles aggregation queries using UC functions."""
    messages = state["messages"]
    user_message = [m for m in messages if isinstance(m, HumanMessage)][-1]

    sql_prompt = f"""You are a SQL worker for a life sciences knowledge base.

Available UC functions for each table (genes/proteins/pathways/compounds):
- count_<table>_rows(): Count total rows
- count_high_confidence_<table>(min_confidence): Count entries above confidence threshold
- avg_confidence_<table>(): Get average confidence

If you can answer using these functions, use them. If the question is beyond these simple functions, respond with: "CANNOT_ANSWER"
"""

    sql_llm = llm.bind_tools(sql_tools)
    sql_messages = [SystemMessage(content=sql_prompt), user_message]

    response = sql_llm.invoke(sql_messages)
    result_messages = [response]

    # Check if SQL worker can handle this
    can_answer = True
    if response.tool_calls:
        tool_node = ToolNode(sql_tools)
        tool_results = tool_node.invoke({"messages": [response]})
        result_messages.extend(tool_results["messages"])

        final_response = llm.invoke(sql_messages + result_messages)
        result_messages.append(final_response)
        final_answer = final_response.content
    else:
        final_answer = response.content

    # Check if SQL worker couldn't answer
    if "CANNOT_ANSWER" in final_answer:
        can_answer = False
        final_answer = "[SQL Worker: Cannot handle this query, routing back to orchestrator]"

    return {
        "sql_can_answer": can_answer,
        "messages": [AIMessage(content=final_answer)]
    }

############################################
# Context Fetcher Node (triggers parallel vector search)
############################################

def context_fetcher_node(state: AgentState, config: RunnableConfig):
    """
    Context fetcher node that triggers parallel vector search.
    Sets a flag so vector workers know to route to context_rewriter instead of synthesizer.
    """
    print("[Context Fetcher: Triggering parallel vector search for context]")
    return {
        "context_added": True,  # Flag to indicate we're in context-fetching mode
        "messages": [AIMessage(content="[Fetching context from vector search...]")]
    }

############################################
# Context Rewriter Node (after vector workers complete)
############################################

def context_rewriter_node(state: AgentState, config: RunnableConfig):
    """
    Rewrites the original question with context from vector search workers.
    Extracts structured information from documents and passes to Genie.
    """
    messages = state["messages"]
    user_message = [m for m in messages if isinstance(m, HumanMessage)][-1]
    worker_results = state.get("worker_results", {})

    # Collect full vector search results without manual parsing
    context_parts = []
    for key, val in worker_results.items():
        source = val.get('source', key)
        result = val.get('result', '')
        context_parts.append(f"{source}:\n{result}")

    context_summary = "\n\n".join(context_parts)

    rewriter_prompt = """You are a question rewriter for a Genie AI system. You have full vector search results about the user's question.

Your task:
1. Review the complete vector search context provided
2. Extract relevant entity IDs, names, and attributes from the results
3. Rewrite the question to be explicit and specific for Genie
4. Include concrete entity identifiers found in the search results
5. Make it a clear structured data query that Genie can answer directly

Example:
- Original: "What compounds do we have for kinases?"
- Context includes: COMP_0048, COMP_0052 (compounds targeting kinases)
- Rewritten: "Show me data for compounds COMP_0048, COMP_0052, COMP_0012 that target kinases, including their confidence scores and bioavailability"
"""

    rewriter_input = f"""Original Question: {user_message.content}

Complete Vector Search Results:
{context_summary}

Rewrite the question to be specific, include entity IDs/names found, and make it answerable by Genie:"""

    response = llm.invoke([
        SystemMessage(content=rewriter_prompt),
        HumanMessage(content=rewriter_input)
    ])
    enriched_question = response.content

    print(f"[Context Rewriter: Creating enriched question for Genie]")
    print(f"  Original: {user_message.content[:100]}")
    print(f"  Context Summary Length: {len(context_summary)} chars")
    print(f"  Context Preview: {context_summary[:200]}...")
    print(f"  Enriched: {enriched_question[:150]}...")

    return {
        "messages": [HumanMessage(content=enriched_question)],
        "context_added": True
    }

############################################
# Direct Genie Node
############################################

def direct_genie_node(state: AgentState, config: RunnableConfig):
    """
    Direct Genie call for well-formed questions.
    """
    messages = state["messages"]
    user_message = [m for m in messages if isinstance(m, HumanMessage)][-1]
    genie_attempts = state.get("genie_attempt_count", 0)

    # Invoke Genie agent directly with the user's message
    genie_response = genie_agent.invoke({"messages": [user_message]})

    # Extract result from Genie response
    if "messages" in genie_response:
        genie_result = genie_response["messages"][-1].content if genie_response["messages"] else "No results from Genie."
    else:
        genie_result = str(genie_response)

    return {
        "messages": [AIMessage(content=f"[Direct Genie Result]\n{genie_result}")],
        "genie_attempt_count": genie_attempts + 1
    }

############################################
# Quality Check Node
############################################

def quality_check_node(state: AgentState, config: RunnableConfig):
    """
    Evaluate quality of Genie response and decide whether to return or retry.
    """
    messages = state["messages"]
    last_message = messages[-1]
    genie_attempts = state.get("genie_attempt_count", 0)

    quality_prompt = """You are a quality checker for Genie AI responses.

Evaluate the Genie result and determine if it adequately answers the user's question.

Consider:
- Does it contain actual data, numbers, tables, or structured results?
- Does it directly address the question asked?
- Tables, comparisons, statistics, and data visualizations are GOOD responses

Respond with ONLY: "pass" if the response contains relevant data/results, "fail" if it's just an error message or completely empty
"""

    quality_messages = [
        SystemMessage(content=quality_prompt),
        HumanMessage(content=f"Genie result to evaluate:\n{last_message.content}")
    ]

    response = llm.invoke(quality_messages)
    quality = response.content.strip().lower()

    # Only allow one retry to avoid loops
    if "fail" in quality and genie_attempts < 2:
        decision = "retry"
    else:
        decision = "pass"

    return {
        "route_decision": decision,
        "messages": [AIMessage(content=f"[Quality Check: {decision}]")]
    }

############################################
# Vector Worker Nodes
############################################

def vector_worker_1_node(state: AgentState, config: RunnableConfig):
    """Vector worker 1: Searches genes and proteins indexes."""
    messages = state["messages"]
    user_message = [m for m in messages if isinstance(m, HumanMessage)][-1]

    worker_tools = [VECTOR_SEARCH_TOOLS[0], VECTOR_SEARCH_TOOLS[1]]

    worker_prompt = """You are Vector Worker 1, specializing in genes and proteins knowledge.

Search the genes and proteins knowledge bases to find relevant information for the user's question.
Use BOTH search tools to gather relevant information.
"""

    worker_llm = llm.bind_tools(worker_tools)
    worker_messages = [SystemMessage(content=worker_prompt), user_message]

    response = worker_llm.invoke(worker_messages)
    result_messages = [response]

    search_results = []
    if response.tool_calls:
        tool_node = ToolNode(worker_tools)
        tool_results = tool_node.invoke({"messages": [response]})
        result_messages.extend(tool_results["messages"])

        for msg in tool_results["messages"]:
            if hasattr(msg, 'content') and msg.content:
                search_results.append(str(msg.content))

        if search_results:
            final_response = llm.invoke(worker_messages + result_messages + [
                SystemMessage(content="Summarize the search results concisely.")
            ])
            result_messages.append(final_response)

    final_result = result_messages[-1].content if result_messages else "No results found"
    if search_results:
        final_result = f"{final_result}\n\nRaw results:\n" + "\n---\n".join(search_results[:500])

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
    """Vector worker 2: Searches pathways and compounds indexes."""
    messages = state["messages"]
    user_message = [m for m in messages if isinstance(m, HumanMessage)][-1]

    worker_tools = [VECTOR_SEARCH_TOOLS[2], VECTOR_SEARCH_TOOLS[3]]

    worker_prompt = """You are Vector Worker 2, specializing in pathways and compounds knowledge.

Search the pathways and compounds knowledge bases to find relevant information for the user's question.
Use BOTH search tools to gather relevant information.
"""

    worker_llm = llm.bind_tools(worker_tools)
    worker_messages = [SystemMessage(content=worker_prompt), user_message]

    response = worker_llm.invoke(worker_messages)
    result_messages = [response]

    search_results = []
    if response.tool_calls:
        tool_node = ToolNode(worker_tools)
        tool_results = tool_node.invoke({"messages": [response]})
        result_messages.extend(tool_results["messages"])

        for msg in tool_results["messages"]:
            if hasattr(msg, 'content') and msg.content:
                search_results.append(str(msg.content))

        if search_results:
            final_response = llm.invoke(worker_messages + result_messages + [
                SystemMessage(content="Summarize the search results concisely.")
            ])
            result_messages.append(final_response)

    final_result = result_messages[-1].content if result_messages else "No results found"
    if search_results:
        final_result = f"{final_result}\n\nRaw results:\n" + "\n---\n".join(search_results[:500])

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
# Synthesizer/Judge/Question-Writer Node
############################################

def synthesizer_node(state: AgentState, config: RunnableConfig):
    """
    Synthesizer evaluates vector results and decides:
    1. Return complete answer directly, OR
    2. Rewrite question with context for Genie
    """
    messages = state["messages"]
    user_message = [m for m in messages if isinstance(m, HumanMessage)][-1]
    worker_results = state.get("worker_results", {})

    results_summary = "\n\n".join([
        f"**{key}**:\n- Confidence: {val['confidence']}\n- Result: {val['result']}"
        for key, val in worker_results.items()
    ])

    synthesizer_prompt = """You are a synthesizer for a life sciences knowledge system.

You have received semantic search results from vector workers. Synthesize them into a clear answer.

If the vector search results are insufficient or empty, respond with:
"I was unable to find sufficient information to answer your question. Please try rephrasing with more specific details."

Otherwise, provide a clear answer based on the search results.
"""

    # Must include both SystemMessage and HumanMessage for LLM API
    final_messages = [
        SystemMessage(content=synthesizer_prompt),
        HumanMessage(content=f"Original Question: {user_message.content}\n\nWorker Results:\n{results_summary}")
    ]
    response = llm.invoke(final_messages)

    return {
        "messages": [response]
    }

############################################
# Genie with Context Node
############################################

def genie_with_context_node(state: AgentState, config: RunnableConfig):
    """
    Call Genie with context-enriched question (already rewritten by context_rewriter_node).
    """
    print("[Genie with Context Node: Calling Genie AI]")
    messages = state["messages"]
    # The last message is the rewritten question from context_rewriter
    enriched_question = messages[-1]

    # Invoke Genie agent with context-enriched question
    genie_response = genie_agent.invoke({"messages": [enriched_question]})

    # Extract result from Genie response
    if "messages" in genie_response:
        genie_result = genie_response["messages"][-1].content if genie_response["messages"] else "No results from Genie."
    else:
        genie_result = str(genie_response)

    return {
        "messages": [AIMessage(content=f"[Genie with Context Result]\n{genie_result}")]
    }

############################################
# Routing Logic
############################################

def route_from_orchestrator(state: AgentState):
    """Route from orchestrator: SQL, Direct Genie, Genie with Context, or Vector-only."""
    decision = state.get("route_decision", "vector")

    if decision == "sql":
        return "sql_worker"
    elif decision == "genie":
        return "direct_genie"
    elif decision == "genie_context":
        # Route to context fetcher which will trigger parallel vector search
        return "context_fetcher"
    else:
        # Pure vector search (no Genie after)
        return [
            Send("vector_worker_1", state),
            Send("vector_worker_2", state),
        ]

def route_after_quality_check(state: AgentState):
    """Route after quality check: END or to context fetcher."""
    decision = state.get("route_decision", "pass")

    if decision == "retry":
        # Failed Genie attempt - get context and try again with enriched question
        print("[Quality Check: Failed, routing to context fetcher]")
        return "context_fetcher"
    else:
        return END

def route_from_sql_worker(state: AgentState):
    """Route from SQL worker: END if answered, back to orchestrator if not."""
    can_answer = state.get("sql_can_answer", True)

    if can_answer:
        return END
    else:
        print("[SQL Worker: Cannot answer, routing back to orchestrator]")
        return "orchestrator"

def route_from_context_fetcher(state: AgentState):
    """Route from context fetcher to parallel vector workers."""
    return [
        Send("vector_worker_1", state),
        Send("vector_worker_2", state),
    ]

def route_from_vector_workers(state: AgentState):
    """Route from vector workers: to context_rewriter if context mode, else synthesizer."""
    context_mode = state.get("context_added", False)

    if context_mode:
        # We're in the genie_context path - go to context_rewriter
        return "context_rewriter"
    else:
        # We're in the vector-only path - go to synthesizer
        return "synthesizer"

def route_after_synthesizer(state: AgentState):
    """Route after synthesizer: just END (synthesizer handles vector-only queries)."""
    return END

############################################
# Build the Graph
############################################

def create_genie_agent():
    """Create the orchestrator-synthesizer-Genie agent graph."""
    workflow = StateGraph(AgentState)

    # Add all nodes
    workflow.add_node("orchestrator", orchestrator_node)
    workflow.add_node("sql_worker", sql_worker_node)
    workflow.add_node("direct_genie", direct_genie_node)
    workflow.add_node("quality_check", quality_check_node)
    workflow.add_node("context_fetcher", context_fetcher_node)
    workflow.add_node("context_rewriter", context_rewriter_node)
    workflow.add_node("vector_worker_1", vector_worker_1_node)
    workflow.add_node("vector_worker_2", vector_worker_2_node)
    workflow.add_node("synthesizer", synthesizer_node)
    workflow.add_node("genie_with_context", genie_with_context_node)

    # Define flow
    workflow.set_entry_point("orchestrator")

    # Orchestrator routes to: sql_worker, direct_genie, context_fetcher, or [vector_worker_1, vector_worker_2]
    workflow.add_conditional_edges("orchestrator", route_from_orchestrator)

    # SQL worker can route back to orchestrator or END
    workflow.add_conditional_edges("sql_worker", route_from_sql_worker)

    # Direct Genie goes to quality check
    workflow.add_edge("direct_genie", "quality_check")

    # Quality check: END or to context_fetcher
    workflow.add_conditional_edges("quality_check", route_after_quality_check)

    # Context fetcher fans out to vector workers (parallel)
    workflow.add_conditional_edges("context_fetcher", route_from_context_fetcher)

    # Vector workers route based on context: to context_rewriter or synthesizer
    workflow.add_conditional_edges("vector_worker_1", route_from_vector_workers)
    workflow.add_conditional_edges("vector_worker_2", route_from_vector_workers)

    # Context rewriter goes to Genie with context
    workflow.add_edge("context_rewriter", "genie_with_context")

    # Genie with context goes to END
    workflow.add_edge("genie_with_context", END)

    # Synthesizer (from vector-only path) goes to END
    workflow.add_conditional_edges("synthesizer", route_after_synthesizer)

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
agent = create_genie_agent()
AGENT = LangGraphResponsesAgent(agent)
mlflow.models.set_model(AGENT)
