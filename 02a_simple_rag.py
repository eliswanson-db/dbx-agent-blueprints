# Databricks notebook source
# MAGIC %md
# MAGIC # Simple RAG Agent
# MAGIC
# MAGIC This notebook implements a basic RAG (Retrieval Augmented Generation) pattern:
# MAGIC
# MAGIC **Architecture:**
# MAGIC - Single vector search retriever for genes knowledge
# MAGIC - Direct LLM synthesis of results
# MAGIC - No UC tools, no parallel workers
# MAGIC - Simple, straightforward question-answer flow
# MAGIC
# MAGIC **Use Case:**
# MAGIC - Semantic questions about genes
# MAGIC - Educational/explanatory queries
# MAGIC - Simple knowledge retrieval

# COMMAND ----------

# MAGIC %pip install -U -qqqq langgraph uv databricks-agents databricks-langchain mlflow-skinny[databricks] databricks-vectorsearch
# MAGIC dbutils.library.restartPython()

# COMMAND ----------
# MAGIC %pip freeze

# COMMAND ----------


# Widgets for configuration
dbutils.widgets.text("catalog", "dbxmetagen", "Catalog")
dbutils.widgets.text("schema", "default", "Schema")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")

print(f"Using catalog: {catalog}, schema: {schema}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Define the Simple RAG Agent
# MAGIC
# MAGIC This is a simple RAG chain with a single agent node:
# MAGIC
# MAGIC ### Architecture Flow
# MAGIC
# MAGIC ```mermaid
# MAGIC graph LR
# MAGIC     Start([User Query]) --> Agent[Agent with Vector Search Tool]
# MAGIC     Agent --> End([Final Answer])
# MAGIC
# MAGIC     style Agent fill:#e1f5ff
# MAGIC     style Start fill:#fce4ec
# MAGIC     style End fill:#fce4ec
# MAGIC ```

# COMMAND ----------

# MAGIC %%writefile agent.py
# MAGIC from typing import Annotated, Any, Generator, Optional, Sequence, TypedDict
# MAGIC import json
# MAGIC
# MAGIC import mlflow
# MAGIC from databricks_langchain import ChatDatabricks, VectorSearchRetrieverTool
# MAGIC from langchain.messages import AIMessage, AIMessageChunk, AnyMessage, HumanMessage, SystemMessage
# MAGIC from langchain_core.runnables import RunnableConfig
# MAGIC from langgraph.graph import END, StateGraph
# MAGIC from langgraph.graph.message import add_messages
# MAGIC from langgraph.prebuilt.tool_node import ToolNode
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
# MAGIC CATALOG = "dbxmetagen"
# MAGIC SCHEMA = "default"
# MAGIC
# MAGIC llm = ChatDatabricks(endpoint=LLM_ENDPOINT_NAME)
# MAGIC
# MAGIC ############################################
# MAGIC # Define Tools
# MAGIC ############################################
# MAGIC
# MAGIC # Single Vector Search Tool - Genes knowledge base
# MAGIC VECTOR_SEARCH_TOOL = VectorSearchRetrieverTool(
# MAGIC     index_name=f"{CATALOG}.{SCHEMA}.genes_knowledge_vs_index",
# MAGIC     columns=["id", "name", "description", "confidence"],
# MAGIC     num_results=5,
# MAGIC     name="search_genes",
# MAGIC     description="Search the genes knowledge base for information about genes, their functions, and associated diseases."
# MAGIC )
# MAGIC
# MAGIC ############################################
# MAGIC # Agent State
# MAGIC ############################################
# MAGIC
# MAGIC class AgentState(TypedDict):
# MAGIC     messages: Annotated[Sequence[AnyMessage], add_messages]
# MAGIC     custom_inputs: Optional[dict[str, Any]]
# MAGIC     custom_outputs: Optional[dict[str, Any]]
# MAGIC
# MAGIC ############################################
# MAGIC # Agent Node
# MAGIC ############################################
# MAGIC
# MAGIC def agent_node(state: AgentState, config: RunnableConfig):
# MAGIC     """
# MAGIC     Simple RAG agent that uses vector search tool to answer questions.
# MAGIC     """
# MAGIC     messages = state["messages"]
# MAGIC
# MAGIC     system_prompt = """You are a helpful assistant specializing in genetics and life sciences.
# MAGIC
# MAGIC You have access to a genes knowledge base through the search_genes tool. Use it to find relevant information when answering questions about genes.
# MAGIC
# MAGIC Guidelines:
# MAGIC - Use the search tool to find relevant information
# MAGIC - Be scientific and precise
# MAGIC - Cite confidence scores when available in search results
# MAGIC - If search results are insufficient, say so clearly
# MAGIC - Explain concepts in an accessible way
# MAGIC """
# MAGIC
# MAGIC     # Add system message if not already present
# MAGIC     if not any(isinstance(m, SystemMessage) for m in messages):
# MAGIC         messages = [SystemMessage(content=system_prompt)] + list(messages)
# MAGIC
# MAGIC     # Agent with tool access
# MAGIC     agent_llm = llm.bind_tools([VECTOR_SEARCH_TOOL])
# MAGIC     response = agent_llm.invoke(messages)
# MAGIC
# MAGIC     # If agent wants to use tools, execute them and get final response
# MAGIC     if response.tool_calls:
# MAGIC         tool_node = ToolNode([VECTOR_SEARCH_TOOL])
# MAGIC         tool_results = tool_node.invoke({"messages": [response]})
# MAGIC
# MAGIC         # Get final response after tool use
# MAGIC         messages_with_tools = list(messages) + [response] + tool_results["messages"]
# MAGIC         final_response = llm.invoke(messages_with_tools)
# MAGIC
# MAGIC         return {"messages": [response] + tool_results["messages"] + [final_response]}
# MAGIC
# MAGIC     return {"messages": [response]}
# MAGIC
# MAGIC ############################################
# MAGIC # Build the Graph
# MAGIC ############################################
# MAGIC
# MAGIC def create_simple_rag_agent():
# MAGIC     """Create the simple RAG agent graph (single node chain)."""
# MAGIC     workflow = StateGraph(AgentState)
# MAGIC
# MAGIC     # Single agent node with tool access
# MAGIC     workflow.add_node("agent", agent_node)
# MAGIC
# MAGIC     # Simple linear flow
# MAGIC     workflow.set_entry_point("agent")
# MAGIC     workflow.add_edge("agent", END)
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
# MAGIC agent = create_simple_rag_agent()
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

print(f"Using catalog: {catalog}, schema: {schema}")

# Import the agent
from agent import AGENT

# COMMAND ----------

# Test 1: Simple gene query
print("=" * 80)
print("Test 1: Simple Gene Query")
print("=" * 80)
result1 = AGENT.predict(
    {
        "input": [
            {
                "role": "user",
                "content": "What genes are associated with cancer?",
            }
        ]
    }
)
print(result1.model_dump(exclude_none=True))

# COMMAND ----------

# Test 2: Function-related query
print("=" * 80)
print("Test 2: Gene Function Query")
print("=" * 80)
result2 = AGENT.predict(
    {
        "input": [
            {
                "role": "user",
                "content": "Tell me about genes involved in DNA repair.",
            }
        ]
    }
)
print(result2.model_dump(exclude_none=True))

# COMMAND ----------

# Test 3: Disease-related query
print("=" * 80)
print("Test 3: Disease-related Query")
print("=" * 80)
result3 = AGENT.predict(
    {
        "input": [
            {
                "role": "user",
                "content": "What genes have high confidence scores related to diabetes?",
            }
        ]
    }
)
print(result3.model_dump(exclude_none=True))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Log the Agent as an MLflow Model

# COMMAND ----------

from agent import VECTOR_SEARCH_TOOL, CATALOG, SCHEMA
import mlflow
from pkg_resources import get_distribution

# Collect resources for automatic auth passthrough
resources = []
resources.extend(VECTOR_SEARCH_TOOL.resources)

# Log the model
with mlflow.start_run():
    logged_agent_info = mlflow.pyfunc.log_model(
        name="agent",
        python_model="agent.py",
        pip_requirements=[
            "databricks-langchain",
            "databricks-vectorsearch",
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
                    "content": "What genes are associated with cancer?",
                }
            ]
        },
        "expected_response": "Genes associated with cancer include tumor suppressors and oncogenes.",
    },
    {
        "inputs": {
            "input": [
                {
                    "role": "user",
                    "content": "Tell me about genes involved in DNA repair.",
                }
            ]
        },
        "expected_response": "DNA repair genes help maintain genomic stability and prevent mutations.",
    },
    {
        "inputs": {
            "input": [
                {"role": "user", "content": "What are the functions of kinase genes?"}
            ]
        },
        "expected_response": "Kinase genes encode enzymes that catalyze phosphorylation reactions.",
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
                "content": "What do you know about tumor suppressor genes?",
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

model_name = "lifesciences_simple_rag_agent"
UC_MODEL_NAME = f"{catalog}.{schema}.{model_name}"

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
# MAGIC ## Deploy the Agent

# COMMAND ----------

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
# MAGIC This is a simple RAG chain that demonstrates the basic pattern:
# MAGIC
# MAGIC ### Flow
# MAGIC
# MAGIC 1. **Agent**: User query → Agent decides to use vector search tool → Synthesizes answer
# MAGIC
# MAGIC ### Key Features
# MAGIC - Single agent node with tool access (true RAG chain)
# MAGIC - Single vector index for simplicity
# MAGIC - No UC functions, parallel workers, or multiple nodes
# MAGIC - Agent handles both retrieval and synthesis
# MAGIC - Good for educational/explanatory questions about genes
