# Databricks notebook source
# MAGIC %md
# MAGIC # Simulate the `ongoing` Agent/Model development (`training/tuning`)
# MAGIC
# MAGIC ## Updates in our Life Sciences Orchestrator-Synthesizer Agent
# MAGIC
# MAGIC We will use our existing `alpha` or `beta` `Agent in Code` Definitions
# MAGIC where we tweaked the responses of workers to yield different response relevance.    
# MAGIC
# MAGIC ---    
# MAGIC
# MAGIC Prior implementations for each Agent: a LangGraph agent with an orchestrator-synthesizer architecture:
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

# dbutils.widgets.removeAll()

# COMMAND ----------

# MAGIC %pip install -U -qqqq langgraph uv databricks-agents databricks-langchain mlflow-skinny[databricks] databricks-vectorsearch
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Widgets
# Widgets for catalog, schema, and model base name

dbutils.widgets.text("catalog", "mmt", "Catalog")
dbutils.widgets.text("schema", "LS_agent", "Schema")
dbutils.widgets.text("model_base_name", "lifesciences_agent", "Model Base Name")
dbutils.widgets.text("vs_endpoint_name", "ls_vs_mmt", "VectorSearch_endpoint") #lifesciences_vector_search

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
model_base_name = dbutils.widgets.get("model_base_name")
vs_endpoint_name = dbutils.widgets.get("vs_endpoint_name")

# Construct Fully Qualified Unity Catalog model name
UC_MODEL_NAME = f"{catalog}.{schema}.{model_base_name}"

print(f"Using catalog: {catalog}, schema: {schema}")
print(f"Model base name: {model_base_name}")
print(f"FQ UC Model Name: {UC_MODEL_NAME}")
print(f"VectorSearch_endpoint: {vs_endpoint_name}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## We will use our previously defined `Agent in Code`
# MAGIC
# MAGIC The defined `agent` uses an orchestrator-synthesizer pattern with true parallel execution using LangGraph's Send API.
# MAGIC
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

# MAGIC %md
# MAGIC ## Simulate updates in `Agent` | `Model` Development
# MAGIC **Randonly select between `alpha` or `beta` Agent Definition 
# MAGIC `-->` then log the Agent as an MLflow Model**

# COMMAND ----------

# DBTITLE 1,selected_agent_file
import random

# Select alpha_agent.py with 1/3 probability, beta_agent.py with 2/3 probability
selected_agent_file = random.choices(
    ['alpha_agent.py', 'beta_agent.py'], weights=[1, 2], k=1
)[0]
print(f"Selected agent file: {selected_agent_file}")

# COMMAND ----------

# DBTITLE 1,MLflow log Agent
from agent import UC_TOOL_NAMES, VECTOR_SEARCH_TOOLS
import mlflow
from mlflow.models.resources import DatabricksFunction
from pkg_resources import get_distribution
import os
import re

# Collect all resources for automatic auth passthrough
resources = []

# Add vector search tools
for tool in VECTOR_SEARCH_TOOLS:
    resources.extend(tool.resources)

# Add UC functions
for tool_name in UC_TOOL_NAMES:
    resources.append(DatabricksFunction(function_name=tool_name))

# Sanitize model name for logging (MLflow does not allow . / % :)
def sanitize_model_name(name):
    return re.sub(r"[./%:]", "-", name)

mlflow_log_name = sanitize_model_name(model_base_name)

print(f"Logging agent from file: {selected_agent_file}")
print(f"Model name for logging: {mlflow_log_name}")
print(f"Model name for registration (FQ): {UC_MODEL_NAME}")

with mlflow.start_run():
    logged_agent_info = mlflow.pyfunc.log_model(
        name=mlflow_log_name,
        python_model=selected_agent_file,
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
# MAGIC ## Evaluate the Agent

# COMMAND ----------

import mlflow
from mlflow.genai.scorers import RelevanceToQuery, Safety

# Load the just-logged agent model
agent_model = mlflow.pyfunc.load_model(logged_agent_info.model_uri)

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
    predict_fn=lambda input: agent_model.predict({"input": input}),
    scorers=[RelevanceToQuery(), Safety()],
)

print("Evaluation complete. Check MLflow UI for results.")

# COMMAND ----------

# DBTITLE 1,display evals
eval_results.result_df.display()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Pre-deployment Validation

# COMMAND ----------

# Run prediction and display the result
import mlflow

try:
    # Load the model as a PyFunc
    model = mlflow.pyfunc.load_model(logged_agent_info.model_uri)
    print("Model loaded successfully.")
    
    # Prepare input
    input_data = {
        "input": [
            {
                "role": "user",
                "content": "What are the high confidence proteins in the knowledge base?",
            }
        ]
    }
    
    # Call predict directly
    prediction = model.predict(input_data)
    print("Prediction result from model.predict:")
    print(prediction)
except Exception as e:
    print(f"Error during model loading or prediction: {e}")
    prediction = None

# If prediction is still None, try mlflow.models.predict as fallback
if prediction is None:
    try:
        print("Attempting to use mlflow.models.predict as a fallback.")
        prediction = mlflow.models.predict(
            model_uri=logged_agent_info.model_uri,
            input_data=input_data,
            env_manager="uv",
        )
        print("Prediction result from mlflow.models.predict:")
        print(prediction)
    except Exception as e:
        print(f"Error during mlflow.models.predict: {e}")

if prediction is None:
    print("Prediction is None. Please check that your agent's predict method returns a value.")
    print("If using a custom agent, ensure the predict method returns a string, dict, or DataFrame.")
    print("Next steps:")
    print("1. Verify the model URI is correct.")
    print("2. Check the input data format.")
    print("3. Ensure the model is trained and capable of making predictions.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Register to Unity Catalog

# COMMAND ----------

mlflow.set_registry_uri("databricks-uc")

# Use the FQ model name from widgets
# model_base_name = "lifesciences_agent"

## Construct Fully Qualified Unity Catalog model name
# UC_MODEL_NAME = f"{catalog}.{schema}.{model_base_name}"

# Check existing versions before registering
from mlflow.tracking import MlflowClient

client = MlflowClient()

try:
    existing_versions = client.search_model_versions(f"name='{UC_MODEL_NAME}'")
    print(f"Existing model versions: {[v.version for v in existing_versions]}")
except Exception as e:
    print(f"No existing versions found (this is fine for first run): {e}")

# Register the model - this creates a new version
print(f"Registering model from run: {logged_agent_info.run_id} with model name: {UC_MODEL_NAME}")
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
# from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedEntityInput
# import time

# w = WorkspaceClient()

# # Check if endpoint already exists and delete if in bad state
# # agents.deploy() creates endpoint with "agents_" prefix and replaces dots with dashes
# vs_endpoint_name = f"agents_{UC_MODEL_NAME.replace('.', '-')}"
# print(f"\n=== Deployment Information ===")
# print(f"Model: {UC_MODEL_NAME}")
# print(f"Version to deploy: {uc_registered_model_info.version}")
# print(f"Endpoint name: {vs_endpoint_name}")

# try:
#     existing_endpoint = w.serving_endpoints.get(vs_endpoint_name)
#     print(f"\nExisting endpoint found:")
#     print(
#         f"  State: {existing_endpoint.state.ready if existing_endpoint.state else 'Unknown'}"
#     )

#     # Check what version is currently served
#     if existing_endpoint.config and existing_endpoint.config.served_entities:
#         current_versions = [
#             e.entity_version for e in existing_endpoint.config.served_entities
#         ]
#         print(f"  Currently serving versions: {current_versions}")

#         if str(uc_registered_model_info.version) in [str(v) for v in current_versions]:
#             print(f"  Version {uc_registered_model_info.version} is already deployed.")
#             print(f"  This might explain why no new deployment event is created.")

#     # Check if endpoint is in failed state or config is None
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
#             f"\nEndpoint {vs_endpoint_name} is in failed/bad state. Deleting and recreating..."
#         )
#         w.serving_endpoints.delete(vs_endpoint_name)
#         print(f"  Waiting 60 seconds for deletion to complete...")
#         time.sleep(60)
#         print("Endpoint deleted - fresh endpoint will be created")
# except Exception as e:
#     if "RESOURCE_DOES_NOT_EXIST" not in str(e):
#         print(f"Note: {e}")
#     else:
#         print("No existing endpoint found - will create new one")

# # Deploy the agent with retry logic
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
#                 "architecture": "orchestrator-synthesizer",
#                 "domain": "life-sciences",
#             },
#             deploy_feedback_model=False,
#         )

#         print(f"\nAgent deployed successfully.")
#         print(f"  Model: {UC_MODEL_NAME}")
#         print(f"  Version: {uc_registered_model_info.version}")
#         print(f"  Endpoint: {vs_endpoint_name}")

#         # Show deployment details
#         deployed_endpoint = w.serving_endpoints.get(vs_endpoint_name)
#         if deployed_endpoint.config and deployed_endpoint.config.served_entities:
#             print(f"\nCurrently serving:")
#             for entity in deployed_endpoint.config.served_entities:
#                 print(
#                     f"  - Version {entity.entity_version} (traffic: {entity.scale_to_zero_enabled})"
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
#                 f"1. Check endpoint status: w.serving_endpoints.get('{vs_endpoint_name}')"
#             )
#             print(
#                 f"2. Try deleting endpoint manually: w.serving_endpoints.delete('{vs_endpoint_name}')"
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
# MAGIC ### Troubleshooting Deployment Issues
# MAGIC
# MAGIC If deployment fails with `auto_capture_config` error, run this cell to manually clean up:

# COMMAND ----------

# # Uncomment and run if deployment fails
# from databricks.sdk import WorkspaceClient
#
# w = WorkspaceClient()
# vs_endpoint_name = UC_MODEL_NAME.replace(".", "_")
#
# try:
#     # Check endpoint status
#     endpoint = w.serving_endpoints.get(vs_endpoint_name)
#     print(f"Endpoint state: {endpoint.state}")
#
#     # Delete if needed
#     # w.serving_endpoints.delete(vs_endpoint_name)
#     # print(f"Deleted endpoint: {vs_endpoint_name}")
#     # print("Wait 30 seconds and re-run deployment cell")
# except Exception as e:
#     print(f"Endpoint does not exist or error: {e}")

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

# COMMAND ----------

# MAGIC %md
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## Extract UC registered model info. from code

# COMMAND ----------

# DBTITLE 1,test UC registered model retrieval
import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd

mlflow.set_registry_uri("databricks-uc")

# UC_MODEL_NAME # Use FullyQualified UC MODEL NAME

client = MlflowClient()

model_info = {}
version_info_list = []

def safe_str(val):
    if isinstance(val, (dict, list)):
        return str(val)
    elif val is None:
        return ''
    else:
        return str(val)

# Get the registered model info
try:
    registered_model = client.get_registered_model(UC_MODEL_NAME)
    model_aliases = getattr(registered_model, 'aliases', {})
    # Convert model-level aliases dict to a string for display
    model_aliases_str = (
        ', '.join([f"{k}:{v}" for k, v in model_aliases.items()]) if model_aliases else ''
    )
    model_info = {
        "type": "model",
        "model_name": safe_str(registered_model.name),
        "description": safe_str(getattr(registered_model, 'description', '')),
        "creation_timestamp": safe_str(getattr(registered_model, 'creation_timestamp', '')),
        "last_updated": safe_str(getattr(registered_model, 'last_updated_timestamp', '')),
        "user_id": safe_str(getattr(registered_model, 'user_id', 'N/A')),
        "tags": safe_str(getattr(registered_model, 'tags', {})),
        "aliases": model_aliases_str,
    }
except Exception as e:
    print(f"Could not fetch model: {e}")

# Get all versions for this model and their aliases
try:
    versions = client.search_model_versions(f"name='{UC_MODEL_NAME}'")
    # Build a mapping of version -> list of aliases (invert model_aliases)
    version_to_aliases = {}
    for alias, version in model_aliases.items():
        version_to_aliases.setdefault(str(version), []).append(alias)

    for v in versions:
        v_aliases = version_to_aliases.get(str(v.version), [])
        # Convert list of aliases to a comma-separated string
        v_aliases_str = ', '.join(v_aliases) if v_aliases else ''
        version_info = {
            "type": "version",
            "version": safe_str(v.version),
            "status": safe_str(getattr(v, 'current_stage', None)),
            "run_id": safe_str(getattr(v, 'run_id', None)),
            "creation_timestamp": safe_str(getattr(v, 'creation_timestamp', None)),
            "source": safe_str(getattr(v, 'source', None)),
            "aliases": v_aliases_str,
        }
        version_info_list.append(version_info)
except Exception as e:
    print(f"Could not fetch model versions: {e}")

# Combine model info and version info for display
all_info = [model_info] + version_info_list if model_info else version_info_list

# # Convert all dict or list fields in all_info to strings
# for info in all_info:
#     for key in info:
#         if isinstance(info[key], (dict, list)):
#             info[key] = safe_str(info[key])

# df = pd.DataFrame(all_info)
# display(df)

# COMMAND ----------

all_info

# COMMAND ----------

# MAGIC %md
# MAGIC
