# Databricks notebook source
# MAGIC %md
# MAGIC # Life Sciences Knowledge Base Setup
# MAGIC
# MAGIC This notebook creates the data infrastructure for a life sciences agent system:
# MAGIC - 4 Delta tables with life sciences data (genes, proteins, pathways, compounds)
# MAGIC - Vector search endpoint
# MAGIC - 4 Vector search indexes (one per table)
# MAGIC
# MAGIC Each table has the same structure:
# MAGIC - `id`: Primary key
# MAGIC - `name`: Entity name
# MAGIC - `description`: Detailed text description
# MAGIC - `confidence`: Confidence score (0-1) for the information

# COMMAND ----------

# MAGIC %pip install -U -qqqq databricks-vectorsearch databricks-sdk mlflow-skinny pyyaml
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

dbutils.widgets.text("vs_endpoint_name", vs_endpoint_name, "VectorSearch_endpoint") #lifesciences_vector_search

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")

vs_endpoint_name = dbutils.widgets.get("vs_endpoint_name")

print(f"Using catalog: {catalog}, schema: {schema}")
print(f"VectorSearch_endpoint: {vs_endpoint_name}")

# COMMAND ----------

# Ensure catalog and schema exist
spark.sql(f"CREATE CATALOG IF NOT EXISTS {catalog}")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{schema}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create Life Sciences Data Tables

# COMMAND ----------

# Generate synthetic gene data
genes_data = [
    (
        f"GENE_{i:04d}",
        f"Gene_{i:04d}",
        f"This gene encodes a protein involved in cellular {['metabolism', 'signaling', 'transport', 'regulation'][i % 4]}. "
        f"It plays a critical role in {['cell growth', 'apoptosis', 'differentiation', 'immune response'][i % 4]} "
        f"and has been associated with {['cancer', 'diabetes', 'cardiovascular disease', 'neurodegenerative disorders'][i % 4]}. "
        f"Expression levels vary across {['liver', 'brain', 'heart', 'kidney'][i % 4]} tissue with peak activity during {['G1 phase', 'S phase', 'G2 phase', 'M phase'][i % 4]}.",
        0.70 + (i % 30) * 0.01,
    )
    for i in range(100)
]

genes_df = spark.createDataFrame(
    genes_data, ["id", "name", "description", "confidence"]
)

table_name = f"{catalog}.{schema}.genes_knowledge"
genes_df.write.format("delta").mode("overwrite").option(
    "overwriteSchema", "true"
).saveAsTable(table_name)
spark.sql(
    f"ALTER TABLE {table_name} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)"
)
print(f"Created table: {table_name}")

# COMMAND ----------

# Generate synthetic protein data
proteins_data = [
    (
        f"PROT_{i:04d}",
        f"Protein_{i:04d}",
        f"This protein functions as a {['kinase', 'phosphatase', 'receptor', 'transcription factor'][i % 4]} "
        f"with {['catalytic', 'regulatory', 'structural', 'binding'][i % 4]} activity. "
        f"It has {i % 5 + 2} functional domains including {['SH2', 'SH3', 'kinase', 'DNA-binding', 'transmembrane'][i % 5]} domains. "
        f"Post-translational modifications include {['phosphorylation', 'acetylation', 'methylation', 'ubiquitination'][i % 4]} "
        f"which regulate its {['stability', 'localization', 'activity', 'interactions'][i % 4]}.",
        0.65 + (i % 35) * 0.01,
    )
    for i in range(100)
]

proteins_df = spark.createDataFrame(
    proteins_data, ["id", "name", "description", "confidence"]
)

table_name = f"{catalog}.{schema}.proteins_knowledge"
proteins_df.write.format("delta").mode("overwrite").option(
    "overwriteSchema", "true"
).saveAsTable(table_name)
spark.sql(
    f"ALTER TABLE {table_name} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)"
)
print(f"Created table: {table_name}")

# COMMAND ----------

# Generate synthetic pathway data
pathways_data = [
    (
        f"PATH_{i:04d}",
        f"Pathway_{i:04d}",
        f"This is a {['metabolic', 'signaling', 'regulatory', 'biosynthetic'][i % 4]} pathway "
        f"involving {i % 10 + 5} key proteins and {i % 15 + 3} metabolites. "
        f"The pathway is activated by {['growth factors', 'hormones', 'stress signals', 'nutrients'][i % 4]} "
        f"and regulated through {['feedback inhibition', 'allosteric regulation', 'covalent modification', 'compartmentalization'][i % 4]}. "
        f"Dysregulation leads to {['oncogenesis', 'metabolic syndrome', 'inflammation', 'developmental defects'][i % 4]} "
        f"with therapeutic targets including {['kinases', 'receptors', 'enzymes', 'transcription factors'][i % 4]}.",
        0.75 + (i % 25) * 0.01,
    )
    for i in range(100)
]

pathways_df = spark.createDataFrame(
    pathways_data, ["id", "name", "description", "confidence"]
)

table_name = f"{catalog}.{schema}.pathways_knowledge"
pathways_df.write.format("delta").mode("overwrite").option(
    "overwriteSchema", "true"
).saveAsTable(table_name)
spark.sql(
    f"ALTER TABLE {table_name} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)"
)
print(f"Created table: {table_name}")

# COMMAND ----------

# Generate synthetic compound data
compounds_data = [
    (
        f"COMP_{i:04d}",
        f"Compound_{i:04d}",
        f"This {['small molecule', 'peptide', 'antibody', 'nucleotide analog'][i % 4]} compound "
        f"with molecular weight {200 + i * 10} Da exhibits {['agonist', 'antagonist', 'inhibitor', 'modulator'][i % 4]} activity. "
        f"It targets {['kinases', 'GPCRs', 'ion channels', 'nuclear receptors'][i % 4]} "
        f"with IC50 values in the {['nM', 'µM', 'pM', 'fM'][i % 4]} range. "
        f"Clinical trials show efficacy in treating {['cancer', 'autoimmune diseases', 'infections', 'metabolic disorders'][i % 4]} "
        f"with common side effects including {['nausea', 'fatigue', 'headache', 'rash'][i % 4]}. "
        f"Bioavailability is {30 + i % 50}% with a half-life of {2 + i % 20} hours.",
        0.60 + (i % 40) * 0.01,
    )
    for i in range(100)
]

compounds_df = spark.createDataFrame(
    compounds_data, ["id", "name", "description", "confidence"]
)

table_name = f"{catalog}.{schema}.compounds_knowledge"
compounds_df.write.format("delta").mode("overwrite").option(
    "overwriteSchema", "true"
).saveAsTable(table_name)
spark.sql(
    f"ALTER TABLE {table_name} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)"
)
print(f"Created table: {table_name}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create UC Functions for SQL Worker
# MAGIC
# MAGIC Create utility functions for the SQL worker agent to query table metadata and aggregations.
# MAGIC
# MAGIC Note: These are SQL-based functions, not Python UDFs, as they need to work across different compute contexts.

# COMMAND ----------

# DBTITLE 1,count_table_rows function
# Create SQL-based UC functions that can work with dynamic table names
# Note: UC SQL functions cannot directly use table_name as a parameter in FROM clauses
# So we create specific functions for each table instead

tables = [
    "genes_knowledge",
    "proteins_knowledge",
    "pathways_knowledge",
    "compounds_knowledge",
]

# Create count functions for each table
for table in tables:
    func_name = f"count_{table}_rows"
    table_display = table.replace("_", " ").title()
    spark.sql(
        f"""
    CREATE OR REPLACE FUNCTION {catalog}.{schema}.{func_name}()
    RETURNS BIGINT
    COMMENT 'Counts the total number of rows in the {table_display} table. Returns the count as a BIGINT value. Use this for simple row count queries.'
    RETURN (SELECT COUNT(*) FROM {catalog}.{schema}.{table})
    """
    )
    print(f"Created function: {catalog}.{schema}.{func_name}")

# COMMAND ----------

# Create high confidence count functions for each table
for table in tables:
    func_name = f"count_high_confidence_{table}"
    table_display = table.replace("_", " ").title()
    spark.sql(
        f"""
    CREATE OR REPLACE FUNCTION {catalog}.{schema}.{func_name}(min_confidence DOUBLE)
    RETURNS BIGINT
    COMMENT 'Counts rows in the {table_display} table with confidence >= the specified threshold. Parameter min_confidence: minimum confidence score (0-1). Returns count as BIGINT. Use this to find high-quality entries.'
    RETURN (SELECT COUNT(*) FROM {catalog}.{schema}.{table} WHERE confidence >= min_confidence)
    """
    )
    print(f"Created function: {catalog}.{schema}.{func_name}")

# COMMAND ----------

# Create average confidence functions for each table
for table in tables:
    func_name = f"avg_confidence_{table}"
    table_display = table.replace("_", " ").title()
    spark.sql(
        f"""
    CREATE OR REPLACE FUNCTION {catalog}.{schema}.{func_name}()
    RETURNS DOUBLE
    COMMENT 'Calculates the average confidence score across all entries in the {table_display} table. Returns average as DOUBLE (0-1 range). Use this to assess overall data quality.'
    RETURN (SELECT AVG(confidence) FROM {catalog}.{schema}.{table})
    """
    )
    print(f"Created function: {catalog}.{schema}.{func_name}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create Vector Search Endpoint and Indexes

# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient

vsc = VectorSearchClient(disable_notice=True)

# Create vector search endpoint
# endpoint_name = "lifesciences_vector_search"
endpoint_name = endpoint_name #use widget to update default values

# Check if endpoint already exists
endpoint_exists = False
try:
    existing_endpoint = vsc.get_endpoint(endpoint_name)
    endpoint_exists = True
    print(f"Vector search endpoint already exists: {endpoint_name}")
    current_state = existing_endpoint.get("endpoint_status", {}).get("state", "UNKNOWN")
    print(f"  Current state: {current_state}")
except Exception as e:
    if "RESOURCE_DOES_NOT_EXIST" in str(e):
        print(f"Creating new vector search endpoint: {endpoint_name}")
    else:
        print(f"Note checking endpoint: {e}")

# Create endpoint if it doesn't exist
if not endpoint_exists:
    try:
        vsc.create_endpoint(name=endpoint_name, endpoint_type="STANDARD")
        print(f"Created vector search endpoint: {endpoint_name}")
    except Exception as e:
        if "RESOURCE_ALREADY_EXISTS" in str(e) or "ALREADY_EXISTS" in str(e):
            print(f"Endpoint created concurrently: {endpoint_name}")
        else:
            print(f"Error creating endpoint: {e}")
            raise

# Wait for endpoint to be ready
import time

print(f"Waiting for endpoint {endpoint_name} to be ONLINE...")
max_wait_time = 600  # 10 minutes
wait_time = 0
while wait_time < max_wait_time:
    status = vsc.get_endpoint(endpoint_name)
    current_state = status.get("endpoint_status", {}).get("state", "UNKNOWN")

    if current_state == "ONLINE":
        print(f"Endpoint {endpoint_name} is ONLINE")
        break
    elif current_state in ["OFFLINE", "PROVISIONING"]:
        print(f"  Endpoint state: {current_state} (waiting...)")
        time.sleep(30)
        wait_time += 30
    else:
        print(f"  Endpoint state: {current_state}")
        time.sleep(30)
        wait_time += 30

if wait_time >= max_wait_time:
    print(
        f"Warning: Endpoint did not become ONLINE within {max_wait_time}s. Current state: {current_state}"
    )
    print(
        f"You may need to check the endpoint status manually and re-run index creation."
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ### Create Vector Search Indexes

# COMMAND ----------

tables = [
    "genes_knowledge",
    "proteins_knowledge",
    "pathways_knowledge",
    "compounds_knowledge",
]

for table in tables:
    source_table = f"{catalog}.{schema}.{table}"
    index_name = f"{catalog}.{schema}.{table}_vs_index"

    # Check if index already exists
    index_exists = False
    try:
        existing_index = vsc.get_index(
            endpoint_name=endpoint_name, index_name=index_name
        )
        index_exists = True
        print(f"Vector search index already exists: {index_name}")
    except Exception as e:
        if "RESOURCE_DOES_NOT_EXIST" in str(e) or "does not exist" in str(e).lower():
            print(f"Creating new vector search index: {index_name}")
        else:
            print(f"Note checking index: {e}")

    # Create index if it doesn't exist
    if not index_exists:
        try:
            index = vsc.create_delta_sync_index(
                endpoint_name=endpoint_name,
                source_table_name=source_table,
                index_name=index_name,
                pipeline_type="TRIGGERED",
                primary_key="id",
                embedding_source_column="description",
                embedding_model_endpoint_name="databricks-gte-large-en",
            )
            print(f"Created vector search index: {index_name}")
        except Exception as e:
            if "RESOURCE_ALREADY_EXISTS" in str(e) or "ALREADY_EXISTS" in str(e):
                print(f"Index created concurrently: {index_name}")
            else:
                print(f"Error creating index for {table}: {e}")
                raise

# COMMAND ----------

# Wait for indexes to be ready and sync
print("\n=== Syncing Vector Search Indexes ===")
print("Waiting for indexes to be ready...")
time.sleep(10)

for table in tables:
    index_name = f"{catalog}.{schema}.{table}_vs_index"
    try:
        index = vsc.get_index(endpoint_name=endpoint_name, index_name=index_name)

        # Check index status
        index_status = index.describe()
        status_state = index_status.get("status", {}).get("state", "UNKNOWN")
        print(f"  {index_name}: {status_state}")

        # Trigger sync
        if status_state in ["ONLINE", "ONLINE_CONTINUOUS_UPDATE"]:
            index.sync()
            print(f"    Sync triggered for {index_name}")
        else:
            print(f"    Index not ready for sync yet (state: {status_state})")
    except Exception as e:
        if "does not exist" in str(e).lower():
            print(f"    Index not found: {index_name}")
        else:
            print(f"    Note: {index_name} - {e}")

print("\nIndex sync triggered. Indexes will update in the background.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Verify Setup

# COMMAND ----------

print("\n=== Setup Summary ===\n")

print("Tables:")
for table in tables:
    table_name = f"{catalog}.{schema}.{table}"
    try:
        count = spark.table(table_name).count()
        print(f"  {table_name}: {count} rows")
    except Exception as e:
        print(f"  {table_name}: Error - {e}")

print(f"\nUC Functions:")
for table in tables:
    print(f"  {catalog}.{schema}.count_{table}_rows")
    print(f"  {catalog}.{schema}.count_high_confidence_{table}")
    print(f"  {catalog}.{schema}.avg_confidence_{table}")

print(f"\nVector Search Endpoint:")
try:
    endpoint_status = vsc.get_endpoint(endpoint_name)
    state = endpoint_status.get("endpoint_status", {}).get("state", "UNKNOWN")
    print(f"  {endpoint_name} (state: {state})")
except Exception as e:
    print(f"  {endpoint_name}: {e}")

print(f"\nVector Search Indexes:")
for table in tables:
    index_name = f"{catalog}.{schema}.{table}_vs_index"
    try:
        index = vsc.get_index(endpoint_name=endpoint_name, index_name=index_name)
        index_status = index.describe()
        state = index_status.get("status", {}).get("state", "UNKNOWN")
        print(f"  {index_name} (state: {state})")
    except Exception as e:
        print(f"  {index_name}: {e}")

print("\nSetup complete. Ready for agent deployment.")
print("\nNote: Vector indexes may take a few minutes to fully sync in the background.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Test Vector Search

# COMMAND ----------

# Test a sample query
test_query = "proteins involved in cell signaling"
index_name = f"{catalog}.{schema}.proteins_knowledge_vs_index"

try:
    results = vsc.get_index(
        endpoint_name=endpoint_name, index_name=index_name
    ).similarity_search(
        query_text=test_query,
        columns=["id", "name", "description", "confidence"],
        num_results=3,
    )
    print(f"Test query: '{test_query}'")
    print(f"Results from {index_name}:")
    print(results)
except Exception as e:
    print(f"Note: Index may still be syncing. Error: {e}")
    print("This is expected if indexes were just created. They will be ready shortly.")

# COMMAND ----------


