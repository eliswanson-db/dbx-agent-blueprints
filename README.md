# Agent Architecture Blueprints

Example agents demonstrating different RAG and agentic patterns for life sciences applications.

## Notebooks

### 01_setup_tables_and_vector_search.py
Sets up the infrastructure:
- Creates synthetic life sciences knowledge bases (genes, proteins, pathways, compounds)
- Creates Delta tables
- Sets up vector search indexes
- Creates Unity Catalog functions for aggregations

### 02a_simple_rag.py
**Simple RAG Chain** - Basic retrieval augmented generation
- Single agent node with vector search tool
- Single vector search index (genes knowledge)
- Agent handles both retrieval and synthesis
- No UC tools or parallel workers
- Best for: Simple semantic questions about genes

### 02b_parallel_rag.py
**Parallel RAG with Orchestrator-Synthesizer Pattern**
- Orchestrator routes between SQL worker and vector workers
- Two parallel vector search workers (genes/proteins and pathways/compounds)
- UC functions for simple aggregations
- Confidence-based synthesis
- Added a better judge
- Best for: Complex queries requiring multiple knowledge sources

### 03_context_for_genie.py
**Context-Enriched Genie Integration**
- 4-way routing: SQL, Direct Genie, Vector+Genie, Vector-only
- Parallel vector search workers fetch context for ambiguous queries
- Context rewriter enriches questions with entity IDs before Genie query
- Quality check for Genie responses with retry logic
- Best for: Structured data queries with Genie, semantic lookup + aggregation

### 04_evaluate_and_promote.py
Model evaluation and promotion workflow:
- Mock evaluation with configurable metrics
- Promotes models to champion alias based on thresholds
- Abstract base classes for extensibility

### 05_deploy_agent.py
Generic deployment system for agent models:
- Deploys models with champion/challenger/latest aliases
- Health checks and cleanup for failed endpoints
- Abstract base classes for different agent types

## Dependencies

| Library | License | Purpose | Version Range |
|---------|---------|---------|---------------|
| langgraph | MIT | Agent orchestration and state management | >=0.2.0 |
| databricks-agents | Databricks License | Agent deployment and management | latest |
| databricks-langchain | Apache 2.0 | LangChain integration for Databricks | latest |
| databricks-vectorsearch | Apache 2.0 | Vector search retrieval | latest |
| databricks-sdk | Apache 2.0 | Databricks workspace API client | latest |
| mlflow-skinny[databricks] | Apache 2.0 | Model tracking and registry | latest |
| psycopg[binary] | LGPL v3+ | PostgreSQL adapter (LangChain checkpointing) | >=3.0 |
| databricks-connect | Databricks License | Databricks Connect runtime | latest |

## Deployment

Use Databricks Asset Bundles to deploy:

```bash
# Validate bundle configuration
databricks bundle validate

# Deploy to dev environment
databricks bundle deploy -t dev

# Run specific job
databricks bundle run lifesciences_setup_job -t dev
```

## Jobs Configuration

The `resources/jobs.yml` defines several workflows:

1. **lifesciences_setup_job** - Infrastructure setup
2. **simple_rag_agent_deployment_job** - Deploy simple RAG agent
3. **parallel_rag_agent_deployment_job** - Deploy parallel RAG agent
4. **context_for_genie_agent_deployment_job** - Deploy Genie-integrated agent
5. **lifesciences_parallel_rag_workflow** - Build → Evaluate → Deploy (parallel RAG)
6. **lifesciences_context_for_genie_workflow** - Build → Evaluate → Deploy (Genie agent)

## Architecture Patterns

### Simple RAG Chain (02a)
```
User Query → Agent (with vector search tool) → Answer
```

### Parallel RAG with Orchestrator (02b)
```
User Query → Orchestrator → [SQL Worker OR Vector Workers (parallel)] → Synthesizer → Answer
```

### Context for Genie (03)
```
User Query → Orchestrator → [SQL OR Direct Genie OR Vector→Context→Genie OR Vector→Synthesizer] → Answer
```

## License

See LICENSE file for details.
