# Databricks notebook source
# MAGIC %md
# MAGIC # Agent Deployment
# MAGIC
# MAGIC Generic deployment system for agent models using abstract base classes.

# COMMAND ----------

# MAGIC %pip install -U -qqqq databricks-agents mlflow-skinny[databricks] databricks-sdk
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

dbutils.widgets.text(
    "model_name", "dbxmetagen.default.lifesciences_agent", "Model Name"
)
dbutils.widgets.text("endpoint_name", "", "Endpoint Name (optional)")
dbutils.widgets.text("catalog", "dbxmetagen", "Catalog")
dbutils.widgets.text("schema", "default", "Schema")
dbutils.widgets.dropdown(
    "model_alias", "champion", ["champion", "challenger", "latest"], "Model Alias"
)

model_name = dbutils.widgets.get("model_name")
endpoint_name = dbutils.widgets.get("endpoint_name")
catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
model_alias = dbutils.widgets.get("model_alias")

if not endpoint_name:
    endpoint_name = f"agents_{model_name.replace('.', '-')}"

print(f"Model: {model_name}")
print(f"Alias: {model_alias}")
print(f"Endpoint: {endpoint_name}")
print(f"Catalog: {catalog}, Schema: {schema}")

# COMMAND ----------

# DBTITLE 1,Define Deployment Class Functions
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import time

import mlflow
from mlflow.tracking import MlflowClient
from mlflow.models.resources import (
    DatabricksFunction,
    DatabricksTable,
    DatabricksGenieSpace,
)
from databricks import agents
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import EndpointStateReady


@dataclass
class DeploymentResult:
    """Result of a deployment operation."""

    success: bool
    endpoint_name: str
    model_version: Optional[str] = None
    message: Optional[str] = None


class AgentDeployer(ABC):
    """Abstract base class for agent deployment."""

    def __init__(
        self,
        model_name: str,
        endpoint_name: str,
        catalog: str,
        schema: str,
        model_alias: str = "champion",
    ):
        """
        Initialize deployer.

        Args:
            model_name: Fully qualified UC model name
            endpoint_name: Name for serving endpoint
            catalog: UC catalog name
            schema: UC schema name
            model_alias: Model alias to deploy (champion/challenger/latest)
        """
        self.model_name = model_name
        self.endpoint_name = endpoint_name
        self.catalog = catalog
        self.schema = schema
        self.model_alias = model_alias
        self.client = MlflowClient()
        self.workspace = WorkspaceClient()

    @abstractmethod
    def get_model_resources(self) -> List[Any]:
        """Get list of Databricks resources required by the model."""
        pass

    @abstractmethod
    def get_deployment_config(self) -> Dict[str, Any]:
        """Get deployment configuration (tags, etc)."""
        pass

    def get_model_version(self) -> str:
        """Fetch model version by alias."""
        mlflow.set_registry_uri("databricks-uc")

        if self.model_alias == "latest":
            versions = self.client.search_model_versions(f"name='{self.model_name}'")
            if not versions:
                raise ValueError(f"No versions found for {self.model_name}")
            return max(versions, key=lambda v: int(v.version)).version
        else:
            model_version = self.client.get_model_version_by_alias(
                self.model_name, self.model_alias
            )
            return model_version.version

    def check_endpoint_health(self) -> bool:
        """Check if endpoint exists and its current state."""
        try:
            endpoint = self.workspace.serving_endpoints.get(self.endpoint_name)
            state = (
                endpoint.state.ready if endpoint.state else EndpointStateReady.NOT_READY
            )

            if (
                state == EndpointStateReady.NOT_READY
                or endpoint.state.config_update == "UPDATE_FAILED"
            ):
                print(f"Endpoint in unhealthy state: {state}")
                return False

            if endpoint.config is None:
                print("Endpoint config is None")
                return False

            print(f"Endpoint exists with state: {state}")
            return True

        except Exception as e:
            if "RESOURCE_DOES_NOT_EXIST" in str(e):
                print("Endpoint does not exist")
                return True
            print(f"Error checking endpoint: {e}")
            return False

    def cleanup(self) -> None:
        """Clean up failed endpoint."""
        try:
            self.workspace.serving_endpoints.delete(self.endpoint_name)
            print(f"Deleted endpoint: {self.endpoint_name}")
            time.sleep(60)
        except Exception as e:
            if "RESOURCE_DOES_NOT_EXIST" not in str(e):
                print(f"Error during cleanup: {e}")

    def deploy(self, max_retries: int = 3, retry_delay: int = 60) -> DeploymentResult:
        """
        Deploy model to serving endpoint with retry logic.

        Args:
            max_retries: Maximum number of deployment attempts
            retry_delay: Seconds to wait between retries

        Returns:
            DeploymentResult with status and details
        """
        try:
            version = self.get_model_version()
            print(f"Deploying version {version}")
        except Exception as e:
            return DeploymentResult(
                success=False,
                endpoint_name=self.endpoint_name,
                message=f"Failed to fetch model version: {e}",
            )

        if not self.check_endpoint_health():
            print("Cleaning up unhealthy endpoint")
            self.cleanup()

        config = self.get_deployment_config()

        for attempt in range(max_retries):
            try:
                print(f"Deployment attempt {attempt + 1}/{max_retries}")

                # Note: Don't pass tags on updates to avoid duplicate key errors
                agents.deploy(
                    self.model_name,
                    version,
                    deploy_feedback_model=False,
                )

                print(f"Deployment successful: {self.endpoint_name}")
                return DeploymentResult(
                    success=True,
                    endpoint_name=self.endpoint_name,
                    model_version=version,
                    message="Deployment completed successfully",
                )

            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {e}")

                if attempt < max_retries - 1:
                    print(f"Retrying in {retry_delay}s")
                    time.sleep(retry_delay)
                else:
                    return DeploymentResult(
                        success=False,
                        endpoint_name=self.endpoint_name,
                        model_version=version,
                        message=f"Deployment failed after {max_retries} attempts: {e}",
                    )


class LifeSciencesAgentDeployer(AgentDeployer):
    """Deployer for life sciences agent models."""

    def get_model_resources(self) -> List[Any]:
        """Get UC functions, tables, and vector search indexes."""
        resources = []

        tables = [
            "genes_knowledge",
            "proteins_knowledge",
            "pathways_knowledge",
            "compounds_knowledge",
        ]

        for table in tables:
            resources.append(
                DatabricksTable(table_name=f"{self.catalog}.{self.schema}.{table}")
            )

            resources.append(
                DatabricksFunction(
                    function_name=f"{self.catalog}.{self.schema}.count_{table}_rows"
                )
            )
            resources.append(
                DatabricksFunction(
                    function_name=f"{self.catalog}.{self.schema}.count_high_confidence_{table}"
                )
            )
            resources.append(
                DatabricksFunction(
                    function_name=f"{self.catalog}.{self.schema}.avg_confidence_{table}"
                )
            )

        return resources

    def get_deployment_config(self) -> Dict[str, Any]:
        """Get deployment tags and configuration."""
        return {
            "tags": {
                "domain": "life-sciences",
                "architecture": "orchestrator-synthesizer",
                "deployed_by": "deployment_notebook",
            }
        }


class LifeSciencesGenieAgentDeployer(LifeSciencesAgentDeployer):
    """Deployer for life sciences agent with Genie integration."""

    def __init__(self, *args, genie_space_id: str, **kwargs):
        super().__init__(*args, **kwargs)
        self.genie_space_id = genie_space_id

    def get_model_resources(self) -> List[Any]:
        """Get resources including Genie space."""
        resources = super().get_model_resources()
        resources.append(DatabricksGenieSpace(genie_space_id=self.genie_space_id))
        return resources

    def get_deployment_config(self) -> Dict[str, Any]:
        """Get deployment configuration with Genie."""
        config = super().get_deployment_config()
        config["tags"]["genie_enabled"] = "true"
        return config


# COMMAND ----------

# DBTITLE 1,Deploy Model
if "genie" in model_name.lower():
    deployer = LifeSciencesGenieAgentDeployer(
        model_name=model_name,
        endpoint_name=endpoint_name,
        catalog=catalog,
        schema=schema,
        model_alias=model_alias,
        genie_space_id="01f0c64ba4c61bd49b1aa03af847407a",
    )
else:
    deployer = LifeSciencesAgentDeployer(
        model_name=model_name,
        endpoint_name=endpoint_name,
        catalog=catalog,
        schema=schema,
        model_alias=model_alias,
    )

result = deployer.deploy()

if result.success:
    print(f"\nDeployment successful:")
    print(f"  Endpoint: {result.endpoint_name}")
    print(f"  Version: {result.model_version}")
else:
    print(f"\nDeployment failed:")
    print(f"  {result.message}")
    raise Exception(result.message)
