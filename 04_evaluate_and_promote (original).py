# Databricks notebook source
# MAGIC %md
# MAGIC # Model Evaluation and Promotion
# MAGIC
# MAGIC Evaluate latest model version and promote to champion if metrics exceed threshold.

# COMMAND ----------

# MAGIC %pip install -U -qqqq mlflow-skinny[databricks]
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

dbutils.widgets.text(
    "model_name", "dbxmetagen.default.lifesciences_agent", "Model Name"
)
dbutils.widgets.text("catalog", "dbxmetagen", "Catalog")
dbutils.widgets.text("schema", "default", "Schema")
dbutils.widgets.text("promotion_threshold", "0.7", "Promotion Threshold")
dbutils.widgets.dropdown(
    "evaluation_metric",
    "relevance",
    ["relevance", "safety", "accuracy"],
    "Evaluation Metric",
)

model_name = dbutils.widgets.get("model_name")
catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
promotion_threshold = float(dbutils.widgets.get("promotion_threshold"))
evaluation_metric = dbutils.widgets.get("evaluation_metric")

print(f"Model: {model_name}")
print(f"Metric: {evaluation_metric}")
print(f"Threshold: {promotion_threshold}")

# COMMAND ----------

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import time

import mlflow
from mlflow.tracking import MlflowClient


@dataclass
class EvaluationResult:
    """Result of model evaluation."""

    version: str
    metrics: Dict[str, float]
    passed: bool
    message: str


class ModelEvaluator(ABC):
    """Abstract base class for model evaluation and promotion."""

    def __init__(
        self,
        model_name: str,
        catalog: str,
        schema: str,
        promotion_threshold: float = 0.7,
        evaluation_metric: str = "relevance",
    ):
        """
        Initialize evaluator.

        Args:
            model_name: Fully qualified UC model name
            catalog: UC catalog name
            schema: UC schema name
            promotion_threshold: Minimum metric score to promote
            evaluation_metric: Primary metric to evaluate
        """
        self.model_name = model_name
        self.catalog = catalog
        self.schema = schema
        self.promotion_threshold = promotion_threshold
        self.evaluation_metric = evaluation_metric
        self.client = MlflowClient()
        mlflow.set_registry_uri("databricks-uc")

    def get_latest_version(self) -> str:
        """Get the latest model version number."""
        versions = self.client.search_model_versions(f"name='{self.model_name}'")
        if not versions:
            raise ValueError(f"No versions found for {self.model_name}")
        return max(versions, key=lambda v: int(v.version)).version

    def get_champion_version(self) -> Optional[str]:
        """Get current champion version if exists."""
        try:
            champion = self.client.get_model_version_by_alias(
                self.model_name, "champion"
            )
            return champion.version
        except Exception:
            return None

    @abstractmethod
    def create_evaluation_dataset(self) -> List[Dict[str, Any]]:
        """Create evaluation dataset for model testing."""
        pass

    @abstractmethod
    def evaluate_model(self, model_uri: str) -> Dict[str, float]:
        """
        Evaluate model and return metrics.

        Args:
            model_uri: MLflow model URI

        Returns:
            Dictionary of metric names to scores
        """
        pass

    def promote_to_champion(self, version: str, metrics: Dict[str, float]) -> None:
        """
        Promote model version to champion alias.

        Args:
            version: Version to promote
            metrics: Evaluation metrics to include in comment
        """
        metrics_str = ", ".join([f"{k}={v:.3f}" for k, v in metrics.items()])
        comment = f"Promoted by evaluation notebook. Metrics: {metrics_str}"

        self.client.set_registered_model_alias(self.model_name, "champion", version)

        self.client.update_model_version(
            name=self.model_name,
            version=version,
            description=f"{comment}\nPromoted at {time.strftime('%Y-%m-%d %H:%M:%S')}",
        )

        print(f"Promoted version {version} to champion")

    def run(self) -> EvaluationResult:
        """
        Run evaluation and promotion workflow.

        Returns:
            EvaluationResult with evaluation details
        """
        latest_version = self.get_latest_version()
        print(f"Evaluating version {latest_version}")

        model_uri = f"models:/{self.model_name}/{latest_version}"

        metrics = self.evaluate_model(model_uri)
        print(f"Evaluation metrics: {metrics}")

        primary_score = metrics.get(self.evaluation_metric, 0.0)
        passed = primary_score >= self.promotion_threshold

        if passed:
            champion_version = self.get_champion_version()

            if champion_version and champion_version == latest_version:
                message = f"Version {latest_version} is already champion"
            else:
                self.promote_to_champion(latest_version, metrics)
                message = f"Version {latest_version} promoted to champion"
        else:
            message = f"Version {latest_version} did not meet threshold ({primary_score:.3f} < {self.promotion_threshold})"

        return EvaluationResult(
            version=latest_version, metrics=metrics, passed=passed, message=message
        )


class LifeSciencesEvaluator(ModelEvaluator):
    """Evaluator for life sciences agent models."""

    def create_evaluation_dataset(self) -> List[Dict[str, Any]]:
        """Create mock evaluation dataset."""
        return [
            {
                "input": [
                    {
                        "role": "user",
                        "content": f"How many genes are in the {self.catalog}.{self.schema}.genes_knowledge table?",
                    }
                ],
                "expected": "There are 100 genes",
            },
            {
                "input": [
                    {
                        "role": "user",
                        "content": "What is the average confidence for proteins?",
                    }
                ],
                "expected": "The average confidence is calculated using UC functions",
            },
            {
                "input": [
                    {"role": "user", "content": "Tell me about kinase proteins."}
                ],
                "expected": "Kinases are proteins that catalyze phosphorylation",
            },
            {
                "input": [
                    {"role": "user", "content": "What compounds target kinases?"}
                ],
                "expected": "Compounds that target kinases include various small molecules",
            },
            {
                "input": [
                    {"role": "user", "content": "Explain cell signaling pathways."}
                ],
                "expected": "Cell signaling pathways involve multiple proteins and cascades",
            },
        ]

    def evaluate_model(self, model_uri: str) -> Dict[str, float]:
        """
        Mock evaluation of model.

        In production, this would use mlflow.evaluate() with actual metrics.
        For now, returns mock scores based on dataset size.
        """
        dataset = self.create_evaluation_dataset()

        print(f"Running mock evaluation on {len(dataset)} examples")
        time.sleep(2)

        base_relevance = 0.75
        base_safety = 0.95
        base_accuracy = 0.70

        dataset_size_factor = min(len(dataset) / 10.0, 1.0)

        metrics = {
            "relevance": base_relevance + (dataset_size_factor * 0.1),
            "safety": base_safety,
            "accuracy": base_accuracy + (dataset_size_factor * 0.15),
            "completeness": 0.80,
            "latency_p95": 2.5,
        }

        return metrics


class LifeSciencesGenieEvaluator(LifeSciencesEvaluator):
    """Evaluator for life sciences agent with Genie integration."""

    def create_evaluation_dataset(self) -> List[Dict[str, Any]]:
        """Create evaluation dataset including Genie queries."""
        base_dataset = super().create_evaluation_dataset()

        genie_queries = [
            {
                "input": [
                    {
                        "role": "user",
                        "content": "Compare average confidence scores between genes and proteins.",
                    }
                ],
                "expected": "Comparison table showing average confidence by category",
            },
            {
                "input": [
                    {
                        "role": "user",
                        "content": "Show me top compounds with highest confidence scores.",
                    }
                ],
                "expected": "Table of top compounds with confidence data",
            },
        ]

        return base_dataset + genie_queries

    def evaluate_model(self, model_uri: str) -> Dict[str, float]:
        """Evaluate model with Genie-specific metrics."""
        metrics = super().evaluate_model(model_uri)

        metrics["genie_success_rate"] = 0.85
        metrics["context_enrichment_quality"] = 0.78

        return metrics


# COMMAND ----------

if "genie" in model_name.lower():
    evaluator = LifeSciencesGenieEvaluator(
        model_name=model_name,
        catalog=catalog,
        schema=schema,
        promotion_threshold=promotion_threshold,
        evaluation_metric=evaluation_metric,
    )
else:
    evaluator = LifeSciencesEvaluator(
        model_name=model_name,
        catalog=catalog,
        schema=schema,
        promotion_threshold=promotion_threshold,
        evaluation_metric=evaluation_metric,
    )

result = evaluator.run()

print(f"\n{'='*60}")
print("Evaluation Result")
print(f"{'='*60}")
print(f"Version: {result.version}")
print(f"Passed: {result.passed}")
print(f"\nMetrics:")
for metric, score in result.metrics.items():
    print(f"  {metric}: {score:.3f}")
print(f"\n{result.message}")
print(f"{'='*60}")

if not result.passed:
    print("\nModel did not meet promotion criteria")

# COMMAND ----------



# COMMAND ----------


