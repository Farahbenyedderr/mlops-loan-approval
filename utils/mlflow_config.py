"""
MLflow Configuration Module
Handles MLflow setup, tracking, and model registry operations
"""
import os
from datetime import datetime
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient


class MLflowConfig:
    """Centralized MLflow configuration and utilities."""
    
    def __init__(
        self,
        experiment_name="Loan_Prediction_Experiment",
        tracking_uri=None,
        artifact_location=None
    ):
        """
        Initialize MLflow configuration.
        
        Args:
            experiment_name: Name of the MLflow experiment
            tracking_uri: MLflow tracking server URI (default: sqlite)
            artifact_location: Path to store artifacts
        """
        self.experiment_name = experiment_name
        
        # Setup paths
        self.base_dir = Path("mlflow_data")
        self.base_dir.mkdir(exist_ok=True)
        
        # Default to SQLite backend
        if tracking_uri is None:
            self.tracking_uri = f"sqlite:///{self.base_dir}/mlflow.db"
        else:
            self.tracking_uri = tracking_uri
            
        # Artifact location
        if artifact_location is None:
            self.artifact_location = str(self.base_dir / "artifacts")
        else:
            self.artifact_location = artifact_location
            
        # Create artifact directory
        Path(self.artifact_location).mkdir(parents=True, exist_ok=True)
        
        # Setup MLflow
        self._setup_mlflow()
        
    def _setup_mlflow(self):
        """Configure MLflow tracking and experiment."""
        # Set tracking URI
        mlflow.set_tracking_uri(self.tracking_uri)
        
        # Set or create experiment
        try:
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            if experiment is None:
                experiment_id = mlflow.create_experiment(
                    self.experiment_name,
                    artifact_location=self.artifact_location
                )
                print(f"✓ Created new experiment: {self.experiment_name} (ID: {experiment_id})")
            else:
                experiment_id = experiment.experiment_id
                print(f"✓ Using existing experiment: {self.experiment_name} (ID: {experiment_id})")
                
            mlflow.set_experiment(self.experiment_name)
            
        except Exception as e:
            print(f"Error setting up experiment: {e}")
            raise
            
        self.client = MlflowClient(tracking_uri=self.tracking_uri)
        
    def start_run(self, run_name=None, tags=None):
        """
        Start an MLflow run with timestamp-based naming.
        
        Args:
            run_name: Custom run name (auto-generated if None)
            tags: Dictionary of tags to add to the run
            
        Returns:
            MLflow run context manager
        """
        if run_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_name = f"run_{timestamp}"
            
        # Default tags
        default_tags = {
            "timestamp": datetime.now().isoformat(),
            "model_type": "RandomForest"
        }
        
        if tags:
            default_tags.update(tags)
            
        return mlflow.start_run(run_name=run_name, tags=default_tags)
    
    def log_model_with_metadata(
        self,
        model,
        model_name="LoanDefaultModel_RF",
        signature=None,
        input_example=None,
        pip_requirements=None
    ):
        """
        Log model with comprehensive metadata and versioning.
        
        Args:
            model: Trained sklearn model
            model_name: Name for model registry
            signature: MLflow model signature
            input_example: Example input for model
            pip_requirements: List of package requirements
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Log the model with metadata
        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name=model_name,
            signature=signature,
            input_example=input_example,
            pip_requirements=pip_requirements
        )
        
        # Add custom tags to the model version
        run_id = mlflow.active_run().info.run_id
        
        # Get the latest version number
        try:
            versions = self.client.search_model_versions(f"name='{model_name}'")
            if versions:
                latest_version = max([int(v.version) for v in versions])
                
                # Add tags to model version
                self.client.set_model_version_tag(
                    name=model_name,
                    version=str(latest_version),
                    key="timestamp",
                    value=timestamp
                )
                self.client.set_model_version_tag(
                    name=model_name,
                    version=str(latest_version),
                    key="run_id",
                    value=run_id
                )
                
                print(f"✓ Model logged: {model_name} v{latest_version} [{timestamp}]")
        except Exception as e:
            print(f"Warning: Could not tag model version: {e}")
            
        return model_info
    
    def log_comprehensive_metrics(self, metrics_dict):
        """
        Log metrics with additional computed metrics.
        
        Args:
            metrics_dict: Dictionary of metrics to log
        """
        # Log all metrics
        mlflow.log_metrics(metrics_dict)
        
        # Compute and log additional metrics
        if "precision" in metrics_dict and "recall" in metrics_dict:
            precision = metrics_dict["precision"]
            recall = metrics_dict["recall"]
            
            # F-beta scores
            if precision > 0 or recall > 0:
                # F2 score (emphasizes recall)
                f2 = (5 * precision * recall) / (4 * precision + recall) if (4 * precision + recall) > 0 else 0
                mlflow.log_metric("f2_score", f2)
                
                # F0.5 score (emphasizes precision)
                f05 = (1.25 * precision * recall) / (0.25 * precision + recall) if (0.25 * precision + recall) > 0 else 0
                mlflow.log_metric("f0.5_score", f05)
        
        print(f"✓ Logged {len(metrics_dict)} metrics to MLflow")
    
    def log_dataset_info(self, train_df, test_df):
        """
        Log dataset statistics and information.
        
        Args:
            train_df: Training dataframe
            test_df: Testing dataframe
        """
        dataset_info = {
            "train_samples": len(train_df),
            "test_samples": len(test_df),
            "train_positive_ratio": train_df["loan_status"].mean() if "loan_status" in train_df.columns else 0,
            "test_positive_ratio": test_df["loan_status"].mean() if "loan_status" in test_df.columns else 0,
            "n_features": len(train_df.columns) - 1
        }
        
        mlflow.log_params(dataset_info)
        print(f"✓ Logged dataset info: {dataset_info['train_samples']} train, {dataset_info['test_samples']} test samples")
    
    def log_feature_importance(self, model, feature_names):
        """
        Log feature importance as artifact.
        
        Args:
            model: Trained model with feature_importances_
            feature_names: List of feature names
        """
        if hasattr(model, 'feature_importances_'):
            import pandas as pd
            
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            # Save as artifact
            importance_path = "feature_importance.csv"
            importance_df.to_csv(importance_path, index=False)
            mlflow.log_artifact(importance_path)
            
            # Clean up temporary file
            os.remove(importance_path)
            
            print(f"✓ Logged feature importance for {len(feature_names)} features")
    
    def get_best_model(self, model_name="LoanDefaultModel_RF", metric="accuracy"):
        """
        Retrieve the best model version based on a metric.
        
        Args:
            model_name: Registered model name
            metric: Metric to use for comparison
            
        Returns:
            Best model version info
        """
        try:
            versions = self.client.search_model_versions(f"name='{model_name}'")
            
            if not versions:
                print(f"No versions found for model: {model_name}")
                return None
            
            best_version = None
            best_metric_value = -float('inf')
            
            for version in versions:
                run_id = version.run_id
                run = self.client.get_run(run_id)
                
                metric_value = run.data.metrics.get(metric, -float('inf'))
                
                if metric_value > best_metric_value:
                    best_metric_value = metric_value
                    best_version = version
            
            if best_version:
                print(f"✓ Best model: {model_name} v{best_version.version} ({metric}={best_metric_value:.4f})")
                return best_version
            
        except Exception as e:
            print(f"Error retrieving best model: {e}")
            return None
    
    def transition_model_stage(self, model_name, version, stage):
        """
        Transition model to a specific stage.
        
        Args:
            model_name: Registered model name
            version: Model version number
            stage: Target stage ('Staging', 'Production', 'Archived')
        """
        try:
            self.client.transition_model_version_stage(
                name=model_name,
                version=version,
                stage=stage
            )
            print(f"✓ Transitioned {model_name} v{version} to {stage}")
        except Exception as e:
            print(f"Error transitioning model stage: {e}")


def get_default_config():
    """Get default MLflow configuration."""
    return MLflowConfig(
        experiment_name="Loan_Prediction_Experiment",
        tracking_uri=None,  # Uses SQLite by default
        artifact_location=None  # Uses local filesystem
    )