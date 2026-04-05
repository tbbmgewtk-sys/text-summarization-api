import mlflow
import pandas as pd
from pathlib import Path

# STEP 0: Get the script directory and define save path
current_script_path = Path(__file__).resolve()
project_root = current_script_path.parents[2]
save_dir = project_root / "fine_tune_analysis"
save_dir.mkdir(parents=True, exist_ok=True)  # ensures directory exists
save_path = save_dir / "mlflow_all_model_runs.csv"

# STEP 1: Connect to MLflow
mlflow.set_tracking_uri("")  # or leave blank for local

# STEP 2: Load experiment
experiment_name = "local-file"
experiment = mlflow.get_experiment_by_name(experiment_name)
experiment_id = experiment.experiment_id

# STEP 3: Get runs
runs_df = mlflow.search_runs(experiment_ids=[experiment_id], output_format="pandas")

# STEP 4: Select relevant columns
run_name = ["tags.mlflow.runName"]
params = [col for col in runs_df.columns if col.startswith("params.")]
metrics = [col for col in runs_df.columns if col.startswith("metrics.")]
info = ["run_id", "start_time", "status"]

summary_df = runs_df[info + run_name + params + metrics]

# STEP 5: Save CSV to target directory
summary_df.to_csv(save_path, index=False)
print(f"✅ CSV saved to: {save_path}")