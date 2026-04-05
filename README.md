# 🧠 Text Summarization: Model Benchmarking with BART, GPT-2 & MLflow

This project demonstrates a lightweight **summarization benchmarking pipeline** using [Hugging Face Transformers](https://github.com/huggingface/transformers), evaluated on the [CNN/DailyMail](https://huggingface.co/datasets/cnn_dailymail) dataset.

It is built for **reproducible experimentation**, with:

- Short training runs on pre-trained models like `facebook/bart-base` and `gpt2`
- Consistent evaluation using ROUGE metrics and generation length
- **MLflow logging** for tracking hyperparameters, metrics, and sample outputs
- A modular CLI-based script: `src/fine_tune.py` (*used for benchmarking, not fine-tuning*)

> 💡 This project is designed to explore how different model types and configurations affect summarization quality — not to perform full-scale fine-tuning.

---

### 🎥 Demo Walkthrough

Want to see the project in action without setting it up?

✅ **Watch the full demo here:**  
[📂 Google Drive Folder – Project Demo](https://drive.google.com/drive/folders/13d9DSMaWFTYVKHugG9ifXBiRbJEaF99F?usp=sharing)

Includes:
- MLflow walkthrough and run comparison  
- Analysis outputs and visualizations  
- Final model usage  
- Live drift detection in action via FastAPI

---

### 🧱 Project Structure

```
text_summarization_project/
├── checkpoints/               # Model checkpoints from small-scale runs
├── checkpoints_final/         # Final model checkpoint from best config
├── deployment/model/          # Saved tokenizer and model for inference
├── fine_tune_analysis/        # Analysis outputs (CSVs, plots)
├── mlruns/                    # MLflow run logs (auto-created)
├── src/
│   ├── fine_tune.py           # CLI-based benchmarking script
│   ├── run_sweep.py           # Loop over all model configs
│   ├── final_train.py         # Run best config at larger scale
│   ├── analysis_scripts/
│   │   ├── extract_fine_tune_results_mlflow.py
│   │   └── fine_tune_analysis.py
│   └── mlops_demo/
│       ├── inference_api.py   # FastAPI app to serve model
│       ├── demo_client.py     # Sends sample requests to server
│       ├── drift_monitor.py   # Drift detection logic
│       └── drift_analyzer.py  # Visualizes drift logs
├── requirements.txt
└── README.md
```
---

### ⚙️ Step 1: Setup Instructions

1. **Clone the repository**:

   ```bash
   git clone https://github.com/yourusername/text_summarization_project.git
   cd text_summarization_project
   ```
Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On macOS/Linux
# Or: venv\Scripts\activate  # On Windows
```

Install dependencies:

```bash
pip install -r requirements.txt
```
(macOS only): Add support for Apple Silicon GPUs:

```python
import os
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
```

🚀 Run a Benchmarking Trial
Use the CLI-based script at src/fine_tune.py to evaluate a summarization model with a chosen config.

▶️ Example: Run BART with small batch and short input/output lengths
```bash
python src/fine_tune.py \
  --model_name_or_path facebook/bart-base \
  --epochs 1 \
  --batch_size 2 \
  --max_input_length 256 \
  --max_target_length 64
```

▶️ Example: Run GPT-2 as a causal LM
```bash
python src/fine_tune.py \
  --model_name_or_path gpt2 \
  --epochs 1 \
  --batch_size 2 \
  --max_input_length 256 \
  --max_target_length 64 \
  --use_causal_lm
```

This will:

- Run a short training + evaluation loop over ~5000 train and ~250 val examples
- Log losses, ROUGE scores, and runtime metrics to MLflow
- Save a few sample predictions as .txt files
- Track hyperparameters and outcomes for comparison across runs

📂 Output Summary
Each run logs:

- 📉 Training & evaluation metrics (loss, ROUGE, speed)
- 📑 Token length stats for predictions and references
- 📝 Sample outputs: article, reference summary, model prediction
- 🧾 Run-specific artifacts via MLflow (including config and example text files)
- 📈 View MLflow Logs

---

### 🧪 Step 2: Reproduce All Experiments
To run the exact 10 benchmarking experiments used in this project, use the src/run_sweep.py script. This script loops through 10 combinations of:

- Model type (facebook/bart-base, gpt2)
- Epochs (1 or 2)
- Batch size (2 or 4)
- Input length (256 or 384)
- Target length (64 or 128)
- Model family (Seq2Seq or Causal LM)

Each experiment is run sequentially and logged via MLflow under the same "local-file" experiment name.

▶️ Run All Benchmarking Sweeps
```bash
python src/run_sweep.py
```

This will:

- Run all 10 model configuration experiments defined in the sweep list
- Automatically handle BART vs. GPT-2 configuration logic
- Log all metrics, ROUGE scores, and samples to MLflow
- Sleep for 5 seconds between runs to ensure system stability

⚠️ Make sure you've already set up your environment and installed requirements before running the sweep.

After it's done, view all runs in one place using the MLflow UI:

```bash
mlflow ui
```
Then go to: http://127.0.0.1:5000 and browse the experiment "local-file".

---

### 📊 Step 3: Analyze Results

Once all experiments have been logged via MLflow, you can extract and analyze the benchmarking results using the following two scripts under `src/analysis_scripts/`.

---

#### 📄 `extract_fine_tune_results_mlflow.py`

This script exports all run metadata and final metrics into a single, clean `.csv` file for further analysis or visualization.

It captures:
- Model config parameters (e.g., model type, batch size, input/output lengths)
- Final ROUGE scores, eval loss, and training speed
- Run info such as status and start time

✅ **Run it after completing all experiments**:

```bash
python src/analysis_scripts/extract_fine_tune_results_mlflow.py
```

This will save the file to:

```
fine_tune_analysis/mlflow_all_model_runs.csv
```

---

#### 📈 `fine_tune_analysis.py`

This script loads all MLflow runs directly, filters the key metrics and parameters, and produces:

- ✅ Leaderboards for best/worst runs
- 📊 A runtime vs. ROUGE-1 scatter plot
- 📊 A ROUGE-1/2/L bar chart (best run per model)
- 📊 A parallel coordinates plot showing hyperparameter impact
- 📁 A `summary.csv` file for downstream Excel / pandas analysis

✅ **Run it anytime after experiments are logged**:

```bash
python src/analysis_scripts/fine_tune_analysis.py
```

This will output:
- 3 figures saved to: `fine_tune_analysis/`
- CSV summary: `fine_tune_analysis/summary.csv`

> 📍 Make sure MLflow still points to the same `mlruns/` directory.

---

### 🏁 Step 4: Run the Best Model at Scale

After completing the sweep and analysis steps, we identified the best-performing configuration (based on ROUGE-1 and eval loss) and ran it on a **larger dataset slice** for a more robust final evaluation.

This was done using the `src/final_train.py` script.

---

#### 📄 `final_train.py`

This script reruns the top configuration from the sweep:
- `facebook/bart-base`
- `1` epoch
- `batch_size=2`
- `max_input_length=384`, `max_target_length=64`

...but scales up the dataset:
- `train`: 25,000 examples  
- `validation`: 1,250 examples  
- `test`: 1,250 examples

It performs:
- ✅ Full training on the larger training split  
- 📊 ROUGE-based evaluation on both validation and test sets  
- 📝 Logging of metrics, lengths, and prediction examples to MLflow  
- 💾 Saving of the final model and tokenizer to `deployment/model/` for downstream use

---

#### ▶️ Run the final training script

```bash
python src/final_train.py
```

This will:
- Log a new MLflow run named `bart-base_final_benchmark_config`
- Store the final model under: `deployment/model/`
- Log validation and test ROUGE scores to MLflow
- Upload sample predictions to MLflow as artifacts

> 📦 The saved model can now be reused for inference or integrated into a downstream application or API.

---

### ⚙️ Step 5: MLOps Demo – Inference & Drift Monitoring

This project includes a complete, lightweight **MLOps simulation** using FastAPI and basic drift monitoring heuristics.

---

#### 🛰️ Start the Inference Server (Terminal 1)

This will load the saved model from `deployment/model/` and expose a `/summarize` endpoint.

```bash
uvicorn src.mlops_demo.inference_api:app --reload --port 8000
```

The server will:
- Tokenize and summarize incoming input
- Log summary token length
- Call drift monitors on:
  - Input entropy & readability
  - Summary length deviation
  - Embedding-based cosine drift

Drift logs are saved to:

```
mlops_demo/drift_logs.log
mlops_demo/alerts.json
```

---

#### 🤖 Run the Client Simulator (Terminal 2)

This script sends both real CNN/DailyMail articles and synthetic "drift-triggering" inputs to the server.

```bash
python src/mlops_demo/demo_client.py
```

The script simulates:
- Normal requests from the dataset (real-world cases)
- Drift cases including:
  - Short or very long inputs
  - Low entropy (repeating tokens)
  - High entropy (gibberish)
  - Embedding-based novelty

Each result includes latency, token count, and a truncated summary.

---

#### 📉 Drift Logging & Monitoring

The following drift types are monitored in `drift_monitor.py`:

- **Output Length Drift**  
  Triggers when average summary length deviates from baseline (56 tokens ±10)
- **Input Entropy Drift**  
  Flags abnormally repetitive or highly chaotic input
- **Embedding Drift**  
  Computes cosine distance to a reference embedding baseline

Drift alerts are logged to:
- `mlops_demo/drift_logs.log` (for audit/debug)
- `mlops_demo/alerts.json` (for alert storage)

---

#### 📊 Visualize Drift Over Time

Use the following script to generate 4 time-series plots based on the logs:

```bash
python src/mlops_demo/drift_analyzer.py
```

It will show:
- Input length over time  
- Input entropy and entropy-based drift  
- Summary length and output drift  
- Embedding cosine distance vs. threshold  

---

> 🖥️ **Note:** You must run the server (`uvicorn ...`) and the client (`python demo_client.py`) in **separate terminals** at the same time.

---

### 🚀 Next Steps

- ✅ Add Docker support for containerized deployment
- ✅ Integrate real-time metrics dashboard for live monitoring
- 🧪 Experiment with larger models (e.g., `bart-large`, `t5-base`)
- 🧠 Incorporate LoRA or quantization for efficient fine-tuning
- 🌐 Deploy API to a cloud platform (e.g., Hugging Face Spaces, Render)

> Contributions and ideas are welcome!

---
### 👋 Contact

Built by **Miray Özcan**  
📧 `miray@uni.minerva.edu`  
🌐 [linkedin.com/in/mirayozcan](https://linkedin.com/in/mirayozcan)

> If you found this useful or want to collaborate, feel free to reach out!
