# ────────────────────────────  src/analysis.py  ──────────────────────────────
# run me with:   python src/analysis.py
# -----------------------------------------------------------------------------
import os, mlflow, pandas as pd, numpy as np, matplotlib.pyplot as plt
from pathlib import Path

# ───────── configuration ─────────
MLRUNS_DIR  = "mlruns"          # where fine_tune.py saves stuff
EXPERIMENT  = "local-file"      # the experiment name used in fine_tune.py
TOP_K       = 5                 # how many rows to show in leader boards
OUT_DIR     = Path("fine_tune_analysis")  # everything will be written here
OUT_DIR.mkdir(exist_ok=True)

# ───────── load runs from MLflow ─────────
client = mlflow.MlflowClient(tracking_uri=f"file://{Path(MLRUNS_DIR).resolve()}")
exp    = client.get_experiment_by_name(EXPERIMENT)
if exp is None:
    raise SystemExit(f"No experiment named “{EXPERIMENT}” found in {MLRUNS_DIR}")

runs   = mlflow.search_runs(exp.experiment_id)
if runs.empty:
    raise SystemExit("No runs found – did you run fine_tune.py / run_all.py ?")

pars = [
    "params.model_name_or_path", "params.epochs", "params.batch_size",
    "params.max_input_length",   "params.max_target_length", "params.use_causal_lm",
]
mets = [
    "metrics.rouge1", "metrics.rouge2", "metrics.rougeL",
    "metrics.eval_loss", "metrics.train_runtime"
]
df   = runs[pars+mets].copy()
df.columns = [c.split(".",1)[1] for c in df.columns]       # nicer headers
df = df.apply(pd.to_numeric, errors="ignore")              # cast numbers

# add short name for legends
def short(m): return "GPT-2" if "gpt2" in m else "BART"
df["model"] = df["model_name_or_path"].apply(short)

# ───────── leader boards ─────────
pd.set_option("display.precision", 2)
print("\n===  Top-{} by ROUGE-1  ===".format(TOP_K))
print(df.sort_values("rouge1", ascending=False).head(TOP_K).to_markdown(index=False))

print("\n===  Worst-{} by eval_loss  ===".format(TOP_K))
print(df.sort_values("eval_loss", ascending=False).head(TOP_K).to_markdown(index=False))

print("\n===  Fastest-{} runs  ===".format(TOP_K))
print(df.sort_values("train_runtime").head(TOP_K).to_markdown(index=False))

# ───────── plots ─────────
try:
    import seaborn as sns; sns.set_theme(style="whitegrid")
except ImportError:
    pass  # fall back to plain matplotlib

# 1) run-time vs ROUGE-1
plt.figure(figsize=(6,4))
for m,g in df.groupby("model"):
    plt.scatter(g["train_runtime"], g["rouge1"], label=m, s=60, alpha=.8)
plt.xlabel("train_runtime  [s]")
plt.ylabel("ROUGE-1  [%]")
plt.title("Training time vs ROUGE-1")
plt.legend()
plt.tight_layout()
plt.savefig(OUT_DIR/"runtime_vs_rouge1.png", dpi=150)
plt.close()

# 2) best run per model – ROUGE triple
best = df.sort_values("rouge1", ascending=False).groupby("model").head(1)
x = np.arange(len(best))
plt.figure(figsize=(6,4))
plt.bar(x-0.2, best["rouge1"], 0.18, label="R-1")
plt.bar(x,      best["rouge2"], 0.18, label="R-2")
plt.bar(x+0.2,  best["rougeL"], 0.18, label="R-L")
plt.xticks(x, best["model"])
plt.ylabel("score  [%]")
plt.title("Best run per model")
plt.legend()
plt.tight_layout()
plt.savefig(OUT_DIR/"rouge_family_best.png", dpi=150)
plt.close()

# 3) parallel coordinates (needs pandas-plotting installed by default)
from pandas.plotting import parallel_coordinates
plt.figure(figsize=(8,4))
parallel_coordinates(
    df[["model","epochs","batch_size","max_input_length","rouge1"]], "model",
    colormap="Set2", alpha=.7
)
plt.title("Hyper-params vs ROUGE-1 (parallel-coords)")
plt.tight_layout()
plt.savefig(OUT_DIR/"parallel_coords.png", dpi=150)
plt.close()

# ───────── export CSV ─────────
csv_path = OUT_DIR/"summary.csv"
df.to_csv(csv_path, index=False)
print(f"\n✓  Saved merged CSV + 3 figures to   {OUT_DIR.resolve()}")
print("   Open the PNGs or carry on analysing summary.csv in pandas / Excel.")