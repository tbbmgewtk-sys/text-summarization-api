from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import os

# ───────── Import drift monitoring functions ─────────
from src.mlops_demo import drift_monitor

# ───────── Load model & tokenizer ─────────
MODEL_DIR = "deployment/model"
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model.to(device)
model.eval()

# ───────── FastAPI app ─────────
app = FastAPI()

# ───────── Input schema ─────────
class SummarizationInput(BaseModel):
    text: str

# ───────── Inference route ─────────
@app.post("/summarize")
def summarize(input: SummarizationInput):
    # Tokenize input
    input_tensor = tokenizer(input.text, return_tensors="pt", truncation=True, max_length=384).to(device)

    # Generate summary
    with torch.no_grad():
        output_ids = model.generate(
            **input_tensor,
            max_new_tokens=64,
            num_beams=4,
            no_repeat_ngram_size=3
        )
    summary = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    summary_tokens = tokenizer.tokenize(summary)

    # ─── Drift Monitoring ───
    drift_monitor.monitor_input_stats(input.text, tokenizer)
    drift_monitor.monitor_output_length(len(summary_tokens))
    drift_monitor.monitor_input_embedding(input_tensor, model)

    return {
        "summary": summary,
        "tokens": len(summary_tokens)
    }