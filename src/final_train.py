import os
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
import argparse, warnings, torch, mlflow, numpy as np, random, shutil, tempfile
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments
)
from evaluate import load
import nltk
from tqdm import tqdm

# ───────────────── Warning filters ─────────────────
warnings.filterwarnings("ignore", message="Redirects are currently not supported in Windows or MacOs")
warnings.filterwarnings("ignore", message="`resume_download` is deprecated")
warnings.filterwarnings("ignore", message="'pin_memory' argument is set as true.*MPS")
warnings.filterwarnings("ignore", message="some non-default generation parameters are set in the model config")

# ───────── REPRODUCIBILITY ─────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False

nltk.download("punkt")

# ───────── DEVICE ─────────
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# ───────── ROUGE ─────────
def compute_rouge(preds, refs):
    rouge = load("rouge")
    res = rouge.compute(predictions=preds, references=refs)
    return {k: float(v * 100) for k, v in res.items()}

# ───────── MAIN ─────────
def main():
    mlruns_path = os.path.abspath("mlruns")
    os.makedirs(mlruns_path, exist_ok=True)

    mlflow.set_tracking_uri(f"file://{mlruns_path}")
    mlflow.set_experiment("local-file")

    with mlflow.start_run(run_name="bart-base_final_benchmark_config"):
        model_name = "facebook/bart-base"
        epochs = 1
        batch_size = 2
        max_input_length = 384
        max_target_length = 64

        mlflow.log_params({
            "model_name_or_path": model_name,
            "epochs": epochs,
            "batch_size": batch_size,
            "max_input_length": max_input_length,
            "max_target_length": max_target_length,
            "use_causal_lm": False
        })

        data = load_dataset("cnn_dailymail", "3.0.0")
        data["train"] = data["train"].shuffle(seed=SEED).select(range(25000))
        data["validation"] = data["validation"].shuffle(seed=SEED).select(range(1250))
        data["test"] = data["test"].shuffle(seed=SEED).select(range(1250))

        tok = AutoTokenizer.from_pretrained(model_name)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token

        def preprocess(ex):
            inputs = tok(ex["article"], max_length=max_input_length, truncation=True)
            targets = tok(text_target=ex["highlights"], max_length=max_target_length, truncation=True)
            inputs["labels"] = targets["input_ids"]
            return inputs

        ds = data.map(preprocess, batched=True, remove_columns=data["train"].column_names)

        model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
        trainer = Seq2SeqTrainer(
            model=model,
            args=Seq2SeqTrainingArguments(
                output_dir="./checkpoints_final",
                evaluation_strategy="epoch",
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=batch_size,
                num_train_epochs=epochs,
                logging_steps=50,
                save_strategy="epoch",
                gradient_accumulation_steps=4,
                predict_with_generate=True,
                remove_unused_columns=True,
                report_to="none",
                seed=SEED
            ),
            train_dataset=ds["train"],
            eval_dataset=ds["validation"],
            data_collator=DataCollatorForSeq2Seq(tok, model=model),
            tokenizer=tok
        )

        print("\n🚀 Training model …")
        trainer.train()

        for log in trainer.state.log_history:
            step = log.get("step", 0)
            for k, v in log.items():
                if isinstance(v, (int, float)):
                    mlflow.log_metric(k, v, step=step)

        # ── Validation generation ──
        print("\n🔍 Running validation eval …")
        model.eval()
        val_preds, val_refs = [], []
        for ex in tqdm(data["validation"], desc="Generating validation summaries"):
            enc = tok(ex["article"], return_tensors="pt", truncation=True, max_length=max_input_length).to(device)
            with torch.no_grad():
                gen_ids = model.generate(
                    **enc,
                    max_new_tokens=max_target_length,
                    num_beams=4,
                    no_repeat_ngram_size=3,
                )
            val_preds.append(tok.decode(gen_ids[0], skip_special_tokens=True))
            val_refs.append(ex["highlights"])

        rouge = compute_rouge(val_preds, val_refs)
        mlflow.log_metrics({f"val_{k}": v for k, v in rouge.items()})
        mlflow.log_metric("val_avg_generated_len", float(np.mean([len(tok.tokenize(s)) for s in val_preds])))
        mlflow.log_metric("val_avg_reference_len", float(np.mean([len(tok.tokenize(s)) for s in val_refs])))
        print("📊 Validation ROUGE:", rouge)

        # ── Test generation ──
        print("\n🔍 Running test eval …")
        test_preds, test_refs = [], []
        for ex in tqdm(data["test"], desc="Generating test summaries"):
            enc = tok(ex["article"], return_tensors="pt", truncation=True, max_length=max_input_length).to(device)
            with torch.no_grad():
                gen_ids = model.generate(
                    **enc,
                    max_new_tokens=max_target_length,
                    num_beams=4,
                    no_repeat_ngram_size=3,
                )
            test_preds.append(tok.decode(gen_ids[0], skip_special_tokens=True))
            test_refs.append(ex["highlights"])

        test_rouge = compute_rouge(test_preds, test_refs)
        mlflow.log_metrics({f"test_{k}": v for k, v in test_rouge.items()})
        mlflow.log_metric("test_avg_generated_len", float(np.mean([len(tok.tokenize(s)) for s in test_preds])))
        mlflow.log_metric("test_avg_reference_len", float(np.mean([len(tok.tokenize(s)) for s in test_refs])))
        print("📊 Test ROUGE:", test_rouge)

        with tempfile.TemporaryDirectory() as tmpdir:
            for i in range(min(3, len(test_preds))):
                with open(os.path.join(tmpdir, f"example_test_{i}.txt"), "w") as f:
                    f.write(
                        f"Article:\n{data['test'][i]['article']}\n\n"
                        f"Reference:\n{data['test'][i]['highlights']}\n\n"
                        f"Prediction:\n{test_preds[i]}"
                    )
            mlflow.log_artifacts(tmpdir, artifact_path="examples")

        model.save_pretrained("deployment/model")
        tok.save_pretrained("deployment/model")

if __name__ == "__main__":
    main()