import os
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
import argparse, os, warnings, torch, mlflow, numpy as np, random, shutil, tempfile
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    DataCollatorForSeq2Seq,
    DataCollatorWithPadding,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    Trainer
)
from evaluate import load
import nltk

# ───────────────── Warning filters ─────────────────
warnings.filterwarnings("ignore", message="Redirects are currently not supported in Windows or MacOs")
warnings.filterwarnings("ignore", message="`resume_download` is deprecated")
warnings.filterwarnings("ignore", message="'pin_memory' argument is set as true.*MPS")
warnings.filterwarnings("ignore", message="some non-default generation parameters are set in the model config")
# ───────────────────────────────────────────────────

# ───────── REPRODUCIBILITY ─────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False
# ──────────────────────────────────────

# one-time punkt download for ROUGE
nltk.download("punkt")

# ───────── DEVICE ─────────
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# ───────── CLI ─────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name_or_path", type=str, default="facebook/bart-base")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--max_input_length", type=int, default=256)
    p.add_argument("--max_target_length", type=int, default=64)
    p.add_argument("--use_causal_lm", action="store_true")
    return p.parse_args()

# ───────── small helper: clean orphaned folders ─────────
def _clean_orphan_experiments(root: str) -> None:
    """Remove any first-level folder that isn’t an experiment (digits) and lacks meta.yaml."""
    for name in os.listdir(root):
        full = os.path.join(root, name)
        if not os.path.isdir(full):
            continue
        if name.isdigit():
            continue  # proper experiment folder
        if not os.path.exists(os.path.join(full, "meta.yaml")):
            try:
                shutil.rmtree(full)
                print(f"[cleanup] removed orphaned dir {full}")
            except Exception as e:
                print(f"[cleanup] could not remove {full}: {e}")

# ───────── ROUGE ─────────
def compute_rouge(preds, refs):
    rouge = load("rouge")
    res = rouge.compute(predictions=preds, references=refs)
    return {k: float(v * 100) for k, v in res.items()}

# ───────── MAIN ─────────
def main():
    # 0) garbage-collect half-written runs (optional)
    mlruns_path = os.path.abspath("mlruns")
    os.makedirs(mlruns_path, exist_ok=True)
    _clean_orphan_experiments(mlruns_path)

    # 1) MLflow initialisation
    mlflow.set_tracking_uri(f"file://{mlruns_path}")
    exp_name = "local-file"
    mlflow.set_experiment(exp_name)

    args = parse_args()
    run_name = (
        f"{args.model_name_or_path.split('/')[-1]}_"
        f"{args.epochs}ep_{args.batch_size}bs_"
        f"{args.max_input_length}in_{args.max_target_length}out"
    )

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(vars(args))

        # quick subset (for dry runs; bump these for real training!)
        data = load_dataset("cnn_dailymail", "3.0.0")
        data["train"] = data["train"].shuffle(seed=42).select(range(5000))
        data["validation"] = data["validation"].shuffle(seed=42).select(range(250))

        tok = AutoTokenizer.from_pretrained(args.model_name_or_path)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token  # for GPT-2 & friends

        max_len = args.max_input_length

        # ---------- preprocessing ----------
        def preprocess(ex):
            art = tok(
                ex["article"],
                max_length=args.max_input_length,
                truncation=True,
                padding=False,
            )

            if args.use_causal_lm:
                summ = tok(
                    ex["highlights"],
                    max_length=args.max_target_length,
                    truncation=True,
                    padding=False,
                )
                inp_ids = [a + s for a, s in zip(art["input_ids"], summ["input_ids"])]
                inp_ids = [seq[:max_len] for seq in inp_ids]

                pad = tok.pad_token_id
                padded_ids   = [seq + [pad] * (max_len - len(seq)) for seq in inp_ids]
                padded_mask  = [[1] * len(seq) + [0] * (max_len - len(seq)) for seq in inp_ids]
                padded_label = [seq + [-100] * (max_len - len(seq)) for seq in inp_ids]

                return {
                    "input_ids":      padded_ids,
                    "attention_mask": padded_mask,
                    "labels":         padded_label,
                }
            else:
                summ = tok(
                    text_target=ex["highlights"],
                    max_length=args.max_target_length,
                    truncation=True,
                    padding=False,
                )
                art["labels"] = summ["input_ids"]
                return art

        ds = data.map(
            preprocess,
            batched=True,
            remove_columns=data["train"].column_names,
        )

        # ───────── model / trainer ─────────
        if args.use_causal_lm:
            model       = AutoModelForCausalLM.from_pretrained(args.model_name_or_path).to(device)
            trainer_cls = Trainer
            collator    = DataCollatorWithPadding(tok, return_tensors="pt")
        else:
            model       = AutoModelForSeq2SeqLM.from_pretrained(args.model_name_or_path).to(device)
            trainer_cls = Seq2SeqTrainer
            collator    = DataCollatorForSeq2Seq(tok, model=model)

        targs = Seq2SeqTrainingArguments(
            output_dir="./checkpoints",
            evaluation_strategy="epoch",
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            num_train_epochs=args.epochs,
            logging_steps=50,
            save_strategy="epoch",
            gradient_accumulation_steps=4,
            predict_with_generate=not args.use_causal_lm,
            remove_unused_columns=True,
            report_to="none",
            seed=SEED,                     # ← ensure Trainer uses our seed
        )

        trainer = trainer_cls(
            model=model,
            args=targs,
            train_dataset=ds["train"],
            eval_dataset=ds["validation"],
            data_collator=collator,
            tokenizer=tok,
        )

        print("🚀 Training …")
        trainer.train()

        # ── manual metric logging ─────────
        wanted = {
            "loss":                     "train_loss",
            "eval_loss":                "eval_loss",
            "grad_norm":                "grad_norm",
            "train_runtime":            "train_runtime",
            "eval_runtime":             "eval_runtime",
            "train_steps_per_second":   "train_steps_per_second",
            "train_samples_per_second": "train_samples_per_second",
            "eval_steps_per_second":    "eval_steps_per_second",
            "eval_samples_per_second":  "eval_samples_per_second",
        }
        for log in trainer.state.log_history:
            step = log.get("step", 0)
            for k, mlname in wanted.items():
                if k in log:
                    mlflow.log_metric(mlname, float(log[k]), step=step)

        # ── evaluation & ROUGE ─────────
        print("🔍 Evaluating …")
        if args.use_causal_lm:
            preds_text, refs_text = [], []
            model.eval()
            for ex in data["validation"]:
                enc = tok(
                    ex["article"],
                    return_tensors="pt",
                    truncation=True,
                    max_length=args.max_input_length,
                ).to(device)
                with torch.no_grad():
                    gen_ids = model.generate(
                        **enc,
                        max_new_tokens=args.max_target_length,
                        num_beams=4,
                        no_repeat_ngram_size=3,
                    )
                preds_text.append(tok.decode(gen_ids[0], skip_special_tokens=True))
                refs_text.append(ex["highlights"])
        else:
            out = trainer.predict(ds["validation"], max_new_tokens=args.max_target_length)
            vsize = tok.vocab_size
            preds = np.where(
                (out.predictions >= 0) & (out.predictions < vsize),
                out.predictions,
                tok.pad_token_id,
            ).astype(np.int32)
            refs = np.where(
                out.label_ids != -100, out.label_ids, tok.pad_token_id
            ).astype(np.int32)
            preds_text = tok.batch_decode(preds, skip_special_tokens=True)
            refs_text  = tok.batch_decode(refs,  skip_special_tokens=True)

        rouge = compute_rouge(preds_text, refs_text)
        print("📊 ROUGE:", rouge)
        mlflow.log_metrics(rouge)

        # ── lengths & examples ─────────
        g_len = [len(tok.tokenize(s)) for s in preds_text]
        r_len = [len(tok.tokenize(s)) for s in refs_text]
        mlflow.log_metric("avg_generated_len", float(sum(g_len) / len(g_len)))
        mlflow.log_metric("avg_reference_len", float(sum(r_len) / len(r_len)))

        with tempfile.TemporaryDirectory() as tmpdir:
            for i in range(min(3, len(preds_text))):
                fp = os.path.join(tmpdir, f"example_{i}.txt")
                with open(fp, "w") as f:
                    f.write(
                        f"Article:\n{data['validation'][i]['article']}\n\n"
                        f"Reference:\n{data['validation'][i]['highlights']}\n\n"
                        f"Prediction:\n{preds_text[i]}"
                    )
            mlflow.log_artifacts(tmpdir, artifact_path="examples")


if __name__ == "__main__":
    main()