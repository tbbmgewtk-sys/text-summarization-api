import subprocess
import time

# List of (model_name, epochs, batch_size, input_len, target_len, use_causal_lm)
experiments = [
    # BART runs
    ("facebook/bart-base", 1, 2, 256, 64, False),
    ("facebook/bart-base", 2, 2, 256, 64, False),
    ("facebook/bart-base", 1, 4, 256, 64, False),
    ("facebook/bart-base", 1, 2, 384, 64, False),
    ("facebook/bart-base", 1, 2, 256, 128, False),

    # GPT2 runs
    ("gpt2", 1, 2, 256, 64, True),
    ("gpt2", 2, 2, 256, 64, True),
    ("gpt2", 1, 4, 256, 64, True),
    ("gpt2", 1, 2, 384, 64, True),
    ("gpt2", 1, 2, 256, 128, True),
]

for idx, (model, ep, bs, in_len, out_len, causal) in enumerate(experiments, 1):
    print(f"\n🧪 Running experiment {idx}/10: {model} | {ep}ep | {bs}bs | in:{in_len} out:{out_len} | causal: {causal}")
    
    cmd = [
        "python", "src/fine_tune.py",
        "--model_name_or_path", model,
        "--epochs", str(ep),
        "--batch_size", str(bs),
        "--max_input_length", str(in_len),
        "--max_target_length", str(out_len),
    ]
    
    if causal:
        cmd.append("--use_causal_lm")

    # Launch the process
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Run {idx} failed: {e}")
        continue

    # Optional pause between runs
    print("✅ Finished run", idx)
    time.sleep(5)