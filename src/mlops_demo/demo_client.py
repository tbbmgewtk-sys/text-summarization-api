import requests
from datasets import load_dataset
import time

# Constants
REAL_SAMPLES = 5    # How many real CNN/DailyMail articles to send
DRIFT_SAMPLES = 5   # How many synthetic drift cases to send (fixed number in test_cases)

# FastAPI endpoint
API_URL = "http://127.0.0.1:8000/summarize"

# ───────── Real CNN/DailyMail Samples ─────────
print("\n📰 Running on real dataset examples (CNN/DailyMail)…\n")

dataset = load_dataset("cnn_dailymail", "3.0.0", split=f"test[:{REAL_SAMPLES}]")

for i, ex in enumerate(dataset):
    text = ex["article"]
    payload = {"text": text}

    try:
        start_time = time.time()
        response = requests.post(API_URL, json=payload)
        latency = time.time() - start_time
        result = response.json()

        print(f"\n✅ Article {i + 1}")
        print(f"→ Tokens: {result['tokens']} | Time: {latency:.2f}s")
        print(f"→ Summary: {result['summary'][:150]}{'...' if len(result['summary']) > 150 else ''}")

    except Exception as e:
        print(f"\n❌ Error on article {i + 1}: {e}")

    time.sleep(1)

# ───────── Drift Test Cases ─────────
print("\n🔍 Running on synthetic drift test cases…\n")

test_cases = {
    "Low Entropy Input": "apple " * 100,
    "Short Input": "Breaking: Fire in downtown LA.",
    "Long Input": " ".join(["This is a sentence about the economy."] * 100),
    "High Entropy Input": (
        "Zygomorphic paradoxes energize xylophonic transductions amidst "
        "cryptobiotic flux oscillators in hyperstring nodules."
    ),
    "Normal Article (for control)": (
        "The United Nations Secretary-General urged member states to increase funding for climate change adaptation. "
        "The announcement came during a summit attended by over 50 world leaders."
    )
}

for i, (label, text) in enumerate(test_cases.items()):
    if i >= DRIFT_SAMPLES:
        break

    print(f"\n🧪 Test Case: {label}")
    payload = {"text": text}

    try:
        start_time = time.time()
        response = requests.post(API_URL, json=payload)
        latency = time.time() - start_time
        result = response.json()

        print(f"→ Tokens: {result['tokens']} | Time: {latency:.2f}s")
        print(f"→ Summary: {result['summary'][:150]}{'...' if len(result['summary']) > 150 else ''}")

    except Exception as e:
        print(f"❌ Error on test case '{label}': {e}")

    time.sleep(1)