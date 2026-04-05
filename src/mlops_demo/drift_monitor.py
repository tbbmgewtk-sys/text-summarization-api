import numpy as np
import logging
import os
import math
import json
from datetime import datetime
from collections import deque
import torch
import textstat
from sklearn.metrics.pairwise import cosine_similarity

# ───────── Constants ─────────
MAX_WINDOW_SIZE = 100
OUTPUT_LEN_THRESHOLD = 10
REFERENCE_MEAN = 56

ENTROPY_MIN = 6.0
ENTROPY_MAX = 9.0

EMBEDDING_DRIFT_THRESHOLD = 0.35  # Cosine distance threshold

# ───────── Globals ─────────
output_lengths = deque(maxlen=MAX_WINDOW_SIZE)
reference_embedding = None

log_path = os.path.join(os.path.dirname(__file__), "drift_logs.log")
alert_path = os.path.join(os.path.dirname(__file__), "alerts.json")

logging.basicConfig(filename=log_path, level=logging.INFO, format="%(message)s")

# ───────── Helper: Save to alerts.json ─────────
def log_alert(alert):
    for key, value in alert.items():
        if isinstance(value, (np.generic, torch.Tensor)):
            alert[key] = float(value)
    with open(alert_path, "a") as f:
        f.write(json.dumps(alert) + "\n")
    print("⚠️  [ALERT]", alert)

# ───────── 1. Output Length Drift ─────────
def monitor_output_length(token_length):
    output_lengths.append(token_length)
    avg = float(np.mean(output_lengths))
    timestamp = datetime.utcnow().isoformat()

    if abs(avg - REFERENCE_MEAN) > OUTPUT_LEN_THRESHOLD:
        alert = {
            "timestamp": timestamp,
            "event": "DRIFT_ALERT",
            "type": "length",
            "current_output_length": int(token_length),
            "running_avg": round(avg, 2),
            "reference_mean": REFERENCE_MEAN
        }
        logging.warning(json.dumps(alert))
        log_alert(alert)
    else:
        logging.info(json.dumps({
            "timestamp": timestamp,
            "event": "LENGTH_OK",
            "current_output_length": int(token_length),
            "running_avg": round(avg, 2)
        }))

# ───────── 2. Entropy + Readability Drift ─────────
def monitor_input_stats(input_text, tokenizer):
    tokens = tokenizer.tokenize(input_text)
    length = len(tokens)
    token_freq = {t: tokens.count(t) for t in set(tokens)}
    probs = [freq / length for freq in token_freq.values()]
    entropy = -sum(p * math.log(p + 1e-10, 2) for p in probs)
    readability = textstat.flesch_reading_ease(input_text)

    timestamp = datetime.utcnow().isoformat()
    log_entry = {
        "timestamp": timestamp,
        "event": "INPUT_STATS",
        "input_token_count": int(length),
        "input_entropy": round(entropy, 4),
        "input_readability": round(readability, 2)
    }
    logging.info(json.dumps(log_entry))

    if entropy < ENTROPY_MIN or entropy > ENTROPY_MAX:
        alert = {
            "timestamp": timestamp,
            "event": "DRIFT_ALERT",
            "type": "entropy",
            "input_token_count": int(length),
            "input_entropy": round(entropy, 4),
            "bounds": [ENTROPY_MIN, ENTROPY_MAX]
        }
        logging.warning(json.dumps(alert))
        log_alert(alert)

# ───────── 3. Embedding Drift Detection ─────────
def monitor_input_embedding(input_tensor, model):
    global reference_embedding
    timestamp = datetime.utcnow().isoformat()

    with torch.no_grad():
        encoder_output = model.model.encoder(input_tensor['input_ids']).last_hidden_state
        embedding = encoder_output.mean(dim=1).cpu().numpy()

    if reference_embedding is None:
        reference_embedding = embedding
        logging.info(json.dumps({
            "timestamp": timestamp,
            "event": "EMBEDDING_BASELINE",
            "cosine_distance": 0.0
        }))
        return

    cos_sim = float(cosine_similarity(reference_embedding, embedding)[0][0])
    cos_dist = round(1 - cos_sim, 4)

    log_entry = {
        "timestamp": timestamp,
        "event": "EMBEDDING_OK",
        "cosine_distance": cos_dist
    }
    logging.info(json.dumps(log_entry))

    if cos_dist > EMBEDDING_DRIFT_THRESHOLD:
        alert = {
            "timestamp": timestamp,
            "event": "DRIFT_ALERT",
            "type": "embedding",
            "cosine_distance": cos_dist,
            "threshold": EMBEDDING_DRIFT_THRESHOLD
        }
        logging.warning(json.dumps(alert))
        log_alert(alert)