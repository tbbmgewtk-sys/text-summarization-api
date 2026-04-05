import os
import json
from datetime import datetime
import matplotlib.pyplot as plt

# ───────── Load Logs ─────────
log_path = os.path.join(os.path.dirname(__file__), "drift_logs.log")

# Containers
timestamps_input, input_lengths, input_entropies = [], [], []
timestamps_output, output_lengths, running_avgs = [], [], []
embedding_timestamps, embedding_scores = [], []

# Drift markers
entropy_drift_points, output_drift_points, embedding_drift_points = [], [], []

with open(log_path, "r") as f:
    for line in f:
        try:
            log = json.loads(line)
            ts = datetime.fromisoformat(log["timestamp"])

            if log["event"] == "INPUT_STATS":
                timestamps_input.append(ts)
                input_lengths.append(log["input_token_count"])
                input_entropies.append(log["input_entropy"])

            elif log["event"] == "LENGTH_OK":
                timestamps_output.append(ts)
                output_lengths.append(log["current_output_length"])
                running_avgs.append(log["running_avg"])

            elif log["event"] == "EMBEDDING_OK" or log["event"] == "EMBEDDING_BASELINE":
                embedding_timestamps.append(ts)
                embedding_scores.append(log["cosine_distance"])

            elif log["event"] == "DRIFT_ALERT":
                if log.get("type") == "length":
                    timestamps_output.append(ts)
                    output_lengths.append(log["current_output_length"])
                    running_avgs.append(log["running_avg"])
                    output_drift_points.append((ts, log["running_avg"]))

                elif log.get("type") == "entropy":
                    timestamps_input.append(ts)
                    input_lengths.append(log["input_token_count"])
                    input_entropies.append(log["input_entropy"])
                    entropy_drift_points.append((ts, log["input_entropy"]))

                elif log.get("type") == "embedding":
                    embedding_timestamps.append(ts)
                    embedding_scores.append(log["cosine_distance"])
                    embedding_drift_points.append((ts, log["cosine_distance"]))

        except Exception as e:
            print(f"⚠️ Skipping malformed line: {e}")

# ───────── PLOTTING ─────────
fig, axes = plt.subplots(4, 1, figsize=(12, 14), sharex=True)

# ── Input Length ──
axes[0].plot(timestamps_input, input_lengths, label="Input Length", marker="o", color="steelblue")
axes[0].set_ylabel("Token Count")
axes[0].set_title("Input Length Over Time")
axes[0].grid(True)
axes[0].legend()

# ── Input Entropy ──
axes[1].plot(timestamps_input, input_entropies, label="Input Entropy", marker="x", color="darkorange")
if entropy_drift_points:
    ts_drift, ent_drift = zip(*entropy_drift_points)
    axes[1].scatter(ts_drift, ent_drift, color="red", label="Entropy Drift ⚠", zorder=5)

axes[1].set_ylabel("Entropy")
axes[1].set_title("Input Entropy Over Time")
axes[1].grid(True)
axes[1].legend()

# ── Output Summary Length ──
axes[2].plot(timestamps_output, output_lengths, label="Summary Length", marker="o")
axes[2].plot(timestamps_output, running_avgs, label="Running Avg", linestyle="--")
axes[2].axhline(56, color="gray", linestyle=":", label="Expected Avg (56)")
if output_drift_points:
    ts_drift, avg_drift = zip(*output_drift_points)
    axes[2].scatter(ts_drift, avg_drift, color="red", label="Output Drift ⚠", zorder=5)

axes[2].set_ylabel("Token Length")
axes[2].set_title("Output Drift Monitoring")
axes[2].grid(True)
axes[2].legend()

# ── Embedding Drift ──
axes[3].plot(embedding_timestamps, embedding_scores, label="Embedding Drift Score", color="purple", marker="^")
if embedding_drift_points:
    ts_drift, dist_drift = zip(*embedding_drift_points)
    axes[3].scatter(ts_drift, dist_drift, color="red", label="Embedding Drift ⚠", zorder=5)

axes[3].set_ylabel("Cosine Distance")
axes[3].set_title("Embedding Drift Over Time")
axes[3].grid(True)
axes[3].legend()

# ── Final Layout ──
plt.tight_layout()
plt.suptitle("Drift Monitoring Summary", y=1.02, fontsize=16)
plt.subplots_adjust(top=0.95)
plt.show()