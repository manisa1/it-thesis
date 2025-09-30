# analyze_results.py
import os, pandas as pd

runs = {
    "static_base": "runs/static_base/metrics.csv",
    "static_sol":  "runs/static_sol/metrics.csv",
    "dyn_base":    "runs/dyn_base/metrics.csv",
    "dyn_sol":     "runs/dyn_sol/metrics.csv",
    "burst_base":  "runs/burst_base/metrics.csv",
    "burst_sol":   "runs/burst_sol/metrics.csv",
    "shift_base":  "runs/shift_base/metrics.csv",
    "shift_sol":   "runs/shift_sol/metrics.csv",
}

rows = []
for name, path in runs.items():
    if not os.path.exists(path):
        print(f"Missing: {path}")
        continue
    df = pd.read_csv(path)
    r = float(df["Recall@K"].iloc[0])
    n = float(df["NDCG@K"].iloc[0])
    rows.append({"run": name, "Recall@20": r, "NDCG@20": n})

summary = pd.DataFrame(rows).set_index("run").loc[
    ["static_base", "static_sol", "dyn_base", "dyn_sol", "burst_base", "burst_sol", "shift_base", "shift_sol"]
]

# Robustness Drop = (clean - noisy) / clean
def robust_drop(clean, noisy):
    return (clean - noisy) / clean if clean > 0 else 0.0

clean_r = summary.loc["static_base", "Recall@20"]
noisy_r_base = summary.loc["dyn_base", "Recall@20"]
noisy_r_sol  = summary.loc["dyn_sol",  "Recall@20"]

clean_n = summary.loc["static_base", "NDCG@20"]
noisy_n_base = summary.loc["dyn_base", "NDCG@20"]
noisy_n_sol  = summary.loc["dyn_sol",  "NDCG@20"]

robust = pd.DataFrame({
    "Metric": ["Recall@20", "NDCG@20"],
    "Baseline Robustness Drop": [
        robust_drop(clean_r, noisy_r_base),
        robust_drop(clean_n, noisy_n_base),
    ],
    "Solution Robustness Drop": [
        robust_drop(clean_r, noisy_r_sol),
        robust_drop(clean_n, noisy_n_sol),
    ],
})

os.makedirs("runs", exist_ok=True)
summary_out = "runs/summary.csv"
robust_out  = "runs/robustness.csv"
summary.to_csv(summary_out)
robust.to_csv(robust_out, index=False)

print("\n=== Summary (per run) ===")
print(summary.round(4))
print(f"\nSaved {summary_out}")

print("\n=== Robustness Drop ((clean - noisy)/clean) ===")
print(robust.round(4))
print(f"Saved {robust_out}")
