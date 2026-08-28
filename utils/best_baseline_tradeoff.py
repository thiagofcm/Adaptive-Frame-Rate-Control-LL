import pandas as pd

#CSV_FILE = "eval/heuristic/summary.csv"
CSV_FILE = "eval/vy_exceed_0.5/heuristic/summary.csv"
OUTPUT_FILE = "top_10_tradeoff.csv"

df = pd.read_csv(CSV_FILE)

# Change these names if needed
SUCCESS_COL = "success_rate"
FRAME_COL = "mean_n_fresh_observations"

# Normalize to [0, 1]
success_norm = (
    (df[SUCCESS_COL] - df[SUCCESS_COL].min())
    / (df[SUCCESS_COL].max() - df[SUCCESS_COL].min())
)

frame_norm = (
    (df[FRAME_COL] - df[FRAME_COL].min())
    / (df[FRAME_COL].max() - df[FRAME_COL].min())
)

# Weight between success and sensing efficiency
alpha = 0.5

df["tradeoff_score"] = (
    alpha * success_norm
    + (1 - alpha) * (1 - frame_norm)
)

top_10 = df.nlargest(10, "tradeoff_score")

print("\nTop 10 success/frame-consumption trade-offs:")
print(top_10.to_string(index=False))

top_10.to_csv(OUTPUT_FILE, index=False)

print(f"\nSaved to: {OUTPUT_FILE}")