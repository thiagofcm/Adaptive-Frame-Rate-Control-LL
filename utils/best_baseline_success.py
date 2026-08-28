import pandas as pd

#CSV_FILE = "eval/heuristic/summary.csv"
CSV_FILE = "eval/vy_exceed_0.5/heuristic/summary.csv"
OUTPUT_FILE = "top_10_success_rate.csv"

df = pd.read_csv(CSV_FILE)

top_10 = df.nlargest(10, "success_rate")

# Print results
print("\nTop 10 results by success rate:")
print(top_10.to_string(index=False))

# Save results
top_10.to_csv(OUTPUT_FILE, index=False)

print(f"\nSaved to: {OUTPUT_FILE}")