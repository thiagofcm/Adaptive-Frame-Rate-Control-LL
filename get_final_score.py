import pandas as pd
import argparse

W_SUCCESS = 0.30
W_VY     = 0.30
W_FRAMES = 0.40

BOUNDS = {
    "success_pct_mean": (0.0, 100.0),
    "vy_mean":          (0.0, 2.0),
    "frames_mean":      (0.0, 250.0),
}

def normalize_fixed(value, min_val, max_val):
    return (value - min_val) / (max_val - min_val)

def compute_scores(csv_path, output_path=None):
    df = pd.read_csv(csv_path)

    S_norm  = normalize_fixed(df["success_pct_mean"], *BOUNDS["success_pct_mean"])
    Vy_norm = normalize_fixed(df["vy_mean"],          *BOUNDS["vy_mean"])
    F_norm  = normalize_fixed(df["frames_mean"],      *BOUNDS["frames_mean"])

    df["S_norm"]  = S_norm
    df["Vy_norm"] = Vy_norm
    df["F_norm"]  = F_norm

    df["score"] = (
        W_SUCCESS * S_norm +
        W_VY      * (1 - Vy_norm) +
        W_FRAMES  * (1 - F_norm)
    )

    df = df.sort_values("score", ascending=False).reset_index(drop=True)

    cols = ["model", "budget", "success_pct_mean", "vy_mean", "frames_mean",
            "S_norm", "Vy_norm", "F_norm", "score"]
    result = df[cols]

    print(result.to_string(index=True, float_format=lambda x: f"{x:.4f}"))

    if output_path:
        result.to_csv(output_path, index=False)
        print(f"\nSaved to {output_path}")

    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute model scores from evaluation CSV.")
    parser.add_argument("--csv", help="Path to the input CSV file")
    parser.add_argument("--output", "-o", default=None, help="Optional path to save scored CSV")
    args = parser.parse_args()

    compute_scores(args.csv, args.output)