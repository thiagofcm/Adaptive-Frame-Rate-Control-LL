"""
compute_statistics.py
──────────────────────
Reproduces the sensing-efficiency statistics pipeline from the F1TENTH AdaptiveFPS
evaluator (AdaptiveFPS/utils/compute_statistics.py) for Lunar Lander: same formulas,
same rounding/NaN conventions, same aggregation/sort logic. Unlike the F1TENTH script
(which reads its per-episode step/fresh-observation counts straight from episodes.csv),
this version recomputes steps_length/n_fresh_observations per episode from that
episode's own episode_XX/steps.csv fresh_observation column -- the per-step source of
truth -- while still reusing the existing `success` value from episodes.csv unchanged.

Performs a full rebuild every run, scanning whatever policy folders currently exist
under eval/lunar_lander/:

    python scripts/compute_statistics.py

Writes:
    eval/lunar_lander/<policy>/statistics.csv   (one row per policy)
    eval/lunar_lander/statistics_summary.csv    (one row per policy, aggregated)
"""

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

EVAL_ROOT = Path("eval/lunar_lander")


def parse_success_column(series: pd.Series) -> pd.Series:
    """Robustly convert a success column to bool, handling True/False,
    "true"/"false" (any case), and 1/0 (numeric or string). Copied verbatim from
    AdaptiveFPS/utils/compute_statistics.py."""
    if series.dtype == bool:
        return series
    return series.astype(str).str.strip().str.lower().map(
        {"true": True, "false": False, "1": True, "0": False}
    )


def parse_bool_column(series: pd.Series) -> pd.Series:
    """Same robust True/False parsing as parse_success_column, applied to
    steps.csv's fresh_observation column."""
    if series.dtype == bool:
        return series
    return series.astype(str).str.strip().str.lower().eq("true")


def compute_statistics(policy_dir: Path):
    episodes_csv = policy_dir / "episodes.csv"
    if not episodes_csv.exists():
        return None

    df = pd.read_csv(episodes_csv)
    if df.empty:
        print(f"Warning: {episodes_csv} has no episode rows; skipping.", file=sys.stderr)
        return None

    # steps_length / n_fresh_observations recomputed per episode from that episode's
    # own steps.csv fresh_observation column -- the per-step source of truth -- rather
    # than trusting the pre-aggregated episodes.csv columns of the same name.
    steps_length_values = []
    n_fresh_values = []
    for episode_index in df["episode_index"]:
        steps_csv = policy_dir / f"episode_{int(episode_index):02d}" / "steps.csv"
        steps_df = pd.read_csv(steps_csv)
        steps_length_values.append(len(steps_df))
        n_fresh_values.append(int(parse_bool_column(steps_df["fresh_observation"]).sum()))

    steps_length = pd.Series(steps_length_values, index=df.index, dtype=float)
    n_fresh = pd.Series(n_fresh_values, index=df.index, dtype=float)

    # ── 1. Measurement : No-Measurement ratio ──────────────────────
    # Computed per episode so we can report the distribution (mean/std), not just
    # an aggregate ratio. Episodes with zero fresh observations would divide by
    # zero, so those are excluded via NaN + nanmean/nanstd.
    no_measurements = steps_length - n_fresh
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = no_measurements / n_fresh
    ratio = ratio.where(n_fresh != 0, np.nan)
    mean_ratio = np.nanmean(ratio)
    std_ratio = np.nanstd(ratio)

    # ── 2. Observation reduction (%) ────────────────────────────────
    # Per-episode percentage of steps without a fresh observation, then aggregated
    # -- deliberately not sum(n_fresh)/sum(steps_length), since that would hide the
    # per-episode distribution.
    with np.errstate(divide="ignore", invalid="ignore"):
        observation_reduction = (1 - n_fresh / steps_length) * 100
    observation_reduction = observation_reduction.where(steps_length != 0, np.nan)
    mean_reduction = np.nanmean(observation_reduction)
    std_reduction = np.nanstd(observation_reduction)

    # ── 3. Fresh observations | success ─────────────────────────────
    success = parse_success_column(df["success"])
    successful_fresh = n_fresh[success]
    n_success = len(successful_fresh)
    if n_success > 0:
        mean_fresh_success = successful_fresh.mean()
        std_fresh_success = successful_fresh.std()
    else:
        mean_fresh_success = float("nan")
        std_fresh_success = float("nan")

    # ── 4. Success rate (%) ──────────────────────────────────────────
    # Independent of the "given success" stats above: this stays a real percentage
    # (0.0, not NaN) even when there are zero successes, since 0% is a meaningful,
    # known result -- unlike the fresh-observations stats, which are genuinely
    # undefined with no successful episodes.
    success_rate_percent = (n_success / len(df)) * 100.0

    out_path = policy_dir / "statistics.csv"
    summary = pd.DataFrame([{
        "mean_measurement_no_measurement_ratio": round(mean_ratio, 4),
        "std_measurement_no_measurement_ratio": round(std_ratio, 4),
        "mean_observation_reduction_percent": round(mean_reduction, 4),
        "std_observation_reduction_percent": round(std_reduction, 4),
        "mean_fresh_observations_given_success": round(mean_fresh_success, 4),
        "std_fresh_observations_given_success": round(std_fresh_success, 4),
        "success_rate_percent": round(success_rate_percent, 4),
    }])
    # na_rep="NaN" so a fully-empty-success policy writes literal "NaN", not a blank cell
    summary.to_csv(out_path, index=False, na_rep="NaN")

    print(
        f"  {policy_dir.name}: {len(df)} episodes, {n_success} successful "
        f"({success_rate_percent:.2f}%) -> {out_path}"
    )
    return out_path


def policy_sort_key(policy: str):
    """Fixed policies (ascending Hz) sort before adaptive policies (ascending frame
    cost, then descending budget); unparsable names sort last, alphabetically.
    Copied verbatim from AdaptiveFPS/utils/compute_statistics.py."""
    fixed_match = re.match(r"^fixed_(\d+(?:\.\d+)?)Hz$", policy)
    if fixed_match:
        return (0, float(fixed_match.group(1)), 0.0, policy)

    adaptive_match = re.match(r"^adaptive_fc_([\d.]+)_bud_([\d.]+)", policy)
    if adaptive_match:
        frame_cost = float(adaptive_match.group(1))
        budget = float(adaptive_match.group(2))
        return (1, frame_cost, -budget, policy)

    return (2, 0.0, 0.0, policy)


def aggregate_statistics(root_path: Path):
    stats_files = sorted(root_path.glob("*/statistics.csv"))
    if not stats_files:
        print("No statistics.csv files found; nothing to aggregate.")
        return

    rows = []
    for stats_file in stats_files:
        try:
            df = pd.read_csv(stats_file)
        except pd.errors.EmptyDataError:
            print(f"Warning: {stats_file} is empty, skipping")
            continue
        if df.empty:
            print(f"Warning: {stats_file} has no rows, skipping")
            continue
        df.insert(0, "policy", stats_file.parent.name)
        rows.append(df)

    if not rows:
        print("No usable statistics.csv files found; nothing to aggregate.")
        return

    combined = pd.concat(rows, ignore_index=True, sort=False)
    combined["_sort_key"] = combined["policy"].map(policy_sort_key)
    combined = (
        combined.sort_values("_sort_key", kind="stable")
        .drop(columns="_sort_key")
        .reset_index(drop=True)
    )

    out_path = root_path / "statistics_summary.csv"
    combined.to_csv(out_path, index=False, na_rep="NaN")

    print(f"\nAggregated {len(rows)} policy evaluations")
    print(f"Saved aggregated statistics to: {out_path}")


def main():
    if not EVAL_ROOT.exists():
        print(f"Error: {EVAL_ROOT} does not exist.", file=sys.stderr)
        sys.exit(1)

    policy_dirs = sorted(p for p in EVAL_ROOT.iterdir() if p.is_dir())
    print(f"Found {len(policy_dirs)} policy folder(s) under {EVAL_ROOT}")

    n_written = 0
    for policy_dir in policy_dirs:
        if compute_statistics(policy_dir) is not None:
            n_written += 1

    print(f"\nComputed statistics.csv for {n_written}/{len(policy_dirs)} policy folder(s).")

    aggregate_statistics(EVAL_ROOT)


if __name__ == "__main__":
    main()
