"""
analyze_lunar_lander.py
────────────────────────
Analysis/plotting for scripts/evaluate_lunar_lander_fps.py results, adapted from
AdaptiveFPS/scripts/analyze_adaptive_fps.py. Never touches the simulator (no
gym/envs/torch imports) -- purely a standalone reader of whatever episodes.csv /
lap_<i>/steps.csv / lap_<i>/trajectory.npy files are already on disk.

Lunar Lander has no fixed racetrack to draw as a background: terrain is randomly
regenerated per episode/seed, so unlike F1TENTH there is no single consistent map
valid across many overlaid episodes (and reconstructing it here would mean importing
the simulator, which this script deliberately avoids). Instead, every plotted x/y
axes gets the fixed, seed-independent reference geometry every trajectory shares:
the ground line, the landing-flag x-bounds used by the success criterion
(-0.2 < x < 0.2), and the out-of-bounds playfield edges (|x| >= 1.0).

Directory-driven rather than model+interval-driven: point --eval-dir at one policy's
evaluation folder directly, e.g.

    eval/adaptive_fc_2.0_bud_50.0
    eval/fixed_10Hz

Figures are saved as PNGs directly inside --eval-dir (matching this project's own
utils/plot_frame_acquisition.py convention), not under a separate paper_media/ tree.

Usage examples (run from the repo root):

    # trajectory colored by speed + by sensing rate, plus a vertical-velocity profile
    python utils/analyze_lunar_lander.py \\
        --eval-dir eval/adaptive_fc_2.0_bud_50.0 --npy-episode 3

    # overlay every episode, successes vs failures, touchdown/crash locations marked
    python utils/analyze_lunar_lander.py \\
        --eval-dir eval/adaptive_fc_2.0_bud_50.0 --overlay

    # altitude / instantaneous nav reward / cumulative nav reward / FPS over time
    python utils/analyze_lunar_lander.py \\
        --eval-dir eval/adaptive_fc_2.0_bud_50.0 --timeseries --episodes 3 11

    # same reader, pointed at a fixed-FPS baseline instead
    python utils/analyze_lunar_lander.py \\
        --eval-dir eval/fixed_10Hz --overlay
"""

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.colors import BoundaryNorm

FPS_CHOICES = [1, 5, 10, 25, 50]  # LunarLander_VarFramerate's fps_choices

# Fixed, seed-independent reference geometry (see module docstring). Sourced from the
# evaluator's own success/termination logic, not invented:
#   PAD_HALF_WIDTH  -- scripts/evaluate_lunar_lander_fps.py's landed_in_flags bound
#   PLAYFIELD_HALF_WIDTH -- envs/lunar_lander_var_fps.py's out-of-bounds/crash condition
PAD_HALF_WIDTH = 0.2
PLAYFIELD_HALF_WIDTH = 1.0

# Maximum number of overlaid episodes for which per-episode endpoint labels stay
# readable; beyond this they cluster together and become illegible clutter.
MAX_ANNOTATED_ENDPOINTS = 8

# Shared publication-style rcParams, matching this project's own
# utils/plot_frame_acquisition.py (itself matching the AdaptiveFPS paper figures).
PAPER_RCPARAMS = {
    "font.size": 10,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "axes.linewidth": 1.1,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
}

# scripts/evaluate_lunar_lander_fps.py's trajectory.npy layout: [x, y, vx, vy, current_fps]
NPY_COLS = {"x": 0, "y": 1, "vx": 2, "vy": 3, "current_fps": 4}


def parse_bool_column(series):
    """Robustly convert a boolean-like column to bool, handling True/False,
    "true"/"false" (any case), and 1/0 (numeric or string)."""
    if series.dtype == bool:
        return series
    return series.astype(str).str.strip().str.lower().map(
        {"true": True, "false": False, "1": True, "0": False}
    )


def parse_run_label(run_label):
    """Extract (frame_cost, budget) from an adaptive run directory name such as
    'adaptive_fc_2.0_bud_50.0'. Returns (None, None) for anything else (e.g.
    'fixed_10Hz') -- that's the normal, expected case for fixed-FPS runs, not an
    error."""
    match = re.search(r"adaptive_fc_([0-9.]+)_bud_([0-9.]+)", str(run_label))
    if match is None:
        return None, None
    try:
        return float(match.group(1)), float(match.group(2))
    except ValueError:
        return None, None


def format_fc_title(frame_cost):
    """Concise frame-cost value for titles, e.g. 2.0 -> '2', 0.0 -> '0'."""
    return "0" if frame_cost == 0.0 else f"{frame_cost:g}"


def format_condition(frame_cost, budget, fallback_desc):
    """Concise experimental-condition string for titles: '$f_c=..., B=...$' for
    adaptive runs, or the fixed-FPS description otherwise."""
    if frame_cost is not None and budget is not None:
        return f"$f_c={format_fc_title(frame_cost)}$, $B={int(budget)}$"
    return fallback_desc


def describe_run(episode_rows):
    """Human-readable identifier for this eval_dir's episodes.csv, e.g. 'mean_fps=8.25'."""
    if episode_rows.empty or "mean_fps" not in episode_rows.columns:
        return "unknown"
    return f"mean_fps={episode_rows.iloc[0]['mean_fps']:.2f}"


def save_figure(out_dir, fig, filename_stem):
    """Save a PNG (300 DPI) -- matching utils/plot_frame_acquisition.py's choice of
    PNG-only, no PDF -- and report the path."""
    png_path = out_dir / f"{filename_stem}.png"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {png_path}")
    return png_path


def load_episode_rows(eval_dir):
    df = pd.read_csv(eval_dir / "episodes.csv")
    for col in ("success", "landed_in_flags", "outside_flags_after_landing",
                "went_up_after", "exceed_vy_vel"):
        if col in df.columns:
            df[col] = parse_bool_column(df[col])
    return df.sort_values("episode_index").reset_index(drop=True)


def load_step_rows(eval_dir, episode_index):
    # episode_index (the CSV column) and the lap_<i> directory suffix are the same
    # integer -- folders are still named lap_<i> on disk even though the CSVs use
    # "episode" terminology.
    path = eval_dir / f"lap_{episode_index:02d}" / "steps.csv"
    df = pd.read_csv(path)
    if "fresh_observation" in df.columns:
        df["fresh_observation"] = parse_bool_column(df["fresh_observation"])
    return df


def find_trajectory_npy(eval_dir, episode_index):
    path = eval_dir / f"lap_{episode_index:02d}" / "trajectory.npy"
    if not path.exists():
        raise FileNotFoundError(f"No trajectory.npy found at {path}")
    return path


def find_touchdown_step(steps):
    """First control step where vy_at_touchdown is set -- the exact touchdown instant,
    as computed by the evaluator (not recomputed here). None if the episode never
    touched down. trajectory.npy has no equivalent column, so this always comes from
    steps.csv."""
    if "vy_at_touchdown" not in steps.columns:
        return None
    touchdown_rows = steps[steps["vy_at_touchdown"].notna()]
    if touchdown_rows.empty:
        return None
    return int(touchdown_rows.iloc[0]["step"])


def load_trajectory_npy(path):
    data = np.load(path)
    if data.shape[1] != len(NPY_COLS):
        raise ValueError(f"{path}: expected {len(NPY_COLS)} columns {list(NPY_COLS)}, "
                          f"got {data.shape[1]}")
    return {name: data[:, idx] for name, idx in NPY_COLS.items()}


def draw_ground(ax):
    """Draw the fixed reference geometry described in the module docstring: the
    ground/pad line, the two landing-flag markers, and the out-of-bounds playfield
    edges. Stands in for AdaptiveFPS's draw_track() (real racetrack raster + waypoint
    centerline), which has no Lunar Lander equivalent."""
    ax.axhline(0.0, color="0.4", linewidth=1.5, zorder=0)
    for x in (-PAD_HALF_WIDTH, PAD_HALF_WIDTH):
        ax.plot([x, x], [0.0, 0.12], color="tab:orange", linewidth=2, zorder=1)
        ax.plot(x, 0.12, marker=(">" if x < 0 else "<"), color="tab:orange",
                 markersize=6, zorder=1)
    for x in (-PLAYFIELD_HALF_WIDTH, PLAYFIELD_HALF_WIDTH):
        ax.axvline(x, color="0.75", linestyle=":", linewidth=1, zorder=0)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X Position (normalized)")
    ax.set_ylabel("Y Position (normalized)")
    for spine in ax.spines.values():
        spine.set_linewidth(1.1)


def plot_single_trajectory(npy_path, episode_index, condition, episode_length, touchdown_step):
    trajectory = load_trajectory_npy(npy_path)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    ax_speed, ax_fps, ax_vy = axes

    draw_ground(ax_speed)
    speed = np.sqrt(trajectory["vx"] ** 2 + trajectory["vy"] ** 2)
    scatter = ax_speed.scatter(trajectory["x"], trajectory["y"], c=speed,
                                cmap="viridis", s=10, zorder=2)
    fig.colorbar(scatter, ax=ax_speed, label="Speed (normalized)")
    ax_speed.set_title("Trajectory — Speed")

    draw_ground(ax_fps)
    fps_values = trajectory["current_fps"]
    unexpected = sorted(set(np.unique(fps_values)) - set(FPS_CHOICES))
    if unexpected:
        raise ValueError(f"{npy_path}: current_fps contains values outside the valid "
                          f"discrete choices {FPS_CHOICES}: {unexpected}")

    # Bucket each sample into its FPS's index, then use a colormap truncated to
    # exactly len(FPS_CHOICES) levels + integer-width bins -- no gradient between the
    # discrete categories, no implied intermediate values.
    fps_index = np.searchsorted(FPS_CHOICES, fps_values)
    cmap = matplotlib.colormaps["viridis"].resampled(len(FPS_CHOICES))
    norm = BoundaryNorm(np.arange(len(FPS_CHOICES) + 1) - 0.5, cmap.N)
    fps_scatter = ax_fps.scatter(trajectory["x"], trajectory["y"], c=fps_index,
                                  cmap=cmap, norm=norm, s=10, zorder=2)
    cbar = fig.colorbar(fps_scatter, ax=ax_fps, ticks=range(len(FPS_CHOICES)))
    cbar.ax.set_yticklabels([f"{fps} Hz" for fps in FPS_CHOICES])
    cbar.set_label("Sensing Rate (Hz)")
    ax_fps.set_title("Trajectory — Sensing Rate")

    control_steps = np.arange(1, len(trajectory["vy"]) + 1)
    ax_vy.plot(control_steps, trajectory["vy"], color="tab:blue", linewidth=1.3)
    if touchdown_step is not None:
        ax_vy.axvline(touchdown_step, color="tab:red", linestyle="--", linewidth=1.3,
                      label=f"Touchdown (step {touchdown_step})")
        ax_vy.legend(loc="best")
    ax_vy.set_xlabel("Control Step")
    ax_vy.set_ylabel("Vertical Velocity (vy)")
    ax_vy.set_title("Vertical Velocity Profile")
    ax_vy.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.35)
    ax_vy.set_axisbelow(True)
    for spine in ax_vy.spines.values():
        spine.set_linewidth(1.1)

    length_label = f"{episode_length} steps" if episode_length is not None else "length unknown"
    fig.suptitle(f"Episode {episode_index} — {condition} — {length_label}")
    fig.tight_layout()

    return fig


def plot_overlay(eval_dir, episode_rows, condition, episode_indices=None):
    rows = episode_rows
    if episode_indices is not None:
        rows = rows[rows["episode_index"].isin(episode_indices)]

    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    draw_ground(ax)

    annotate_endpoints = len(rows) <= MAX_ANNOTATED_ENDPOINTS
    if not annotate_endpoints:
        print(f"Note: {len(rows)} episodes overlaid - skipping per-episode endpoint "
              f"labels (would overlap); showing trajectory structure only.")

    success_labelled, failure_labelled, failure_loc_labelled = False, False, False
    for i, (_, row) in enumerate(rows.iterrows()):
        episode_index = int(row["episode_index"])
        steps = load_step_rows(eval_dir, episode_index)
        xs = steps["x"].to_numpy()
        ys = steps["y"].to_numpy()

        if row["success"]:
            ax.plot(xs, ys, color="tab:green", alpha=0.6, linewidth=1,
                    label="Successful" if not success_labelled else None, zorder=2)
            success_labelled = True
        else:
            ax.plot(xs, ys, color="tab:red", alpha=0.6, linewidth=1,
                    label="Failed" if not failure_labelled else None, zorder=2)
            ax.plot(xs[-1], ys[-1], "x", color="black", markersize=8, zorder=3,
                    label="Failure Location" if not failure_loc_labelled else None)
            failure_labelled = True
            failure_loc_labelled = True

        if annotate_endpoints:
            # Unlike an F1TENTH finish line (spread around a large track), every
            # Lunar Lander episode ends within the tiny landing-pad zone -- a fixed
            # (3, 3) offset would stack every label on top of each other regardless
            # of episode count, so each one is fanned out vertically by its row index
            # instead, with a thin leader line back to its actual endpoint.
            ax.annotate(
                f"ep{episode_index} ({int(row['episode_length'])} steps)",
                (xs[-1], ys[-1]), fontsize=6, color="0.2",
                xytext=(30, 20 + 14 * i), textcoords="offset points",
                arrowprops=dict(arrowstyle="-", color="0.6", linewidth=0.5, alpha=0.7),
            )

    ax.legend(loc="best", frameon=True, framealpha=0.95, edgecolor="black",
              borderpad=0.6, handletextpad=0.6)

    n_success = int(rows["success"].sum())
    success_pct = 100.0 * n_success / len(rows) if len(rows) else float("nan")
    ax.set_title(f"Evaluation Trajectories — Lunar Lander\n"
                 f"{condition} — Success Rate: {success_pct:.0f}%")
    fig.tight_layout()

    return fig


def plot_timeseries(eval_dir, episode_indices, condition):
    fig, axes = plt.subplots(4, 1, figsize=(8, 2.3 * 4), sharex=True)
    ax_alt, ax_reward, ax_return, ax_fps = axes

    show_legend = len(episode_indices) <= MAX_ANNOTATED_ENDPOINTS
    if not show_legend:
        print(f"Note: {len(episode_indices)} episodes requested - omitting the "
              f"per-episode legend to avoid clutter.")

    for episode_index in episode_indices:
        steps = load_step_rows(eval_dir, episode_index)
        control_steps = steps["step"].to_numpy()
        label = f"Episode {episode_index}"

        ax_alt.plot(control_steps, steps["y"], label=label, linewidth=1.3)
        ax_reward.plot(control_steps, steps["instant_nav_reward"], label=label, linewidth=1.3)
        ax_return.plot(control_steps, steps["cumulative_nav_reward"], label=label, linewidth=1.3)
        ax_fps.plot(control_steps, steps["current_fps"], label=label, linewidth=1.3)

    ax_alt.set_ylabel("Altitude (Y)")
    ax_reward.set_ylabel("Instantaneous Nav Reward")
    ax_return.set_ylabel("Cumulative Nav Reward")
    ax_fps.set_ylabel("Sensing Rate (Hz)")
    axes[-1].set_xlabel("Control Step")

    for ax in axes:
        ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.35)
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_linewidth(1.1)

    if show_legend:
        ax_alt.legend(loc="best", frameon=True, framealpha=0.95, edgecolor="black",
                      borderpad=0.6, handletextpad=0.6)

    fig.suptitle(f"Episode Dynamics — Lunar Lander\n{condition}")
    fig.tight_layout()

    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--eval-dir", required=True,
                         help="policy evaluation directory, e.g. eval/adaptive_fc_2.0_bud_50.0 or eval/fixed_10Hz")
    parser.add_argument("--episodes", type=int, nargs="+", default=None,
                         help="restrict --overlay/--timeseries to these episode indices")
    parser.add_argument("--npy-episode", type=int, default=None,
                         help="plot one episode's trajectory.npy (track colored by speed, and by sensing rate, plus a vertical-velocity profile)")
    parser.add_argument("--overlay", action="store_true", help="overlay successful/failed trajectories")
    parser.add_argument("--timeseries", action="store_true", help="plot altitude/reward/return/FPS over time for --episodes")
    parser.add_argument("--output-dir", default=None,
                         help="where to save figures; defaults to --eval-dir itself (matching utils/plot_frame_acquisition.py)")
    args = parser.parse_args()

    if not (args.npy_episode is not None or args.overlay or args.timeseries):
        parser.error("nothing to do: pass --npy-episode, --overlay, or --timeseries")

    plt.rcParams.update(PAPER_RCPARAMS)

    eval_dir = Path(args.eval_dir.rstrip("/"))
    if not eval_dir.exists():
        raise FileNotFoundError(f"Evaluation directory not found: {eval_dir}")
    out_dir = Path(args.output_dir) if args.output_dir else eval_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    episode_rows = load_episode_rows(eval_dir)
    frame_cost, budget = parse_run_label(eval_dir.name)
    condition = format_condition(frame_cost, budget, describe_run(episode_rows))

    if args.npy_episode is not None:
        npy_path = find_trajectory_npy(eval_dir, args.npy_episode)
        matching = episode_rows[episode_rows["episode_index"] == args.npy_episode]
        episode_length = int(matching["episode_length"].iloc[0]) if len(matching) else None
        touchdown_step = find_touchdown_step(load_step_rows(eval_dir, args.npy_episode))
        fig = plot_single_trajectory(npy_path, args.npy_episode, condition, episode_length, touchdown_step)
        save_figure(out_dir, fig, f"trajectory_episode{args.npy_episode:02d}")

    if args.overlay:
        fig = plot_overlay(eval_dir, episode_rows, condition, episode_indices=args.episodes)
        extra = "_episodes" + "-".join(f"{i:02d}" for i in args.episodes) if args.episodes else ""
        save_figure(out_dir, fig, f"overlay{extra}")

    if args.timeseries:
        episode_indices = args.episodes if args.episodes is not None else episode_rows["episode_index"].tolist()
        fig = plot_timeseries(eval_dir, episode_indices, condition)
        extra = "-".join(f"{i:02d}" for i in episode_indices)
        save_figure(out_dir, fig, f"timeseries_episodes{extra}")


if __name__ == "__main__":
    main()
