import json
import os
import random
import subprocess
import sys
from itertools import product
from pathlib import Path

# Path to your main script
SCRIPT = Path(__file__).parent / "LG_for_parameter_tuning.py"

# Where to store sweep results
RESULTS_JSON = Path(__file__).parent / "param_sweep_results.json"

# Which metric to optimize (must be a key from get_metrics)
OBJECTIVE_KEYS = [
    "rmse_error_m_ekf",      # primary: lower is better
    "mean_error_m_ekf",      # fallback if present
    "std_error_m_ekf",       # fallback if present
]

# Global parameter bounds (used for refinement)
PARAM_BOUNDS = {
    "POS_BASE_STD":      (40.0, 80.0),
    "POS_MIN_SCALE":     (0.1, 1.0),
    "POS_MAX_SCALE":     (1.2, 5.0),
    "HEADING_BASE_STD_DEG": (2.0, 8.0),
    "HEADING_MIN_SCALE": (0.1, 1.2),
    "HEADING_MAX_SCALE": (1.5, 5.0),
    "MIN_CONF_FOR_EKF":  (0.05, 0.4),
    "WEIGHT_LG_SHAPE":   (0.0, 1.0),
    "S_ERR":             (1.0, 5.0),
    "N_ERR_MIN":         (40, 150),   # int
    "ALPHA_ERR":         (0.0, 0.5),
}

# Coarse grid (for a subset of parameters)
# We will sample from the full cartesian product of these.
COARSE_GRID = {
    "POS_BASE_STD":        [45.0, 60.0, 75.0],
    "POS_MIN_SCALE":       [0.1, 0.55, 1.0],
    "POS_MAX_SCALE":       [1.5, 3.1, 4.7],
    "HEADING_BASE_STD_DEG":[2.0, 5.0, 8.0],
    "MIN_CONF_FOR_EKF":    [0.05, 0.225, 0.4],
    "WEIGHT_LG_SHAPE":     [0.2, 0.35, 0.6],
}

# Parameters that stay fixed during the coarse grid phase
FIXED_PARAMS = {
    "HEADING_MIN_SCALE": 0.5,
    "HEADING_MAX_SCALE": 2.75,
    "S_ERR": 2.0,
    "N_ERR_MIN": 80,
    "ALPHA_ERR": 0.2,
}


def run_once(env_overrides):
    """
    Run LG_for_parameter_tuning.py with given env vars.
    Returns a dict with metrics or None if run failed.
    Assumes LG_for_parameter_tuning.py ends with:

        results = main(HP)
        print(results)

    where 'results' is a dict (from get_metrics).
    """
    env = os.environ.copy()
    env.update(env_overrides)

    proc = subprocess.run(
        [sys.executable, str(SCRIPT)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    # Debug info on failures
    if proc.returncode != 0:
        print("Run failed with return code", proc.returncode)
        print("stderr:\n", proc.stderr)
        return None

    # Optional: show stderr if there are warnings
    if proc.stderr.strip():
        print("stderr (warnings from LG script):")
        print(proc.stderr)

    # Get last non-empty line (we assume that's the printed metrics dict)
    last_line = ""
    for line in proc.stdout.strip().splitlines()[::-1]:
        if line.strip():
            last_line = line.strip()
            break

    if not last_line:
        print("No output from LG_for_parameter_tuning.py to parse.")
        return {"raw_output": proc.stdout}

    # Very simple eval – relies on Python repr of a dict.
    # For extra safety you could make the LG script print JSON instead.
    try:
        metrics = eval(last_line, {"__builtins__": {}})
    except Exception as e:
        print("Failed to parse metrics from output line:")
        print(last_line)
        print("Error:", e)
        metrics = {"raw_output": proc.stdout}

    return metrics


def extract_score(metrics):
    """
    Extract a scalar score from the metrics dict.
    Lower is better (we're minimizing error).
    """
    if not isinstance(metrics, dict):
        return None

    for key in OBJECTIVE_KEYS:
        if key in metrics:
            try:
                return float(metrics[key])
            except (TypeError, ValueError):
                continue

    return None


def save_results(results):
    """
    Save results (list of dicts) to RESULTS_JSON.
    Tries to keep things JSON-serializable by converting metrics to float/str.
    """
    clean_results = []
    for r in results:
        clean = {
            "params": r["params"],
            "metrics": {},
            "score": r["score"],
            "phase": r.get("phase", "unknown"),
        }
        metrics = r.get("metrics", {})
        if isinstance(metrics, dict):
            for k, v in metrics.items():
                try:
                    clean["metrics"][k] = float(v)
                except (TypeError, ValueError):
                    clean["metrics"][k] = str(v)
        else:
            clean["metrics"] = str(metrics)
        clean_results.append(clean)

    with open(RESULTS_JSON, "w") as f:
        json.dump(clean_results, f, indent=2)


def sort_results_by_score(results):
    """
    Sort results list ascending by score (None scores go last).
    """
    return sorted(
        results,
        key=lambda r: float("inf") if r["score"] is None else r["score"]
    )


def grid_search(param_grid, fixed_params, max_trials=48, seed=42):
    """
    Phase 1: Sampled grid search on a subset of parameters.

    param_grid: dict of param -> list of values
    fixed_params: dict of param -> fixed value (used during grid phase)
    max_trials: max number of grid points to evaluate
    seed: RNG seed for reproducible sampling of the grid
    """
    random.seed(seed)

    keys = list(param_grid.keys())
    value_lists = [param_grid[k] for k in keys]

    all_combos = list(product(*value_lists))
    print(f"Total grid points in full grid: {len(all_combos)}")

    if max_trials is not None and max_trials < len(all_combos):
        all_combos = random.sample(all_combos, max_trials)
        print(f"Sampling {len(all_combos)} grid points from full grid.")

    results = []

    for i, combo in enumerate(all_combos, start=1):
        params = dict(zip(keys, combo))
        # add fixed params
        params.update(fixed_params)

        env_overrides = {k: str(v) for k, v in params.items()}

        print(f"\n=== GRID Trial {i}/{len(all_combos)} ===")
        print("Params:", env_overrides)

        metrics = run_once(env_overrides)
        if metrics is None:
            print("Skipping trial due to run failure.")
            continue

        score = extract_score(metrics)

        trial_result = {
            "params": params,
            "metrics": metrics,
            "score": score,
            "phase": "grid",
        }
        results.append(trial_result)

        print("Metrics:", metrics)
        print("Score (objective):", score)

        # Save intermediate results
        save_results(results)

    return results


def make_local_bounds(center, name, shrink_factor=0.3):
    """
    Make local bounds around a center value for refinement phase.

    shrink_factor is fraction of full global range to use around center.
    Example: if full range is [0, 10] and shrink_factor=0.3,
             local width is 3.0, so we take [center-1.5, center+1.5] (clipped).
    """
    gmin, gmax = PARAM_BOUNDS[name]
    full_range = gmax - gmin
    local_width = full_range * shrink_factor
    half = local_width / 2.0

    low = max(gmin, center - half)
    high = min(gmax, center + half)

    if name == "N_ERR_MIN":
        low = int(round(low))
        high = int(round(high))
        if low == high:
            high = min(high + 1, int(gmax))
    return low, high


def refine_random_search(base_results, n_best=4, trials_per_best=12, seed=123):
    """
    Phase 2: local random search around the best grid points.

    base_results: list of trial dicts (from grid_search)
    n_best: number of best grid points to refine
    trials_per_best: random samples around each best point
    seed: RNG seed
    """
    random.seed(seed)

    sorted_base = sort_results_by_score(base_results)
    centers = [r for r in sorted_base[:n_best] if r["score"] is not None]

    all_refine_results = []

    for idx, center_trial in enumerate(centers):
        center_params = center_trial["params"]
        print(f"\n=== Refinement around center #{idx+1} ===")
        print("Center score:", center_trial["score"])
        print("Center params:", center_params)

        for j in range(trials_per_best):
            params = {}
            for name, (gmin, gmax) in PARAM_BOUNDS.items():
                center_val = center_params[name]
                low, high = make_local_bounds(center_val, name, shrink_factor=0.3)

                if name == "N_ERR_MIN":
                    params[name] = random.randint(low, high)
                else:
                    params[name] = random.uniform(low, high)

            env_overrides = {k: str(v) for k, v in params.items()}

            trial_idx = idx * trials_per_best + (j + 1)
            total_trials = n_best * trials_per_best
            print(f"\n--- Refinement trial {trial_idx}/{total_trials} ---")
            print("Params:", env_overrides)

            metrics = run_once(env_overrides)
            if metrics is None:
                print("Skipping trial due to run failure.")
                continue

            score = extract_score(metrics)

            trial_result = {
                "params": params,
                "metrics": metrics,
                "score": score,
                "phase": "refine",
                "center_score": center_trial["score"],
            }
            all_refine_results.append(trial_result)

            print("Metrics:", metrics)
            print("Score (objective):", score)

            # Append to existing grid results and save
            combined = base_results + all_refine_results
            save_results(combined)

    return all_refine_results


if __name__ == "__main__":
    # With 8 hours and ~5 minutes per run, you can do ~96 runs.
    # Here we choose:
    # - 48 sampled grid points (Phase 1)
    # - 4 best centers * 12 refinement trials each = 48 (Phase 2)
    # Total ~96

    # Phase 1: coarse grid
    grid_results = grid_search(
        param_grid=COARSE_GRID,
        fixed_params=FIXED_PARAMS,
        max_trials=48,
        seed=42,
    )

    # Phase 2: refinement
    refine_results = refine_random_search(
        base_results=grid_results,
        n_best=4,
        trials_per_best=12,
        seed=123,
    )

    all_results = grid_results + refine_results
    best_sorted = sort_results_by_score(all_results)

    print("\n===== BEST CONFIGS OVERALL =====")
    for r in best_sorted[:10]:
        print("\nphase:", r.get("phase"))
        print("score:", r["score"])
        print("params:", r["params"])
        print("metrics:", r["metrics"])
