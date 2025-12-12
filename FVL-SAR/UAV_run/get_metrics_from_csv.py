import csv
import math
import statistics
from pathlib import Path


def safe_float(val):
    """Convert to float, return None if it's 'N/A' or invalid."""
    if val is None:
        return None
    val = str(val).strip()
    if val.upper() == "N/A" or val == "":
        return None
    try:
        return float(val)
    except ValueError:
        return None


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def rmse(xs):
    """Root-mean-square error."""
    if not xs:
        return float("nan")
    return math.sqrt(sum(x * x for x in xs) / len(xs))


def std(xs):
    """Sample standard deviation (NaN if <2 samples)."""
    if len(xs) < 2:
        return float("nan")
    return statistics.stdev(xs)


def get_metrics(csv_path, feature_threshold):
    """
    Computes metrics from a results CSV file, split into:
      - 'overall'     : all rows that have a valid features_drone value
      - 'less_than'   : features_drone < feature_threshold
      - 'greater_equal': features_drone >= feature_threshold

    Each group has the same metrics as the original function:
      mean_error_m, rmse_error_m, std_error_m, std_time_s, etc.
    """
    csv_path = Path(csv_path)

    # ----- Overall accumulators -----
    errors_all = []
    ekf_errors_all = []
    heading_diffs_all = []
    ekf_heading_diffs_all = []
    dxs_all = []
    dys_all = []
    dxs_ekf_all = []
    dys_ekf_all = []
    times_all = []
    ekf_times_all = []

    unsuccessful_no_homography_all = 0
    unsuccessful_low_confidence_all = 0
    unsuccessful_unknown_confidence_all = 0
    unsuccessful_matches_ekf_all = 0
    total_rows_all = 0

    # ----- Accumulators for < threshold features -----
    errors_lt = []
    ekf_errors_lt = []
    heading_diffs_lt = []
    ekf_heading_diffs_lt = []
    dxs_lt = []
    dys_lt = []
    dxs_ekf_lt = []
    dys_ekf_lt = []
    times_lt = []
    ekf_times_lt = []

    unsuccessful_no_homography_lt = 0
    unsuccessful_low_confidence_lt = 0
    unsuccessful_unknown_confidence_lt = 0
    unsuccessful_matches_ekf_lt = 0
    total_rows_lt = 0

    # ----- Accumulators for >= threshold features -----
    errors_ge = []
    ekf_errors_ge = []
    heading_diffs_ge = []
    ekf_heading_diffs_ge = []
    dxs_ge = []
    dys_ge = []
    dxs_ekf_ge = []
    dys_ekf_ge = []
    times_ge = []
    ekf_times_ge = []

    unsuccessful_no_homography_ge = 0
    unsuccessful_low_confidence_ge = 0
    unsuccessful_unknown_confidence_ge = 0
    unsuccessful_matches_ekf_ge = 0
    total_rows_ge = 0

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Decide which group this row belongs to (and whether we use it at all)
            features = safe_float(row.get("features_drone"))
            if features is None:
                # Can't group without features_drone; skip this row entirely
                continue

            # Parse metrics for this row
            e          = safe_float(row.get("error"))
            e_ekf      = safe_float(row.get("ekf_error"))
            hd         = safe_float(row.get("heading_diff"))
            hd_ekf     = safe_float(row.get("ekf_heading_diff"))
            dx         = safe_float(row.get("dx"))
            dy         = safe_float(row.get("dy"))
            dx_ekf_val = safe_float(row.get("dx_ekf"))
            dy_ekf_val = safe_float(row.get("dy_ekf"))
            t          = safe_float(row.get("time_s"))
            t_ekf      = safe_float(row.get("ekf_time_s"))  # may be missing
            overall_conf = safe_float(row.get("overall_confidence"))

            # ---- Update OVERALL accumulators ----
            total_rows_all += 1

            # Split unsuccessful raw matches by overall_confidence
            if e is None:
                if overall_conf is None:
                    unsuccessful_unknown_confidence_all += 1
                elif overall_conf == 0:
                    unsuccessful_no_homography_all += 1
                else:
                    unsuccessful_low_confidence_all += 1
            else:
                errors_all.append(e)

            # EKF error handling
            if e_ekf is None:
                unsuccessful_matches_ekf_all += 1
            else:
                ekf_errors_all.append(e_ekf)

            # Other metrics
            if hd is not None: heading_diffs_all.append(hd)
            if hd_ekf is not None: ekf_heading_diffs_all.append(hd_ekf)
            if dx is not None: dxs_all.append(dx)
            if dy is not None: dys_all.append(dy)
            if dx_ekf_val is not None: dxs_ekf_all.append(dx_ekf_val)
            if dy_ekf_val is not None: dys_ekf_all.append(dy_ekf_val)
            if t is not None: times_all.append(t)
            if t_ekf is not None: ekf_times_all.append(t_ekf)

            # ---- Group label ----
            if features < feature_threshold:
                group = "lt"
            else:
                group = "ge"

            # ---- Update group-specific accumulators ----
            if group == "lt":
                total_rows_lt += 1

                # Split unsuccessful raw matches by overall_confidence
                if e is None:
                    if overall_conf is None:
                        unsuccessful_unknown_confidence_lt += 1
                    elif overall_conf == 0:
                        unsuccessful_no_homography_lt += 1
                    else:
                        unsuccessful_low_confidence_lt += 1
                else:
                    errors_lt.append(e)

                # EKF error handling
                if e_ekf is None:
                    unsuccessful_matches_ekf_lt += 1
                else:
                    ekf_errors_lt.append(e_ekf)

                # Other metrics
                if hd is not None: heading_diffs_lt.append(hd)
                if hd_ekf is not None: ekf_heading_diffs_lt.append(hd_ekf)
                if dx is not None: dxs_lt.append(dx)
                if dy is not None: dys_lt.append(dy)
                if dx_ekf_val is not None: dxs_ekf_lt.append(dx_ekf_val)
                if dy_ekf_val is not None: dys_ekf_lt.append(dy_ekf_val)
                if t is not None: times_lt.append(t)
                if t_ekf is not None: ekf_times_lt.append(t_ekf)

            else:  # group == "ge"
                total_rows_ge += 1

                # Split unsuccessful raw matches by overall_confidence
                if e is None:
                    if overall_conf is None:
                        unsuccessful_unknown_confidence_ge += 1
                    elif overall_conf == 0:
                        unsuccessful_no_homography_ge += 1
                    else:
                        unsuccessful_low_confidence_ge += 1
                else:
                    errors_ge.append(e)

                # EKF error handling
                if e_ekf is None:
                    unsuccessful_matches_ekf_ge += 1
                else:
                    ekf_errors_ge.append(e_ekf)

                # Other metrics
                if hd is not None: heading_diffs_ge.append(hd)
                if hd_ekf is not None: ekf_heading_diffs_ge.append(hd_ekf)
                if dx is not None: dxs_ge.append(dx)
                if dy is not None: dys_ge.append(dy)
                if dx_ekf_val is not None: dxs_ekf_ge.append(dx_ekf_val)
                if dy_ekf_val is not None: dys_ekf_ge.append(dy_ekf_val)
                if t is not None: times_ge.append(t)
                if t_ekf is not None: ekf_times_ge.append(t_ekf)

    # ---- Build results for each group ----

    # Overall
    unsuccessful_matches_all = (
        unsuccessful_no_homography_all
        + unsuccessful_low_confidence_all
        + unsuccessful_unknown_confidence_all
    )

    results_all = {
        "mean_error_m": mean(errors_all),
        "mean_error_m_ekf": mean(ekf_errors_all),
        "mean_heading_deg": mean(heading_diffs_all),
        "mean_heading_deg_ekf": mean(ekf_heading_diffs_all),
        "mean_dx_m": mean(dxs_all),
        "mean_dy_m": mean(dys_all),
        "mean_dx_m_ekf": mean(dxs_ekf_all),
        "mean_dy_m_ekf": mean(dys_ekf_all),
        "mean_time_s": mean(times_all),
        "mean_time_s_ekf": mean(ekf_times_all),

        "rmse_error_m": rmse(errors_all),
        "rmse_error_m_ekf": rmse(ekf_errors_all),

        "std_error_m": std(errors_all),
        "std_error_m_ekf": std(ekf_errors_all),

        "std_time_s": std(times_all),
        "std_time_s_ekf": std(ekf_times_all),

        "unsuccessful_matches": unsuccessful_matches_all,
        "unsuccessful_no_homography": unsuccessful_no_homography_all,
        "unsuccessful_low_confidence": unsuccessful_low_confidence_all,
        "unsuccessful_unknown_confidence": unsuccessful_unknown_confidence_all,

        "unsuccessful_matches_ekf": unsuccessful_matches_ekf_all,

        "total_rows": total_rows_all,
        "successful_rows": len(errors_all),
    }

    # < threshold
    unsuccessful_matches_lt = (
        unsuccessful_no_homography_lt
        + unsuccessful_low_confidence_lt
        + unsuccessful_unknown_confidence_lt
    )

    results_lt = {
        "mean_error_m": mean(errors_lt),
        "mean_error_m_ekf": mean(ekf_errors_lt),
        "mean_heading_deg": mean(heading_diffs_lt),
        "mean_heading_deg_ekf": mean(ekf_heading_diffs_lt),
        "mean_dx_m": mean(dxs_lt),
        "mean_dy_m": mean(dys_lt),
        "mean_dx_m_ekf": mean(dxs_ekf_lt),
        "mean_dy_m_ekf": mean(dys_ekf_lt),
        "mean_time_s": mean(times_lt),
        "mean_time_s_ekf": mean(ekf_times_lt),

        "rmse_error_m": rmse(errors_lt),
        "rmse_error_m_ekf": rmse(ekf_errors_lt),

        "std_error_m": std(errors_lt),
        "std_error_m_ekf": std(ekf_errors_lt),

        "std_time_s": std(times_lt),
        "std_time_s_ekf": std(ekf_times_lt),

        "unsuccessful_matches": unsuccessful_matches_lt,
        "unsuccessful_no_homography": unsuccessful_no_homography_lt,
        "unsuccessful_low_confidence": unsuccessful_low_confidence_lt,
        "unsuccessful_unknown_confidence": unsuccessful_unknown_confidence_lt,

        "unsuccessful_matches_ekf": unsuccessful_matches_ekf_lt,

        "total_rows": total_rows_lt,
        "successful_rows": len(errors_lt),
    }

    # >= threshold
    unsuccessful_matches_ge = (
        unsuccessful_no_homography_ge
        + unsuccessful_low_confidence_ge
        + unsuccessful_unknown_confidence_ge
    )

    results_ge = {
        "mean_error_m": mean(errors_ge),
        "mean_error_m_ekf": mean(ekf_errors_ge),
        "mean_heading_deg": mean(heading_diffs_ge),
        "mean_heading_deg_ekf": mean(ekf_heading_diffs_ge),
        "mean_dx_m": mean(dxs_ge),
        "mean_dy_m": mean(dys_ge),
        "mean_dx_m_ekf": mean(dxs_ekf_ge),
        "mean_dy_m_ekf": mean(dys_ekf_ge),
        "mean_time_s": mean(times_ge),
        "mean_time_s_ekf": mean(ekf_times_ge),

        "rmse_error_m": rmse(errors_ge),
        "rmse_error_m_ekf": rmse(ekf_errors_ge),

        "std_error_m": std(errors_ge),
        "std_error_m_ekf": std(ekf_errors_ge),

        "std_time_s": std(times_ge),
        "std_time_s_ekf": std(ekf_times_ge),

        "unsuccessful_matches": unsuccessful_matches_ge,
        "unsuccessful_no_homography": unsuccessful_no_homography_ge,
        "unsuccessful_low_confidence": unsuccessful_low_confidence_ge,
        "unsuccessful_unknown_confidence": unsuccessful_unknown_confidence_ge,

        "unsuccessful_matches_ekf": unsuccessful_matches_ekf_ge,

        "total_rows": total_rows_ge,
        "successful_rows": len(errors_ge),
    }

    # Return overall + the two groups
    return {
        "overall": results_all,
        "less_than": results_lt,
        "greater_equal": results_ge,
    }

"""
# --- Example usage ---
metrics = get_metrics(
    "C:/Users/signe/P7_Drone_Geolocalization/CNN_based_sulution/UAV_run/outputs/03/results_03.csv",
    2000,
)

print("OVERALL:",
      "mean error:", metrics["overall"]["mean_error_m"],
      "rmse:", metrics["overall"]["rmse_error_m"],
      "STD:", metrics["overall"]["std_error_m"],
      "---------------",
      "mean error ekf:", metrics["overall"]["mean_error_m_ekf"],
      "rmse ekf:", metrics["overall"]["rmse_error_m_ekf"],
      "STD ekf:", metrics["overall"]["std_error_m_ekf"],
      "---------------",
      "mean time:", metrics["overall"]["mean_time_s"],
      "unsuccessful matches:", metrics["overall"]["unsuccessful_matches"],
      "unsuccessful matches ekf:", metrics["overall"]["unsuccessful_matches_ekf"],
      "total_rows:", metrics["overall"]["total_rows"])

for group_name, label in [("less_than", "< 2000 feats"),
                          ("greater_equal", "≥ 2000 feats")]:
    m = metrics[group_name]
    print(label,
          "mean error:", m["mean_error_m"],
          "rmse:", m["rmse_error_m"],
          "STD:", m["std_error_m"],
          "mean time:", m["mean_time_s"],
          "unsuccessful matches:", m["unsuccessful_matches"],
          "total_rows:", m["total_rows"])
"""