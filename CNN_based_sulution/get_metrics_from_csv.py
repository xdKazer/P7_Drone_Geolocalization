import csv
from pathlib import Path


def safe_float(val):
    """Convert to float, return None if it's 'N/A' or invalid."""
    if val is None:
        return None
    val = val.strip()
    if val.upper() == "N/A" or val == "":
        return None
    try:
        return float(val)
    except ValueError:
        return None


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def get_metrics(csv_path):
    """
    Computes metrics from a results CSV file.

    Returns a dictionary containing all mean metrics.
    """
    csv_path = Path(csv_path)

    # Accumulators
    errors = []
    ekf_errors = []
    heading_diffs = []
    ekf_heading_diffs = []
    dxs = []
    dys = []
    dxs_ekf = []
    dys_ekf = []
    times = []
    ekf_times = []

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Metrics
            e          = safe_float(row.get("error"))
            e_ekf      = safe_float(row.get("ekf_error"))
            hd         = safe_float(row.get("heading_diff"))
            hd_ekf     = safe_float(row.get("ekf_heading_diff"))
            dx         = safe_float(row.get("dx"))
            dy         = safe_float(row.get("dy"))
            dx_ekf_val = safe_float(row.get("dx_ekf"))
            dy_ekf_val = safe_float(row.get("dy_ekf"))
            t          = safe_float(row.get("time_s"))
            t_ekf      = safe_float(row.get("ekf_time_s"))

            if e is not None: errors.append(e)
            if e_ekf is not None: ekf_errors.append(e_ekf)
            if hd is not None: heading_diffs.append(hd)
            if hd_ekf is not None: ekf_heading_diffs.append(hd_ekf)
            if dx is not None: dxs.append(dx)
            if dy is not None: dys.append(dy)
            if dx_ekf_val is not None: dxs_ekf.append(dx_ekf_val)
            if dy_ekf_val is not None: dys_ekf.append(dy_ekf_val)
            if t is not None: times.append(t)
            if t_ekf is not None: ekf_times.append(t_ekf)

    # Compute means
    results = {
        "mean_error_m": mean(errors),
        "mean_error_m_ekf": mean(ekf_errors),
        "mean_heading_deg": mean(heading_diffs),
        "mean_heading_deg_ekf": mean(ekf_heading_diffs),
        "mean_dx_m": mean(dxs),
        "mean_dy_m": mean(dys),
        "mean_dx_m_ekf": mean(dxs_ekf),
        "mean_dy_m_ekf": mean(dys_ekf),
        "mean_time_s": mean(times),
        "mean_time_s_ekf": mean(ekf_times),
    }

    return results
