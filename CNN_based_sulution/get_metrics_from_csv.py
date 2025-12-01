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


def get_metrics(csv_path):
    """
    Computes metrics from a results CSV file.

    Adds:
      - RMSE + STD for position errors (error, ekf_error)
      - STD for time (time_s, ekf_time_s if present)
      - unsuccessful split:
           * error N/A + overall_confidence == 0  -> no homography
           * error N/A + overall_confidence > 0   -> low confidence
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

    # Counters for N/A error causes
    unsuccessful_no_homography = 0
    unsuccessful_low_confidence = 0
    unsuccessful_unknown_confidence = 0

    unsuccessful_matches_ekf = 0
    total_rows = 0

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total_rows += 1

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
            t_ekf      = safe_float(row.get("ekf_time_s"))  # may be missing in some CSVs

            # Split unsuccessful raw matches by overall_confidence
            if e is None:
                overall_conf = safe_float(row.get("overall_confidence"))
                if overall_conf is None:
                    unsuccessful_unknown_confidence += 1
                elif overall_conf == 0:
                    unsuccessful_no_homography += 1
                else:  # overall_conf > 0
                    unsuccessful_low_confidence += 1
            else:
                errors.append(e)

            # EKF error handling
            if e_ekf is None:
                unsuccessful_matches_ekf += 1
            else:
                ekf_errors.append(e_ekf)

            # Other metrics
            if hd is not None: heading_diffs.append(hd)
            if hd_ekf is not None: ekf_heading_diffs.append(hd_ekf)
            if dx is not None: dxs.append(dx)
            if dy is not None: dys.append(dy)
            if dx_ekf_val is not None: dxs_ekf.append(dx_ekf_val)
            if dy_ekf_val is not None: dys_ekf.append(dy_ekf_val)
            if t is not None: times.append(t)
            if t_ekf is not None: ekf_times.append(t_ekf)

    unsuccessful_matches = (
        unsuccessful_no_homography
        + unsuccessful_low_confidence
        + unsuccessful_unknown_confidence
    )

    results = {
        # Means (original)
        "mean_error_m": mean(errors),
        "mean_error_m_ekf": mean(ekf_errors),
        #"mean_heading_deg": mean(heading_diffs),
        #"mean_heading_deg_ekf": mean(ekf_heading_diffs),
        #"mean_dx_m": mean(dxs),
        #"mean_dy_m": mean(dys),
        #"mean_dx_m_ekf": mean(dxs_ekf),
        #"mean_dy_m_ekf": mean(dys_ekf),
        #"mean_time_s": mean(times),
        #"mean_time_s_ekf": mean(ekf_times),

        # RMSE for position error
        "rmse_error_m": rmse(errors),
        "rmse_error_m_ekf": rmse(ekf_errors),

        # STD for position error
        "std_error_m": std(errors),
        "std_error_m_ekf": std(ekf_errors),

        # STD for time
        #"std_time_s": std(times),
        #"std_time_s_ekf": std(ekf_times),

        # Unsuccessful split counts
        #"unsuccessful_matches": unsuccessful_matches,
        #"unsuccessful_no_homography": unsuccessful_no_homography,
        #"unsuccessful_low_confidence": unsuccessful_low_confidence,
        #"unsuccessful_unknown_confidence": unsuccessful_unknown_confidence,

        # EKF unsuccessful (optional)
        #"unsuccessful_matches_ekf": unsuccessful_matches_ekf,

        "total_rows": total_rows,
        "successful_rows": len(errors),
    }

    return results
