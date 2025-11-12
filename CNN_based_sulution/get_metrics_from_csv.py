import csv
from pathlib import Path

WORKSPACE = Path(__file__).parent.resolve()

sat_number = "03"

mean_error_m = []
mean_error_m_ekf = []

mean_error_heading_deg = []
mean_error_heading_deg_ekf = []

mean_error_dx_m = []
mean_error_dy_m = []

mean_error_dx_m_ekf = []
mean_error_dy_m_ekf = []

CSV_RESULT_PATH = WORKSPACE / "outputs" / sat_number / "results_03.csv"

with open(CSV_RESULT_PATH, newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        num_inliers = int(row["inliers"])
        avg_confidence = float(row["avg_confidence"])
        median_err_px = float(row["median_reproj_error"])

