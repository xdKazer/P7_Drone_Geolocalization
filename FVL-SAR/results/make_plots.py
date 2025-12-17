import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ============================
# Config
# ============================

# Path to your CSV file
BASE = Path(__file__).parent.resolve()

CSV_PATH = BASE / "collected_results_known_heading.csv"

# Directory to save plots
OUTPUT_DIR = BASE / "plots"


def main():
    # ============================
    # Load data
    # ============================
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Read CSV, treating "N/A" as missing
    df = pd.read_csv(CSV_PATH, na_values=["N/A"], engine="python", usecols=range(25))

    # ----------------------------
    # 1) Features vs error & ekf_error
    # ----------------------------
    plt.figure()
    plt.scatter(df["features_drone"], df["error"], label="Measurement error")
    plt.scatter(df["features_drone"], df["ekf_error"], label="EKF error")
    plt.xlabel("features_drone")
    plt.ylabel("Error")
    plt.title("Features vs Measurement Error and EKF Error")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "features_vs_errors.png"))
    plt.close()

    # ----------------------------
    # 2) EKF error with & without measurements
    # ----------------------------
    # "Has measurements" = x_meas is not NaN (can also check y_meas if you want)
    has_meas = df["x_meas"].notna()
    no_meas = ~has_meas

    plt.figure()
    # Explicit colors here because you asked for different colors
    plt.scatter(
        df.loc[no_meas, "features_drone"],
        df.loc[no_meas, "ekf_error"],
        label="EKF error (no measurements)",
        alpha=0.8,
    )
    plt.scatter(
        df.loc[has_meas, "features_drone"],
        df.loc[has_meas, "ekf_error"],
        label="EKF error (with measurements)",
        alpha=0.8,
    )

    plt.xlabel("features_drone")
    plt.ylabel("EKF error")
    plt.title("EKF Error: With vs Without Measurements")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "ekf_with_without_measurements.png"))
    plt.close()

    # ----------------------------
    # 3) "Confusion matrix" (correlation heatmap)
    # ----------------------------
    pretty_names = {
        "features_drone": "Features",
        "total_matches": "Matches",
        "inliers": "Inliers",
        "LG_confidence": "LG Conf.",
        "median_reproj_error_px": "Reproj. Err.",
        "shape_score": "Shape Score",
        "overall_confidence": "Overall Conf.",
        "error": "Meas. Error",
        "ekf_error": "EKF Error",
    }

    cols = [
        "features_drone",
        "total_matches",
        "inliers",
        "LG_confidence",
        "median_reproj_error_px",
        "shape_score",
        "overall_confidence",
        "error",
        "ekf_error",
    ]

    # Ensure these columns exist
    cols = [c for c in cols if c in df.columns]

    # Convert to numeric in case anything weird slipped in
    corr_df = df[cols].apply(pd.to_numeric, errors="coerce")
    
    if "inliers" in df.columns and "median_reproj_error_px" in corr_df.columns:
        mask_low_inliers = df["inliers"] <= 80
        corr_df.loc[mask_low_inliers, "median_reproj_error_px"] = np.nan
        
    corr = corr_df.corr()

    fig, ax = plt.subplots(figsize=(8, 5))

    # Heatmap
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)

    # Show all ticks and label them
    ax.set_xticks(np.arange(len(cols)))
    ax.set_yticks(np.arange(len(cols)))
    ax.set_xticklabels([pretty_names[c] for c in cols], rotation=45, ha="right")
    ax.set_yticklabels([pretty_names[c] for c in cols])


    # Make cells square-ish
    ax.set_aspect("equal")

    # Annotate each cell with the correlation value
    for i in range(len(cols)):
        for j in range(len(cols)):
            value = corr.iloc[i, j]
            ax.text(
                j, i,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=8,
                color="black",
            )

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Correlation", rotation=270, labelpad=15)

    ax.set_title("Correlation Heatmap", pad=10)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "correlation_matrix.png"))
    plt.close()

    print(f"All plots saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
