import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.nonparametric.smoothers_lowess import lowess
import math

# ------------------------------------------------------
# Load CSV
# ------------------------------------------------------
df = pd.read_csv(
    r"D:\P7_Drone_Geolocalization\CNN_based_sulution\outputs\03\results_03.csv"
)

df["error"] = pd.to_numeric(df["error"], errors="coerce")
df["avg_confidence"] = pd.to_numeric(df["avg_confidence"], errors="coerce")
df["overall_confidence"] = pd.to_numeric(df["overall_confidence"], errors="coerce")


df["median_reproj_error_px"] = pd.to_numeric(
    df["median_reproj_error_px"], errors="coerce"
)

# ------------------------------------------------------
# Parse shape_terms "(a,b,c,d)" → 4 columns
# ------------------------------------------------------
def parse_shape_terms(s):
    if isinstance(s, str):
        try:
            out = list(ast.literal_eval(s))
            if len(out) == 4:          
                return out
        except Exception:
            pass
    return [np.nan] * 4

shape_cols = [
    "shape_sides",          # s_sides
    "shape_aspect_ratio",   # s_aspect_ratio
    "shape_angle",          # s_angle
    "shape_scale",          # s_scale
]

shape_df = df["shape_terms"].apply(parse_shape_terms).apply(pd.Series)
shape_df.columns = shape_cols
df = pd.concat([df, shape_df], axis=1)

# ------------------------------------------------------
# Helper: Scatter + LOWESS + Pearson + Spearman
# ------------------------------------------------------
def plot_relationship_ax(df, xcol, ycol, ax, frac=0.3):
    # Drop NaNs for this pair
    mask = df[[xcol, ycol]].dropna()
    x = mask[xcol].values
    y = mask[ycol].values

    if len(x) == 0:
        ax.set_title(f"{ycol} vs {xcol}\n(no data)")
        ax.set_xlabel(xcol)
        ax.set_ylabel(ycol)
        return

    # Scatter
    ax.scatter(x, y, alpha=0.5, s=10)

    # LOWESS smoothing
    try:
        smooth = lowess(y, x, frac=frac)
        ax.plot(smooth[:, 0], smooth[:, 1], linewidth=2)
    except Exception:
        pass

    # Correlations
    if len(x) > 1:
        pear = float(np.corrcoef(x, y)[0, 1])
        spear = float(pd.Series(x).corr(pd.Series(y), method="spearman"))
        title = f"{ycol} vs {xcol}\nPearson={pear:.3f}, Spearman={spear:.3f}"
    else:
        title = f"{ycol} vs {xcol}\n(not enough points)"

    ax.set_xlabel(xcol)
    ax.set_ylabel(ycol)
    ax.set_title(title)

# ------------------------------------------------------
# 1) Combined quality (simple mean over shape + confidences)
# ------------------------------------------------------
quality_cols = shape_cols + ["avg_confidence", "overall_confidence"]
df["combined_quality"] = df[quality_cols].mean(axis=1)

# ------------------------------------------------------
# 2) Plot all relationships in a single multi-panel figure
# ------------------------------------------------------
x_columns_to_plot = (
    shape_cols
    + ["median_reproj_error_px",  
       "overall_confidence",
       "avg_confidence",
       "combined_quality"]
)

n_plots = len(x_columns_to_plot)
n_cols = 3
n_rows = math.ceil(n_plots / n_cols)

fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4.5 * n_rows))
axes = np.array(axes).reshape(n_rows, n_cols)

ycol = "error"

for idx, xcol in enumerate(x_columns_to_plot):
    r = idx // n_cols
    c = idx % n_cols
    ax = axes[r, c]
    plot_relationship_ax(df, xcol, ycol, ax)

# Turn off any unused subplots
for idx in range(n_plots, n_rows * n_cols):
    r = idx // n_cols
    c = idx % n_cols
    axes[r, c].axis("off")

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.subplots_adjust(hspace=0.35, wspace=0.25)   
plt.show()

# ------------------------------------------------------
# 3) Correlation heatmap
# ------------------------------------------------------
corr = df[
    [ycol,
     "LG_confidence",
     "overall_confidence",
     "median_reproj_error_px"]  
    + shape_cols
].corr()

plt.figure(figsize=(8, 5))
sns.heatmap(corr, annot=True, cmap="coolwarm", center=0, fmt=".2f")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()
