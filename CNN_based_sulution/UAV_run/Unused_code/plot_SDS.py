import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.nonparametric.smoothers_lowess import lowess

# ------------------------------------------------------
# Load CSV (SDS is already a named header column)
# ------------------------------------------------------
csv_path = r"D:\P7_Drone_Geolocalization\CNN_based_sulution\outputs\03\results_03.csv"
df = pd.read_csv(csv_path)

# Ensure numeric
df["error"] = pd.to_numeric(df["error"], errors="coerce")
df["SDS"]   = pd.to_numeric(df["SDS"],   errors="coerce")

# Drop rows with missing values in these two columns
df = df[["SDS", "error"]].dropna()

# ------------------------------------------------------
# 1) Scatter + LOWESS + Pearson + Spearman
# ------------------------------------------------------
x = df["SDS"].values
y = df["error"].values

plt.figure(figsize=(6, 5))
plt.scatter(x, y, alpha=0.5, s=15)

# LOWESS smoothing
try:
    smooth = lowess(y, x, frac=0.3)
    plt.plot(smooth[:, 0], smooth[:, 1], linewidth=2)
except Exception:
    pass

# Correlations
pear = float(np.corrcoef(x, y)[0, 1]) if len(x) > 1 else np.nan
spear = float(pd.Series(x).corr(pd.Series(y), method="spearman")) if len(x) > 1 else np.nan

plt.xlabel("SDS")
plt.ylabel("error")
plt.title(f"error vs SDS\nPearson={pear:.3f}, Spearman={spear:.3f}")
plt.tight_layout()
plt.show()

# ------------------------------------------------------
# 2) “Confusion matrix” between SDS and error (2x2)
#    - Split SDS and error at their medians
# ------------------------------------------------------
sds_thr   = df["SDS"].median()
error_thr = df["error"].median()

# Discretize into low / high
sds_cat = np.where(df["SDS"]   <= sds_thr,   "SDS_low",   "SDS_high")
err_cat = np.where(df["error"] <= error_thr, "err_low",   "err_high")

conf_df = pd.crosstab(err_cat, sds_cat)

print("2x2 confusion table (rows = error, cols = SDS):")
print(conf_df)

# Plot as heatmap
plt.figure(figsize=(4, 4))
sns.heatmap(conf_df, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion between SDS and error\n(median-based bins)")
plt.ylabel("error category")
plt.xlabel("SDS category")
plt.tight_layout()
plt.show()
