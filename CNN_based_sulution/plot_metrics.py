import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.nonparametric.smoothers_lowess import lowess

# ------------------------------------------------------
# Load CSV
# ------------------------------------------------------
df = pd.read_csv("D:\\P7_Drone_Geolocalization\\CNN_based_sulution\\outputs\\03\\results_03.csv")

df["error"] = pd.to_numeric(df["error"], errors="coerce")
df["avg_confidence"] = pd.to_numeric(df["avg_confidence"], errors="coerce")
df["overall_confidence"] = pd.to_numeric(df["overall_confidence"], errors="coerce")

# ------------------------------------------------------
# Parse shape_terms "(a,b,c,d,e)"
# ------------------------------------------------------
def parse_shape_terms(s):
    if isinstance(s, str):
        try:
            out = list(ast.literal_eval(s))
            if len(out) == 5:
                return out
        except:
            pass
    return [np.nan]*5

shape_cols = ["shape_rect","shape_pair","shape_ar","shape_scale","shape_inside"]
shape_df = df["shape_terms"].apply(parse_shape_terms).apply(pd.Series)
shape_df.columns = shape_cols
df = pd.concat([df, shape_df], axis=1)

# ------------------------------------------------------
# Scatter + LOWESS + Pearson + Spearman
# ------------------------------------------------------
def plot_relationship(df, xcol, ycol="error"):
    mask = df[[xcol, ycol]].dropna()
    x = mask[xcol].values
    y = mask[ycol].values
    
    plt.figure(figsize=(6,5))
    plt.scatter(x, y, alpha=0.5)
    
    # LOWESS smooth
    smooth = lowess(y, x, frac=0.3)
    plt.plot(smooth[:,0], smooth[:,1], linewidth=3)
    
    # correlations
    pear = np.corrcoef(x, y)[0,1]
    spear = pd.Series(x).corr(pd.Series(y), method="spearman")
    
    plt.xlabel(xcol)
    plt.ylabel(ycol)
    plt.title(f"{ycol} vs {xcol}\nPearson={pear:.3f}, Spearman={spear:.3f}")
    plt.tight_layout()
    plt.show()


# ------------------------------------------------------
# 1) Error vs shape terms
# ------------------------------------------------------
for col in shape_cols:
    plot_relationship(df, col, "error")

# ------------------------------------------------------
# 2) Error vs overall / avg conf
# ------------------------------------------------------
plot_relationship(df, "overall_confidence", "error")
plot_relationship(df, "avg_confidence", "error")

# ------------------------------------------------------
# 3) Correlation heatmap
# ------------------------------------------------------
corr = df[["error","avg_confidence","overall_confidence"] + shape_cols].corr()
plt.figure(figsize=(10,6))
sns.heatmap(corr, annot=True, cmap="coolwarm", center=0)
plt.title("Correlation Heatmap")
plt.show()

# ------------------------------------------------------
# 4) Build a single "quality predictor"
# ------------------------------------------------------
quality_cols = shape_cols + ["avg_confidence", "overall_confidence"]
df["combined_quality"] = df[quality_cols].mean(axis=1)

plot_relationship(df, "combined_quality", "error")
