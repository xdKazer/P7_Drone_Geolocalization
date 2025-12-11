import pandas as pd
import numpy as np

# Load CSV
df = pd.read_csv("geolocalization_dinov3/dataset_data/logs/VPAIR_results_KH.csv")

# Convert columns to numeric (in case empty fields are interpreted as strings)
df["error_meters"] = pd.to_numeric(df["error_meters"], errors="coerce")
df["measurement_error"] = pd.to_numeric(df["measurement_error"], errors="coerce")
df["processing_time_sec"] = pd.to_numeric(df["processing_time_sec"], errors="coerce")

# ---- Successful localizations ----
total_images = len(df)
successful = df["measurement_available"].astype(bool).sum()
success_ratio = successful / total_images

# ---- Mean Error ----
mean_error = df["error_meters"].mean()

mean_measurement_error = df["measurement_error"].mean()

# ---- RMSE ----
rmse = np.sqrt(np.nanmean(df["error_meters"] ** 2))

# ---- Standard deviation of error ----
std_error = df["error_meters"].std()

# ---- Time statistics ----
mean_time = df["processing_time_sec"].mean()
std_time = df["processing_time_sec"].std()
total_time = df["processing_time_sec"].sum()

# ---- Print results ----
print("Total images:", total_images)
print("Successful localisations:", successful)
print("Success ratio:", success_ratio)

print("Mean Error (m):", mean_error)
print("Mean Measurement Error (m):", mean_measurement_error)
print("RMSE (m):", rmse)
print("STD Error (m):", std_error)

print("Mean Processing Time (s):", mean_time)
print("STD Processing Time (s):", std_time)
print("Total Processing Time (s):", total_time)