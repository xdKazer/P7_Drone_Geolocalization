import pandas as pd

# Load your CSV, letting pandas read the header
dataset_number = "11"  # e.g., "01", "02", "03", ...
df = pd.read_csv(f"C:\\Users\\signe\\P7_Drone_Geolocalization\\CNN_based_sulution\\UAV_run\\UAV_VisLoc_dataset\\{dataset_number}\\{dataset_number}.csv")

# Convert the 'date' column to datetime
df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%dT%H:%M:%S")

# Compute time delta in seconds
df["dt"] = df["date"].diff().dt.total_seconds()

# Set threshold (seconds)
THRESHOLD = 10

# Find rows with large time gaps
large_time_gaps = df[df["dt"] > THRESHOLD]

print("Large time gaps:")
print(large_time_gaps[["filename", "date", "dt"]])