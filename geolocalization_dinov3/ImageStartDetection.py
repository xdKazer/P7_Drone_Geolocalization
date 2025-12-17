import pandas as pd

# Load your CSV, letting pandas read the header
df = pd.read_csv("geolocalization_dinov3/dataset_data/csv_files/01.csv")

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