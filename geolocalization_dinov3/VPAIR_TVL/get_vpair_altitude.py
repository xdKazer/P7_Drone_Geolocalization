import csv

def get_min_max_altitude(csv_path):
    altitudes = []

    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Convert altitude string to float
            altitudes.append(float(row['altitude']))

    return min(altitudes), max(altitudes)

# Example usage:
csv_file = "CNN_based_sulution/VPAIR_run/vpair_dataset/poses_drone.csv"
min_alt, max_alt = get_min_max_altitude(csv_file)
print("Min altitude:", min_alt)
print("Max altitude:", max_alt)
