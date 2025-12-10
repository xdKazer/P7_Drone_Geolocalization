import csv
import math

def haversine(lat1, lon1, lat2, lon2):
    """Return distance in km between two lat/lon points using the Haversine formula."""
    R = 6371.0  # Earth radius in km
    
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    
    return R * c

def compute_total_distance(csv_file):
    coords = []

    # Read CSV file
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            coords.append((float(row['lat']), float(row['lon'])))

    # Compute cumulative distance
    total_distance = 0.0
    for i in range(1, len(coords)):
        total_distance += haversine(*coords[i-1], *coords[i])

    return total_distance

if __name__ == "__main__":
    file_path = "geolocalization_dinov3/dataset_data/csv_files/11.csv"
    distance_km = compute_total_distance(file_path)
    print(f"Total distance flown: {distance_km:.3f} km")
