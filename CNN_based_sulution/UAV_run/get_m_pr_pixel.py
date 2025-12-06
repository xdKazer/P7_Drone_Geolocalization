import cv2
import numpy as np
import csv
import math
import matplotlib.pyplot as plt  # NEW

# ----------------- CONFIG -----------------
sat_number = "03"  # "01" | "02" | "03" | ...
SAT_IMG_PATH   = rf"C:/Users/signe/P7_Drone_Geolocalization/CNN_based_sulution/UAV_run/UAV_VisLoc_dataset/{sat_number}/satellite{sat_number}.tif"
DRONE_IMG_PATH = rf"C:/Users/signe/P7_Drone_Geolocalization/CNN_based_sulution/UAV_run/UAV_VisLoc_dataset/{sat_number}/drone/{sat_number}_0010.JPG"
# For 01: use 01_0089.JPG, for 02: use 02_0022.JPG, use 03_0010.JPG for 03,
"""
=== RESULTS 01 ===
Satellite pixel distance: 110.041 px
Drone pixel distance:     338.740 px
Satellite GSD:            0.277868 m/px
Object real length:       30.577 m
Drone GSD:                0.090266 m/px
Drone/Sat GSD ratio:      0.325 ( >1 = coarser, <1 = finer )

=== RESULTS 02===
Satellite pixel distance: 296.378 px
Drone pixel distance:     897.813 px
Satellite GSD:            0.277803 m/px
Object real length:       82.335 m
Drone GSD:                0.091706 m/px
Drone/Sat GSD ratio:      0.330 ( >1 = coarser, <1 = finer )

=== RESULTS 03 ===
Satellite pixel distance: 99.126 px
Drone pixel distance:     260.461 px
Satellite GSD:            0.274126 m/px
Object real length:       27.173 m
Drone GSD:                0.104327 m/px
Drone/Sat GSD ratio:      0.381 ( >1 = coarser, <1 = finer )
"""


def haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dl   = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))

# ----------------- HELPERS -----------------
def distance_pixels(p1, p2):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    return math.sqrt(dx * dx + dy * dy)


def measure_in_image(img, window_name="Image"):
    """
    Interactive measurement using matplotlib:

      - Use mouse wheel / touchpad to zoom.
      - Use toolbar to pan/zoom if you want.
      - Left-click two points on the same object.
      - Returns (p1, p2, pixel_distance) in ORIGINAL image coordinates.
    """
    # Prepare figure
    fig, ax = plt.subplots(num=window_name)

    # BGR (OpenCV) -> RGB (matplotlib)
    if img.ndim == 2:
        ax.imshow(img, cmap="gray")
    else:
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    ax.set_title(f"{window_name}: zoom/pan, then LEFT-CLICK two points")
    points = []  # will hold (x_float, y_float) in image coords

    print(f"\n=== {window_name} ===")
    print("Instructions:")
    print("  - Use mouse wheel / touchpad to zoom.")
    print("  - Use right-click + drag or toolbar to pan/zoom.")
    print("  - LEFT-CLICK two points on the same object.")
    print("  - Close the window or hit the X after second click.\n")

    def onclick(event):
        nonlocal points

        # Only accept clicks inside the axes, left button
        if event.inaxes != ax:
            return
        if event.button != 1:  # 1 = left click
            return
        if event.xdata is None or event.ydata is None:
            return

        x = float(event.xdata)
        y = float(event.ydata)
        points.append((x, y))
        print(f"[{window_name}] Clicked point {len(points)} at (x={x:.2f}, y={y:.2f})")

        # draw marker
        ax.plot(x, y, "rx")
        fig.canvas.draw_idle()

        if len(points) == 2:
            # draw line
            xs = [points[0][0], points[1][0]]
            ys = [points[0][1], points[1][1]]
            ax.plot(xs, ys, "g-")
            fig.canvas.draw_idle()
            # auto-close window after second point
            plt.close(fig)

    cid = fig.canvas.mpl_connect("button_press_event", onclick)

    plt.show()  # this blocks until window is closed

    if len(points) < 2:
        print(f"[{window_name}] Not enough points selected (need 2).")
        return None, None, float("nan")

    # Convert from float (data coords) to integer pixel indices
    p1 = (int(round(points[0][0])), int(round(points[0][1])))
    p2 = (int(round(points[1][0])), int(round(points[1][1])))
    pix_dist = distance_pixels(p1, p2)

    print(f"[{window_name}] Final pixel distance: {pix_dist:.2f} px")
    return p1, p2, pix_dist


# ----------------- MAIN LOGIC -----------------
def main():
    # ---------- Load SAT image ----------
    sat = cv2.imread(SAT_IMG_PATH, cv2.IMREAD_UNCHANGED)
    if sat is None:
        print(f"Failed to load satellite image: {SAT_IMG_PATH}")
        return

    H_sat, W_sat = sat.shape[:2]
    print(f"Full satellite shape: {H_sat} x {W_sat}")

    # ---------- Crop SAT image region you care about ----------
    CROP_region = 1000
    w = 30000
    h =17000
    # set crop region here:
    y0, y1 =h, h + CROP_region
    x0, x1 =  w, w + CROP_region

    """ (01) :  y0, y1 =23000, 23000 + CROP_region
    x0, x1 =  2100, 2100 + CROP_region"""
    """ (02) :     CROP_region = 500
    w = 1900
    h =12240
    """
    """ (03) : CROP_region = 1000
    w = 30000
    h =17000 """
    """ (04) : """
    """ (05) : """
    """ (06) : """
    """ (07) : """
    """ (08) : """
    """ (09) : """
    """ (10) : """
    """ (11) : """

    # clamp crop to image bounds
    y0 = max(0, min(H_sat, y0))
    y1 = max(0, min(H_sat, y1))
    x0 = max(0, min(W_sat, x0))
    x1 = max(0, min(W_sat, x1))

    if y1 <= y0 or x1 <= x0:
        print(f"Invalid crop region after clamping: y0={y0}, y1={y1}, x0={x0}, x1={x1}")
        return

    sat_img = sat[y0:y1, x0:x1].copy()
    H_crop, W_crop = sat_img.shape[:2]
    print(f"Cropped satellite shape: {H_crop} x {W_crop} (y: {y0}-{y1}, x: {x0}-{x1})")

    # ---------- Read map extent and compute SAT GSD ----------
    coordinate_range_lat_lon_sat = None
    with open(
        "C:/Users/signe/P7_Drone_Geolocalization/CNN_based_sulution/UAV_run/UAV_VisLoc_dataset/satellite_coordinates_range.csv",
        newline="",
    ) as f:
        for r in csv.DictReader(f):
            if r["mapname"] == f"satellite{sat_number}.tif":
                LT_lat = np.float64(r["LT_lat_map"])
                LT_lon = np.float64(r["LT_lon_map"])
                RB_lat = np.float64(r["RB_lat_map"])
                RB_lon = np.float64(r["RB_lon_map"])
                coordinate_range_lat_lon_sat = [
                    (LT_lat, LT_lon),  # Left Top, (lat, lon)
                    (RB_lat, RB_lon),  # Right Bottom, (lat, lon)
                ]
                break

    if coordinate_range_lat_lon_sat is None:
        print(f"No coordinate range found for satellite{sat_number}.tif")
        return

    (lat_min, lon_min), (lat_max, lon_max) = coordinate_range_lat_lon_sat

    sat_height_m = haversine_m(lat_min, lon_min, lat_max, lon_min)
    sat_width_m  = haversine_m(lat_min, lon_min, lat_min, lon_max)

    sat_m_px_h = sat_height_m / H_sat  # use FULL sat size
    sat_m_px_w = sat_width_m  / W_sat
    SAT_M_PER_PX = math.sqrt(sat_m_px_h * sat_m_px_w)
    print(f"Satellite GSD: {SAT_M_PER_PX:.6f} m/px")

    # ---------- Load DRONE image ----------
    drone_img = cv2.imread(DRONE_IMG_PATH, cv2.IMREAD_UNCHANGED)
    if drone_img is None:
        print(f"Failed to load drone image: {DRONE_IMG_PATH}")
        return
    print(f"Drone shape: {drone_img.shape[0]} x {drone_img.shape[1]}")

    # ---------- Measure in SATELLITE ----------
    sat_p1, sat_p2, sat_px_dist = measure_in_image(sat_img, window_name="Satellite")
    if math.isnan(sat_px_dist):
        print("No valid measurement in satellite image.")
        return

    # ---------- Measure in DRONE ----------
    drone_p1, drone_p2, drone_px_dist = measure_in_image(drone_img, window_name="Drone")
    if math.isnan(drone_px_dist):
        print("No valid measurement in drone image.")
        return

    # ---------- Compute GSD ----------
    real_len_m = sat_px_dist * SAT_M_PER_PX
    drone_m_per_px = real_len_m / drone_px_dist
    gsd_ratio = drone_m_per_px / SAT_M_PER_PX

    print("\n=== RESULTS ===")
    print(f"Satellite pixel distance: {sat_px_dist:.3f} px")
    print(f"Drone pixel distance:     {drone_px_dist:.3f} px")
    print(f"Satellite GSD:            {SAT_M_PER_PX:.6f} m/px")
    print(f"Object real length:       {real_len_m:.3f} m")
    print(f"Drone GSD:                {drone_m_per_px:.6f} m/px")
    print(f"Drone/Sat GSD ratio:      {gsd_ratio:.3f} ( >1 = coarser, <1 = finer )")


if __name__ == "__main__":
    main()
