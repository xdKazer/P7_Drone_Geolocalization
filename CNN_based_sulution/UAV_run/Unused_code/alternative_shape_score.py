def get_shape_score(
    corners,
    img_w,
    img_h,
    scale_drone_to_tile,   # drone_px per tile_px
    tau_side_l_pairs=0.15,          # tolerance for opposite side length consistency ≈0.01 score when one side is ~2x the other, More tolerant → increase τ (e.g. 0.18–0.20), More strict → decrease τ (e.g. 0.12–0.13)
    tau_aspect_ratio=0.30,            # tolerance for aspect ratio
    tau_angle=15.0,        # degrees tolerance for angles 
    tau_scale=0.5,     # how fast we penalize scale error  set loose as meters pr pixel is estimated not known exactly
):
    """
    Shape score in [0,1] using only the 4 warped corners (in tile coords).

    Checks:
      - s_w, s_h: opposite side length consistency (rectangularity)
      - s_aspect_ratio: aspect ratio vs original drone W/H
      - s_angle: all 4 angles ~ 90°
      - s_scale_abs: absolute scale vs expected (with dead-band)

    Combination:
        score = 0.6 * min(terms) + 0.4 * mean(terms)
    (Strict wrt the worst term, but not as brittle as pure min.)
    """

    # ---------- Validate input ----------
    if corners is None:
        return 0.0, np.full(5, np.nan, dtype=np.float64)

    corners = np.asarray(corners, dtype=np.float64)
    if corners.shape != (4, 2) or not np.isfinite(corners).all():
        return 0.0, np.full(5, np.nan, dtype=np.float64)

    # =====================================================================
    # 0) Compute side vectors & lengths from corners
    #     Order: [top-left, top-right, bottom-right, bottom-left]
    # =====================================================================
    vT = corners[1] - corners[0]  # top
    vR = corners[2] - corners[1]  # right
    vB = corners[3] - corners[2]  # bottom
    vL = corners[0] - corners[3]  # left

    lT = np.linalg.norm(vT)
    lR = np.linalg.norm(vR)
    lB = np.linalg.norm(vB)
    lL = np.linalg.norm(vL)

    sides = np.array([lT, lR, lB, lL], dtype=np.float64)
    if np.any(sides < 1e-3) or not np.isfinite(sides).all():
        return 0.0

    # =====================================================================
    # 1) OPPOSITE SIDE EQUALITY  twice in size -> 0 conf  |  same -> 1 conf
    # =====================================================================
    d_w = abs(lT - lB) / (lT + lB + 1e-6)
    d_h = abs(lR - lL) / (lR + lL + 1e-6)

    s_w = float(np.exp(- (d_w / tau_side_l_pairs) ** 2))
    s_h = float(np.exp(- (d_h / tau_side_l_pairs) ** 2))
    s_sides = (s_w + s_h) / 2.0

    # =====================================================================
    # 2) ASPECT RATIO  PICKED SO THAT 50 distortion -> 0.3 conf  |  0 -> 1 conf  This is softly set as angle etc can be hard on it
    # =====================================================================
    w_est = 0.5 * (lT + lB)
    h_est = 0.5 * (lR + lL)

    r_est = w_est / (h_est + 1e-6)
    r0    = img_w / float(img_h)

    ratio_err = np.log(r_est / (r0 + 1e-6))
    s_aspect_ratio = float(np.exp(- (ratio_err / tau_aspect_ratio) ** 2))

    # =====================================================================
    # 3) ANGLES  90° -> 1 conf  |  +-30° -> 0 conf
    # =====================================================================
    def angle(a, b, c):
        """Angle ABC in degrees, with B as corner."""
        BA = a - b
        BC = c - b
        den = (np.linalg.norm(BA) * np.linalg.norm(BC) + 1e-9)
        cosang = np.dot(BA, BC) / den
        cosang = np.clip(cosang, -1.0, 1.0)
        return np.degrees(np.arccos(cosang))

    ang = []
    for i in range(4):
        a = corners[i - 1]
        b = corners[i]
        c = corners[(i + 1) % 4]
        ang.append(angle(a, b, c))
    ang = np.asarray(ang, dtype=np.float64)

    ang_err = np.abs(ang - 90.0)  # want ~90° at all 4 corners
    rms_ang_err = float(np.sqrt(np.mean(ang_err**2)))

    s_angle = float(np.exp(- (rms_ang_err / tau_angle) ** 2))

    # =====================================================================
    # 4) ABSOLUTE SCALE 
    # =====================================================================
    area_now = w_est * h_est      # warped rect area in tile px²
    area0    = float(img_w * img_h)  # original drone rect area in drone px²
    if area_now <= 0 or area0 <= 0:
        return 0.0

    S = float(scale_drone_to_tile)     # drone_px per tile_px
    # expected: scale_now ~ 1
    scale_now = S * math.sqrt(area_now / area0)
    scale_err = abs(scale_now - 1.0)

    s_scale = float(np.exp(- (scale_err / tau_scale) ** 2))

    # =====================================================================
    # 5) Combine terms (strict wrt worst)
    # =====================================================================
    terms = np.array([s_sides, s_aspect_ratio, s_angle, s_scale], dtype=np.float64)

    mean_t = float(terms.mean())
    min_t  = float(terms.min())

    shape_score = 0.6 * min_t + 0.4 * mean_t
    return float(np.clip(shape_score, 0.0, 1.0)), terms