import numpy as np

def _cum_arclen(points: np.ndarray) -> np.ndarray:
    """Return cumulative arc-length s[i] along polyline points."""
    d = np.linalg.norm(np.diff(points, axis=0), axis=1)
    s = np.zeros(points.shape[0], dtype=float)
    s[1:] = np.cumsum(d)
    return s

def _interp_along_s(points: np.ndarray, s: np.ndarray, s_query: np.ndarray) -> np.ndarray:
    """
    Piecewise-linear interpolation of polyline points by arc-length parameter.
    points: (M,3), s: (M,), s_query: (K,)
    return: (K,3)
    """
    s_query = np.clip(s_query, s[0], s[-1])
    idx = np.searchsorted(s, s_query, side="right") - 1
    idx = np.clip(idx, 0, len(s) - 2)

    s0, s1 = s[idx], s[idx + 1]
    t = (s_query - s0) / np.maximum(s1 - s0, 1e-12)
    p0, p1 = points[idx], points[idx + 1]
    return (1 - t)[:, None] * p0 + t[:, None] * p1

def _tangents(points: np.ndarray) -> np.ndarray:
    """Unit tangents for a polyline (central differences)."""
    dp = np.zeros_like(points)
    dp[1:-1] = points[2:] - points[:-2]
    dp[0] = points[1] - points[0]
    dp[-1] = points[-1] - points[-2]
    n = np.linalg.norm(dp, axis=1, keepdims=True)
    return dp / np.maximum(n, 1e-12)

def _parallel_transport_frames(curve: np.ndarray, t: np.ndarray, up: np.ndarray = None):
    """
    Build stable frames (T, N, B) along curve using parallel transport.
    Returns:
      T: (M,3), N: (M,3), B: (M,3)
    """
    M = curve.shape[0]
    T = t

    if up is None:
        up = np.array([0.0, 0.0, 1.0], dtype=float)
    up = up / (np.linalg.norm(up) + 1e-12)

    # pick initial normal not parallel to T0
    T0 = T[0]
    if abs(np.dot(T0, up)) > 0.95:
        up = np.array([0.0, 1.0, 0.0], dtype=float)

    N = np.zeros_like(curve)
    B = np.zeros_like(curve)

    # initial N0
    N0 = up - np.dot(up, T0) * T0
    N0 = N0 / (np.linalg.norm(N0) + 1e-12)
    B0 = np.cross(T0, N0)
    B0 = B0 / (np.linalg.norm(B0) + 1e-12)

    N[0], B[0] = N0, B0

    for i in range(1, M):
        Ti_1 = T[i - 1]
        Ti = T[i]

        # rotation that takes Ti_1 to Ti around axis = cross(Ti_1, Ti)
        v = np.cross(Ti_1, Ti)
        sv = np.linalg.norm(v)
        cv = np.dot(Ti_1, Ti)

        if sv < 1e-12:
            # almost same tangent: keep N,B
            N[i] = N[i - 1]
            B[i] = B[i - 1]
            continue

        axis = v / sv
        angle = np.arctan2(sv, np.clip(cv, -1.0, 1.0))

        # Rodrigues rotate previous N and B
        def rodrigues(vec):
            return (vec * np.cos(angle) +
                    np.cross(axis, vec) * np.sin(angle) +
                    axis * np.dot(axis, vec) * (1 - np.cos(angle)))

        Ni = rodrigues(N[i - 1])
        # enforce orthonormality
        Ni = Ni - np.dot(Ni, Ti) * Ti
        Ni = Ni / (np.linalg.norm(Ni) + 1e-12)
        Bi = np.cross(Ti, Ni)
        Bi = Bi / (np.linalg.norm(Bi) + 1e-12)

        N[i], B[i] = Ni, Bi

    return T, N, B

def map_stent_to_curve(
    stent: np.ndarray,
    curve: np.ndarray,
    z0: float = None,
    z1: float = None,
    up: np.ndarray = None,
):
    """
    Sweep/warp a cylindrical stent point cloud (initially around z-axis) onto a spatial curve.

    stent: (N,3) ndarray, originally centered on z-axis (centerline x=y=0)
    curve: (M,3) ndarray, same arc length as stent centerline length

    z0,z1: stent centerline z-range; if None, inferred from stent[:,2].min/max
    up: initial "up" direction to set initial frame twist (optional)

    return:
      stent_warped: (N,3)
    """
    stent = np.asarray(stent, dtype=float)
    curve = np.asarray(curve, dtype=float)
    if stent.ndim != 2 or stent.shape[1] != 3:
        raise ValueError("stent must be (N,3)")
    if curve.ndim != 2 or curve.shape[1] != 3 or curve.shape[0] < 2:
        raise ValueError("curve must be (M,3), M>=2")

    if z0 is None:
        z0 = float(np.min(stent[:, 2]))
    if z1 is None:
        z1 = float(np.max(stent[:, 2]))
    L_stent = z1 - z0
    if L_stent <= 0:
        raise ValueError("Invalid stent z-range (z1 must be > z0)")

    # curve arc-length parameter
    s_curve = _cum_arclen(curve)
    L_curve = s_curve[-1]
    if L_curve <= 0:
        raise ValueError("Curve length is zero")

    # If lengths are "same" but numerically a bit off, we just scale s mapping.
    scale = L_curve / L_stent

    # Build frames along curve
    T = _tangents(curve)
    T, N, B = _parallel_transport_frames(curve, T, up=up)

    # We'll need N(s), B(s), C(s). Interpolate by arc-length.
    # For vectors, linear interp then re-orthonormalize with tangent at query.
    z = stent[:, 2]
    s_query = (z - z0) * scale  # map stent axial coordinate to curve arclength

    Cq = _interp_along_s(curve, s_curve, s_query)
    Tq = _interp_along_s(T, s_curve, s_query)
    Nq = _interp_along_s(N, s_curve, s_query)
    Bq = _interp_along_s(B, s_curve, s_query)

    # normalize & re-orthonormalize
    Tq = Tq / np.maximum(np.linalg.norm(Tq, axis=1, keepdims=True), 1e-12)
    # make Nq orthogonal to Tq
    Nq = Nq - (np.sum(Nq * Tq, axis=1, keepdims=True)) * Tq
    Nq = Nq / np.maximum(np.linalg.norm(Nq, axis=1, keepdims=True), 1e-12)
    # Bq = T x N
    Bq = np.cross(Tq, Nq)
    Bq = Bq / np.maximum(np.linalg.norm(Bq, axis=1, keepdims=True), 1e-12)

    # local cross-section coordinates in initial stent frame
    # centerline is z-axis => offset is (x,y) in XY plane
    x = stent[:, 0]
    y = stent[:, 1]

    # warp
    stent_warped = Cq + x[:, None] * Nq + y[:, None] * Bq
    return stent_warped
