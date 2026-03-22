import aerosandbox as asb
import aerosandbox.numpy as np
import csv
import os

LOG_PATH = "optimization.csv"

# Constants
N_SECTIONS = 5
SEMI_SPAN  = 0.3     # m
V_INF      = 20.0    # m/s
RHO        = 1.225   # kg/m³
W_TARGET   = 4.0     # N
CD0        = 0.01    # Parasitic drag coefficient
SM_TARGET  = 0.05    # Static margin (fraction of MAC)
CL_MAX     = 1.5     # NACA 4415 approximate maximum lift coefficient

# Score configuration
#   Each entry is (weight, reference_value)
#   weight > 0  → rewarded term  (score += w * metric / ref)
#   weight < 0  → penalised term (score += w * metric / ref, so sign flips)
#   weight = 0  → term disabled
SCORE_CONFIG = {
    "LD":     ( 1.0,     15.0  ),   # reward high L/D
    "AREA":   (-0.3,     0.10  ),   # penalise wetted area          [m²]
    "PITCH":  ( 0.0,     0.05  ),   # penalise Cm > 0 (nose-up)
    "TRIM":   (-0.4,     0.05  ),   # penalise |Cm|
    "SMOOTH": (-0.15,    1.0   ),   # penalise LE curvature         [1/m]
    "ROLL":   (-0.7,     0.05  ),   # penalise Clb > 0 (roll unstable) [1/rad]
    "YAW":    (-0.6,     0.05  ),   # penalise Cnb < 0 (yaw unstable)  [1/rad]
    "STALL":  (-0.5,     6.0   ),   # penalise high stall speed     [m/s]
    "BM":     (-0.4,     1.0   ),   # penalise root bending moment  [N·m]
}

def planform_curvature(x_le, y_le, z_le, chords, n_sections):
    n_interior = n_sections - 2
    dy_avg  = 0.5 * ((y_le[1:-1] - y_le[:-2]) + (y_le[2:] - y_le[1:-1]))
    denom   = dy_avg**2 + 1e-8
    d2x_le  = (x_le[2:]   - 2.0*x_le[1:-1]   + x_le[:-2])   / denom
    d2chord = (chords[2:] - 2.0*chords[1:-1] + chords[:-2]) / denom
    d2z     = (z_le[2:]   - 2.0*z_le[1:-1]   + z_le[:-2])   / denom
    return np.sqrt(np.sum(d2x_le**2 + d2chord**2 + d2z**2) / n_interior + 1e-8)

def calc_cg(y_le, x_le, chords, static_margin=SM_TARGET):
    y0, y1 = y_le[:-1], y_le[1:]
    x0, x1 = x_le[:-1], x_le[1:]
    c0, c1 = chords[:-1], chords[1:]
    dy = y1 - y0
    S = np.sum(0.5 * (c0 + c1) * dy)
    mac_integrand = dy / 3.0 * (c0**2 + c0 * c1 + c1**2)
    mac = np.sum(mac_integrand) / S
    a, b   = x0, x1 - x0       # LE position linear coefficients
    c_r, d = c0, c1 - c0       # chord linear coefficients
    xle_c_integrand = dy * (a * c_r + (a * d + b * c_r) / 2.0 + b * d / 3.0)
    x_ac = (np.sum(xle_c_integrand) + 0.25 * np.sum(mac_integrand)) / S
    return x_ac - static_margin * mac

def stall_speed(S_ref, cl, w=W_TARGET, rho=RHO):
    return np.sqrt((2.0 * w) / (rho * S_ref * cl))

def root_bending_moment(y_le, chords, CL, v_inf=V_INF, rho=RHO):
    q   = 0.5 * rho * v_inf**2
    y0, y1 = y_le[:-1], y_le[1:]
    c0, c1 = chords[:-1], chords[1:]
    dy  = y1 - y0
    # ∫ c(y)·y dy per panel (same closed form as y_mac numerator in calc_cg)
    cy_integrand = dy * (y0 * (c0 + c1) / 2.0 + dy * (c0 + 2.0 * c1) / 6.0)
    return q * CL * np.sum(cy_integrand)

def compute_score(
    CL, CD_total, Cm, S_ref, S_wet,
    x_le, y_le, z_le, chords, n_sections,
    Clb=0.0, Cnb=0.0,
    cfg=SCORE_CONFIG, verbose=False,
):
    LD     = CL / CD_total
    kappa  = planform_curvature(x_le, y_le, z_le, chords, n_sections)
    v_stall = stall_speed(S_ref, CL)
    M_root  = root_bending_moment(y_le, chords, CL)

    w_ld,     r_ld     = cfg["LD"]
    w_area,   r_area   = cfg["AREA"]
    w_pitch,  r_pitch  = cfg["PITCH"]
    w_trim,   r_trim   = cfg["TRIM"]
    w_smooth, r_smooth = cfg["SMOOTH"]
    w_roll,   r_roll   = cfg["ROLL"]
    w_yaw,    r_yaw    = cfg["YAW"]
    w_stall,  r_stall  = cfg["STALL"]
    w_bm,     r_bm     = cfg["BM"]

    t_ld     =  w_ld     * (LD               / r_ld)
    t_area   =  w_area   * (S_wet            / r_area)
    t_pitch  =  w_pitch  * (np.fmax(-Cm, 0.0) / r_pitch)   # penalise Cm > 0
    t_trim   =  w_trim   * (np.fabs(Cm)      / r_trim)
    t_smooth =  w_smooth * (kappa            / r_smooth)
    t_roll   =  w_roll   * (np.fmax(Clb, 0.0) / r_roll)    # penalise Clb > 0
    t_yaw    =  w_yaw    * (np.fmax(-Cnb, 0.0) / r_yaw)    # penalise Cnb < 0
    t_stall  =  w_stall  * (v_stall          / r_stall)     # penalise high V_s
    t_bm     =  w_bm     * (M_root           / r_bm)        # penalise high bending

    score = t_ld + t_area + t_pitch + t_trim + t_smooth + t_roll + t_yaw + t_stall + t_bm

    if verbose:
        print("\n══ SCORE BREAKDOWN ══════════════════════════════")
        print(f"  L/D term      (+) :  {float(t_ld):+.4f}   (L/D = {float(LD):.2f})")
        print(f"  Area term     (-) :  {float(t_area):+.4f}   (S_wet = {float(S_wet)*1e4:.0f} cm²)")
        print(f"  Pitch term    (-) :  {float(t_pitch):+.4f}   (max(Cm,0) = {max(float(Cm),0):.4f})")
        print(f"  Trim term     (-) :  {float(t_trim):+.4f}   (|Cm| = {abs(float(Cm)):.4f})")
        print(f"  Smooth term   (-) :  {float(t_smooth):+.4f}   (curvature = {float(kappa):.4f} /m)")
        print(f"  Roll stab     (-) :  {float(t_roll):+.4f}   (Clb = {float(Clb):.4f} /rad, want < 0)")
        print(f"  Yaw  stab     (-) :  {float(t_yaw):+.4f}   (Cnb = {float(Cnb):.4f} /rad, want > 0)")
        print(f"  Stall speed   (-) :  {float(t_stall):+.4f}   (V_stall = {float(v_stall):.2f} m/s)")
        print(f"  Bending mom.  (-) :  {float(t_bm):+.4f}   (M_root = {float(M_root):.3f} N·m)")
        print(f"  ══ SCORE          :  {float(score):+.4f}")

    return score

def report_best_design(log_path = LOG_PATH):
    # Find best row
    if not os.path.exists(log_path):
        print(f"[report_best_design] Log file not found: {log_path}")
        return
 
    best_row = None
    best_score = float("-inf")
 
    with open(log_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                s = float(row["score"])
            except (ValueError, KeyError):
                continue
            if s > best_score:
                best_score = s
                best_row = row
 
    if best_row is None:
        print("[report_best_design] No valid rows found in log.")
        return
 
    # Unpack geometry
    sol_chords  = np.array([float(best_row[f"chord{k}"]) / 100.0 for k in range(N_SECTIONS)])
    sol_twists  = np.array([float(best_row[f"twist{k}"])          for k in range(N_SECTIONS)])
    sol_y_le    = np.array([float(best_row[f"y{k}"])     / 100.0 for k in range(N_SECTIONS)])
    sol_x_le    = np.array([float(best_row[f"x{k}"])     / 100.0 for k in range(N_SECTIONS)])
    sol_z_le    = np.array([float(best_row[f"z{k}"])     / 100.0 for k in range(N_SECTIONS)])
 
    # Unpack aerodynamic quantities
    sol_CL       = float(best_row["CL"])
    sol_CD_total = float(best_row["CD_total"])   # already includes CD0
    sol_Cm       = float(best_row["Cm"])
    sol_Cm_CG    = float(best_row["Cm_CG"])
    sol_AR       = float(best_row["AR"])
    sol_L        = float(best_row["L_N"])
    sol_Clb      = float(best_row["Clb"])
    sol_Cnb      = float(best_row["Cnb"])
    sol_alpha    = float(best_row["alpha"])
 
    # Re-derive quantities
    sol_S_ref    = float(best_row["S_ref_cm2"]) / 1e4   # cm² → m²
    sol_S_wet    = 2.0 * sol_S_ref
    sol_V_stall  = float(stall_speed(sol_S_ref, sol_CL))
    sol_M_root   = float(root_bending_moment(sol_y_le, sol_chords, sol_CL))
    sol_cg       = calc_cg(sol_y_le, sol_x_le, sol_chords)   # scalar (m)
 
    # Print report
    print("\n═══ BEST DESIGN ══════════════════════════")
    print(f"  (score = {best_score:.4f})")
    print(f"  Alpha          : {sol_alpha:.2f} °")
    print(f"  CL             : {sol_CL:.4f}")
    print(f"  CD (total)     : {sol_CD_total:.4f}")
    print(f"  L/D            : {sol_CL / sol_CD_total:.3f}")
    print(f"  Cm             : {sol_Cm:.4f}")
    print(f"  Lift           : {sol_L:.3f} N  (required ≥ {W_TARGET} N)")
    print(f"  Wing area      : {sol_S_ref * 1e4:.1f} cm²")
    print(f"  Wetted area    : {sol_S_wet * 1e4:.1f} cm²")
    print(f"  Aspect ratio   : {sol_AR:.2f}")
    print(f"  Stall speed    : {sol_V_stall:.2f} m/s")
    print(f"  Root bending   : {sol_M_root:.3f} N·m")
    print(f"  Chords (cm)    : {np.round(sol_chords * 100, 1)}")
    print(f"  Twists (°)     : {np.round(sol_twists, 2)}")
    print(f"  Span (cm)      : {np.round(sol_y_le * 100, 1)}")
    print(f"  Sweep x (cm)   : {np.round(sol_x_le * 100, 1)}")
    print(f"  Dihedral (cm)  : {np.round(sol_z_le * 100, 1)}")
    print(f"  x_CG (mm)      : {float(sol_cg) * 100:.1f} mm from ref")
    print(f"  Static margin  : {SM_TARGET * 100:.1f}% MAC")
    print(f"  Cm about CG    : {sol_Cm_CG:.6f}  (should be ~0)")
    print(f"  Clb            : {sol_Clb:.4f} /rad  (want < 0, roll stable)")
    print(f"  Cnb            : {sol_Cnb:.4f} /rad  (want > 0, yaw  stable)")
    print("═════════════════════════════════════════════════")
 
    compute_score(
        sol_CL, sol_CD_total, sol_Cm_CG, sol_S_ref, sol_S_wet,
        sol_x_le, sol_y_le, sol_z_le, sol_chords, N_SECTIONS,
        Clb=sol_Clb, Cnb=sol_Cnb,
        verbose=True,
    )
 
    # Build plane
    sol_airplane = asb.Airplane(
        name    = "Best Wing",
        xyz_ref = [0.0, 0.0, 0.0],
        wings   = [
            asb.Wing(
                name      = "Main Wing",
                symmetric = True,
                xsecs     = [
                    asb.WingXSec(
                        xyz_le  = [sol_x_le[i], sol_y_le[i], sol_z_le[i]],
                        chord   = sol_chords[i],
                        twist   = sol_twists[i],
                        airfoil = asb.Airfoil("naca4415"),
                    )
                    for i in range(N_SECTIONS)
                ],
            )
        ],
    )
    sol_airplane.draw()

opti = asb.Opti()

y_le = opti.variable(
    init_guess  = np.linspace(0, SEMI_SPAN, N_SECTIONS),
    lower_bound = 0.0, upper_bound = SEMI_SPAN,
)
x_le = opti.variable(
    init_guess  = np.linspace(0.0, 0.03, N_SECTIONS),
    lower_bound = 0.0, upper_bound = 0.12,
)
z_le = opti.variable(
    init_guess  = np.zeros(N_SECTIONS),
    lower_bound = 0.0, upper_bound = 0.08,
)
chords = opti.variable(
    init_guess  = np.linspace(0.18, 0.08, N_SECTIONS),
    lower_bound = 0.07, upper_bound = 0.20,
)
twists = opti.variable(
    init_guess  = np.zeros(N_SECTIONS),
    lower_bound = -6.0, upper_bound = 6.0,
)
alpha = opti.variable(
    init_guess  = 4.0,
    lower_bound = 2.0,
    upper_bound = 8.0,
)

sections = [
    asb.WingXSec(
        xyz_le  = [x_le[i], y_le[i], z_le[i]],
        chord   = chords[i],
        twist   = twists[i],
        airfoil = asb.Airfoil("naca4415"),
    )
    for i in range(N_SECTIONS)
]

cg = calc_cg(y_le, x_le, chords)
airplane = asb.Airplane(
    name    = "Plane",
    xyz_ref = [cg, 0.0, 0.0],
    wings   = [asb.Wing(name="Main Wing", symmetric=True, xsecs=sections)],
)

op_point = asb.OperatingPoint(velocity=V_INF, alpha=alpha)
vlm = asb.VortexLatticeMethod(
    airplane             = airplane,
    op_point             = op_point,
    spanwise_resolution  = 3,
    chordwise_resolution = 5,
)
aero = vlm.run_with_stability_derivatives(alpha=True, beta=True)

Cm_CG = aero["Cm"]
Clb   = aero["Clb"]   # Negative = roll stable
Cnb   = aero["Cnb"]   # Positive = yaw  stable

# ── Derived quantities ────────────────────────────────────────────────────────
S_ref    = 2.0 * np.sum(0.5 * (chords[:-1] + chords[1:]) * np.diff(y_le))
b        = 2.0 * SEMI_SPAN
AR       = b**2 / S_ref
q        = 0.5 * RHO * V_INF**2
L        = q * S_ref * aero["CL"]
S_wet    = 2.0 * S_ref
CD_total = aero["CD"] + CD0

# ── Hard constraints ──────────────────────────────────────────────────────────
opti.subject_to(x_le[0] == 0.0)
opti.subject_to(y_le[0] == 0.0)
opti.subject_to(z_le[0] == 0.0)
opti.subject_to(y_le[-1] == SEMI_SPAN)
opti.subject_to(np.diff(y_le)   >= 0.03)    # Min panel width
opti.subject_to(np.diff(y_le) <= SEMI_SPAN * 0.45) # Max panel width
opti.subject_to(np.diff(chords) <= -0.004)  # Monotone taper
opti.subject_to(np.diff(x_le)   >= 0.0)    # Monotone sweep
opti.subject_to(np.diff(z_le)   >= 0.0)    # Monotone dihedral
opti.subject_to(np.diff(twists) <=  2.5)   # Twist smoothness
opti.subject_to(np.diff(twists) >= -2.5)
opti.subject_to(AR >= 4.0)
opti.subject_to(AR <= 10.0)
opti.subject_to(S_ref >= 0.02)             # m²
opti.subject_to(S_ref <= 0.20)             # m²
opti.subject_to(L >= W_TARGET)             # Lift ≥ 6 N
opti.subject_to(L <= 2.0 * W_TARGET)
opti.subject_to(aero["CL"] >= 0.3)        # Positive lift
CD_induced_min = aero["CL"]**2 / (np.pi * 12.0)
opti.subject_to(aero["CD"] >= CD_induced_min)
opti.subject_to(Cm_CG <= 0.02)
opti.subject_to(Cm_CG >= -0.05)

# ── Objective ─────────────────────────────────────────────────────────────────
LOG_FIELDS = [
    "score", "LD", "CL", "CD_total", "Cm", "Cm_CG",
    "S_ref_cm2", "AR", "L_N", "V_stall",
    "M_root", "Clb", "Cnb",
    "alpha",
    "chord0","chord1","chord2","chord3","chord4",
    "twist0","twist1","twist2","twist3","twist4",
    "y0","y1","y2","y3","y4",
    "x0","x1","x2","x3","x4",
    "z0","z1","z2","z3","z4",
]

# Open file and write header immediately so it exists even if solver crashes
_log_file   = open(LOG_PATH, "w", newline="")
_log_writer = csv.DictWriter(_log_file, fieldnames=LOG_FIELDS)
_log_writer.writeheader()
_log_file.flush()
_iter_count = [0]

def _r(val):
    try:
        return round(float(val), 4)
    except Exception:
        return float("nan")
 
def _log_iteration(i):
    def v(expr):
        try:
            return float(opti.debug.value(expr))
        except Exception:
            return float("nan")
 
    sol_CL_i     = v(aero["CL"])
    sol_CD_i     = v(aero["CD"]) + CD0
    sol_S_ref_i  = v(S_ref)
    sol_y_le_i   = [v(y_le[k]) for k in range(N_SECTIONS)]
    sol_chords_i = [v(chords[k]) for k in range(N_SECTIONS)]
 
    try:
        v_stall_i = float(stall_speed(sol_S_ref_i, sol_CL_i)) if sol_CL_i > 0 else float("nan")
    except Exception:
        v_stall_i = float("nan")
 
    try:
        m_root_i = float(root_bending_moment(
            np.array(sol_y_le_i), np.array(sol_chords_i), sol_CL_i
        )) if sol_CL_i > 0 else float("nan")
    except Exception:
        m_root_i = float("nan")
 
    try:
        score_i = -float(opti.debug.value(-score))
    except Exception:
        score_i = float("nan")
 
    row = {
        "score":     _r(score_i),
        "LD":        _r(sol_CL_i / sol_CD_i if sol_CD_i > 0 else float("nan")),
        "CL":        _r(sol_CL_i),
        "CD_total":  _r(sol_CD_i),
        "Cm":        _r(v(aero["Cm"])),
        "Cm_CG":     _r(v(Cm_CG)),
        "S_ref_cm2": _r(sol_S_ref_i * 1e4),
        "AR":        _r(v(AR)),
        "L_N":       _r(v(L)),
        "V_stall":   _r(v_stall_i),
        "M_root":    _r(m_root_i),
        "Clb":       _r(v(Clb)),
        "Cnb":       _r(v(Cnb)),
        "alpha":     _r(v(alpha)),
        **{f"chord{k}": _r(sol_chords_i[k] * 100) for k in range(N_SECTIONS)},
        **{f"twist{k}": _r(v(twists[k]))           for k in range(N_SECTIONS)},
        **{f"y{k}":     _r(sol_y_le_i[k] * 100)    for k in range(N_SECTIONS)},
        **{f"x{k}":     _r(v(x_le[k]) * 100)       for k in range(N_SECTIONS)},
        **{f"z{k}":     _r(v(z_le[k]) * 100)       for k in range(N_SECTIONS)},
    }
    _log_writer.writerow(row)
    _log_file.flush()
 
opti.callback(_log_iteration)

# ── Objective ─────────────────────────────────────────────────────────────────
score = compute_score(
    aero["CL"], CD_total, aero["Cm"], S_ref, S_wet,
    x_le, y_le, z_le, chords, N_SECTIONS,
    Clb=Clb, Cnb=Cnb,
)
opti.minimize(-score)

print("Starting optimization...")
try:
    sol = opti.solve(
    max_iter = 50000,
    options  = {
        "ipopt.tol":                        5e-4,
        "ipopt.constr_viol_tol":            1e-6,
        "ipopt.acceptable_tol":             5e-3,
        "ipopt.acceptable_dual_inf_tol":    1e2,
        "ipopt.acceptable_constr_viol_tol": 1e-6,
        "ipopt.acceptable_iter":            5,
        "ipopt.mu_strategy":                "adaptive",
        "ipopt.nlp_scaling_method":         "gradient-based",
        "ipopt.bound_push":                 1e-4,
        "ipopt.bound_frac":                 1e-4,
        "ipopt.hessian_approximation":      "limited-memory",  # avoid 440ms Hessian
        "ipopt.limited_memory_max_history": 20,
        }
    )
except RuntimeError:
    print("  Solver did not fully converge — using best point found.")
    sol = opti.debug

# ── Results ───────────────────────────────────────────────────────────────────
_log_file.close()
print(f"\nOptimization log written → {LOG_PATH}")

report_best_design()