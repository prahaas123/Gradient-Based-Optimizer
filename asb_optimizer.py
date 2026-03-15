import aerosandbox as asb
import aerosandbox.numpy as np

# ══════════════════════════════════════════════════════════════════════════════
#  WING OPTIMIZER
# ══════════════════════════════════════════════════════════════════════════════

# ── Problem Constants ─────────────────────────────────────────────────────────
N_SECTIONS = 5
SEMI_SPAN  = 0.3    # m
V_INF      = 10.0   # m/s
RHO        = 1.225  # kg/m³
W_TARGET   = 6.0    # N
CD0        = 0.01   # Parasitic drag
SM_TARGET  = 0.05   # Static margin

# ── Normalisation References  ────────────────────────────────────────────────
LD_REF    = 15.0
S_WET_REF = 0.10   # m²
CM_REF    = 0.05
SMOOTH_REF = 1.0  # m
CLB_REF    = 0.05  # 1/rad
CNB_REF    = 0.05  # 1/rad 

# ── Penalty weights ───────────────────────────────────────────────────────────
W_LD    = 1.0    # reward L/D
W_AREA  = 0.3    # penalise wetted area
W_PITCH = 0.0    # penalise Cm > 0
W_TRIM   = 0.0   # penalise |Cm|
W_SMOOTH = 0.15   # penalise LE curvature changes
W_ROLL   = 0.7   # penalise Clb > 0
W_YAW    = 0.6   # penalise Cnb < 0

def planform_curvature(x_le, y_le, z_le, chords, n_sections):
    n_interior = n_sections - 2
    dy_avg  = 0.5 * ((y_le[1:-1] - y_le[:-2]) + (y_le[2:] - y_le[1:-1]))
    denom   = dy_avg**2 + 1e-8
    d2x_le  = (x_le[2:]   - 2.0*x_le[1:-1]   + x_le[:-2])   / denom
    d2chord = (chords[2:] - 2.0*chords[1:-1] + chords[:-2]) / denom
    d2z     = (z_le[2:]   - 2.0*z_le[1:-1]   + z_le[:-2])   / denom
    return np.sqrt(np.sum(d2x_le**2 + d2chord**2 + d2z**2) / n_interior + 1e-8)

def compute_score(CL, CD_total, Cm, S_wet, x_le, y_le, z_le, chords, n_sections, Clb=0.0, Cnb=0.0, verbose=False):
    LD     = CL / CD_total
    kappa  = planform_curvature(x_le, y_le, z_le, chords, n_sections)

    t_ld     =  W_LD     * (LD    / LD_REF)
    t_area   = -W_AREA   * (S_wet / S_WET_REF)
    t_pitch  = -W_PITCH  * np.fmax(-Cm, 0.0) / CM_REF
    t_trim   = -W_TRIM   * np.fabs(Cm)      / CM_REF
    t_smooth = -W_SMOOTH * (kappa / SMOOTH_REF)
    t_roll   = -W_ROLL   * np.fmax(Clb, 0.0) / CLB_REF
    t_yaw    = -W_YAW    * np.fmax(-Cnb, 0.0) / CNB_REF

    score = t_ld + t_area + t_pitch + t_trim + t_smooth + t_roll + t_yaw

    if verbose:
        print("\n══ SCORE BREAKDOWN ══════════════════════════════")
        print(f"  L/D term      (+) :  {float(t_ld):+.4f}   (L/D = {float(LD):.2f})")
        print(f"  Area term     (-) :  {float(t_area):+.4f}   (S_wet = {float(S_wet)*1e4:.0f} cm²)")
        print(f"  Pitch term    (-) :  {float(t_pitch):+.4f}   (max(Cm,0) = {max(float(Cm),0):.4f})")
        print(f"  Trim term     (-) :  {float(t_trim):+.4f}   (|Cm| = {abs(float(Cm)):.4f})")
        print(f"  Smooth term   (-) :  {float(t_smooth):+.4f}   (curvature = {float(kappa):.4f} /m)")
        print(f"  Roll stab     (-) :  {float(t_roll):+.4f}   (Clb = {float(Clb):.4f} /rad, want < 0)")
        print(f"  Yaw  stab     (-) :  {float(t_yaw):+.4f}   (Cnb = {float(Cnb):.4f} /rad, want > 0)")
        print(f"  ══ SCORE          :  {float(score):+.4f}")

    return score

# ══════════════════════════════════════════════════════════════════════════════
#  OPTIMIZATION
# ══════════════════════════════════════════════════════════════════════════════
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
    lower_bound = 0.04, upper_bound = 0.35,
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

airplane = asb.Airplane(
    name    = "Plane",
    xyz_ref = [0.0, 0.0, 0.0],
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

# CG Calculation
panel_spans  = np.diff(y_le)
panel_chords = 0.5 * (chords[:-1] + chords[1:])
MAC = np.sum(panel_chords**2 * panel_spans) / np.sum(panel_chords * panel_spans)
x_NP = aero["x_np"]
x_CG = x_NP - SM_TARGET * MAC

# Pitching moment transferred to the CG
x_ref = 0.0   # xyz_ref[0]
Cm_CG = aero["Cm"] - (x_CG - x_ref) / MAC * aero["CL"]

# Stability derivatives
Clb = aero["Clb"]   # Negative = roll stable
Cnb = aero["Cnb"]   # Positive = yaw  stable

# ── Derived quantities ────────────────────────────────────────────────────────
S_ref = 2.0 * np.sum(0.5 * (chords[:-1] + chords[1:]) * np.diff(y_le))
b     = 2.0 * SEMI_SPAN
AR    = b ** 2 / S_ref
q     = 0.5 * RHO * V_INF ** 2
L     = q * S_ref * aero["CL"]
S_wet = 2.0 * S_ref
CD_total = aero["CD"] + CD0

# ── Hard constraints ──────────────────────────────────────────────────────────
opti.subject_to(x_le[0] == 0.0)
opti.subject_to(y_le[0] == 0.0)
opti.subject_to(z_le[0] == 0.0)
opti.subject_to(y_le[-1] == SEMI_SPAN)
opti.subject_to(np.diff(y_le)   >= 0.03)    # Min panel width
opti.subject_to(np.diff(chords) <= -0.004)  # Monotone taper
opti.subject_to(np.diff(x_le)  >= 0.0)     # Monotone sweep
opti.subject_to(np.diff(z_le)  >= 0.0)     # Monotone dihedral
opti.subject_to(np.diff(twists) <=  2.5)    # twist smoothness
opti.subject_to(np.diff(twists) >= -2.5)
opti.subject_to(AR >= 4.0)
opti.subject_to(AR <= 12.0)
opti.subject_to(S_ref >= 0.02)   # m²
opti.subject_to(S_ref <= 0.20)   # m²
opti.subject_to(L >= W_TARGET)   # lift >= 6 N
opti.subject_to(aero["CL"] >= 0.3)   # Positive lift
CD_induced_min = aero["CL"] ** 2 / (np.pi * 12.0)
opti.subject_to(aero["CD"] >= CD_induced_min)
opti.subject_to(Cm_CG <= 0.02)
opti.subject_to(Cm_CG >= -0.1)

# ── Objective ─────────────────────────────────────────────────────────────────
score = compute_score(aero["CL"], CD_total, aero["Cm"], S_wet, x_le, y_le, z_le, chords, N_SECTIONS, Clb=Clb, Cnb=Cnb)
opti.minimize(-score)

print("Starting optimization...")
try:
    sol = opti.solve(
        max_iter = 500,
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
        },
    )
except RuntimeError:
    print("  Solver did not fully converge — using best point found.")
    sol = opti.debug

# ── Results ───────────────────────────────────────────────────────────────────
sol_CL       = sol.value(aero["CL"])
sol_CD       = sol.value(aero["CD"])
sol_Cm       = sol.value(aero["Cm"])
sol_chords   = sol.value(chords)
sol_y_le     = sol.value(y_le)
sol_S_ref    = sol.value(S_ref)
sol_S_wet    = sol.value(S_wet)
sol_AR       = sol.value(AR)
sol_L        = sol.value(L)
sol_CD_total = sol_CD + CD0

print("\n═══ OPTIMAL WING ════════════════════════════════")
print(f"  Alpha          : {sol.value(alpha):.2f} °")
print(f"  CL             : {sol_CL:.4f}")
print(f"  CD (total)     : {sol_CD_total:.4f}")
print(f"  L/D            : {sol_CL / sol_CD_total:.3f}")
print(f"  Cm             : {sol_Cm:.4f}")
print(f"  Lift           : {sol_L:.3f} N  (required ≥ {W_TARGET} N)")
print(f"  Wing area      : {sol_S_ref * 1e4:.1f} cm²")
print(f"  Wetted area    : {sol_S_wet * 1e4:.1f} cm²")
print(f"  Aspect ratio   : {sol_AR:.2f}")
print(f"  Chords (cm)    : {np.round(sol_chords * 100, 1)}")
print(f"  Twists (°)     : {np.round(sol.value(twists), 2)}")
print(f"  Span (cm)      : {np.round(sol_y_le * 100, 1)}")
print(f"  Sweep x (cm)   : {np.round(sol.value(x_le) * 100, 1)}")
print(f"  Dihedral (cm)  : {np.round(sol.value(z_le) * 100, 1)}")
print(f"  x_NP (% MAC)   : {sol.value(x_NP)*100:.1f} mm from ref")
print(f"  x_CG (% MAC)   : {sol.value(x_CG)*100:.1f} mm from ref")
print(f"  Static margin  : {SM_TARGET*100:.1f}% MAC")
print(f"  MAC            : {sol.value(MAC)*100:.1f} mm")
print(f"  Cm about CG    : {sol.value(Cm_CG):.6f}  (should be ~0)")
print(f"  Clb            : {sol.value(Clb):.4f} /rad  (want < 0, roll stable)")
print(f"  Cnb            : {sol.value(Cnb):.4f} /rad  (want > 0, yaw  stable)")
print("═════════════════════════════════════════════════")

sol_Cm_CG = sol.value(Cm_CG)
compute_score(sol_CL, sol_CD_total, sol_Cm_CG, sol_S_wet,
              sol.value(x_le), sol_y_le, sol.value(z_le), sol_chords, N_SECTIONS,
              Clb=sol.value(Clb), Cnb=sol.value(Cnb),
              verbose=True)

# Rebuild solution airplane
sol_x_le   = sol.value(x_le)
sol_z_le   = sol.value(z_le)
sol_twists = sol.value(twists)

sol_airplane = asb.Airplane(
    name    = "Optimal Wing",
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