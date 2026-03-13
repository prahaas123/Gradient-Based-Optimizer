import aerosandbox as asb
import aerosandbox.numpy as np

# ══════════════════════════════════════════════════════════════════════════════
#  WING OPTIMIZER
# ══════════════════════════════════════════════════════════════════════════════

# ── Problem constants ─────────────────────────────────────────────────────────
N_SECTIONS = 5
SEMI_SPAN  = 0.3    # m
V_INF      = 10.0   # m/s
RHO        = 1.225  # kg/m³
W_TARGET   = 6.0    # N 
CD0        = 0.01

# ── Normalisation references (set to expected typical values) ─────────────────
LD_REF    = 15.0   # expected L/D at a reasonable design point
S_WET_REF = 0.10   # m²  — full-span wetted area reference (~1000 cm²)
CM_REF    = 0.05   # expected |Cm| magnitude from VLM on this geometry

# ── Penalty weights ───────────────────────────────────────────────────────────
W_LD    = 0.8   # reward L/D
W_AREA  = 0.3   # penalise wetted area
W_PITCH = 0.0   # penalise Cm > 0
W_TRIM  = 0.0   # penalise |Cm|


def compute_score(CL, CD_total, Cm, S_wet, verbose=False):
    """
    SCORE = w_ld·(L/D / LD_REF)
          - w_area·(S_wet / S_WET_REF)
          - w_pitch·max(Cm, 0) / CM_REF
          - w_trim·|Cm| / CM_REF

    All terms are O(1) at typical values, so weights are directly meaningful.

    Parameters
    ----------
    CL       : lift coefficient
    CD_total : total drag coefficient (induced + CD0)
    Cm       : pitching moment coefficient
    S_wet    : full-span wetted area (m²)
    """
    LD = CL / CD_total

    t_ld    =  W_LD    * (LD   / LD_REF)
    t_area  = -W_AREA  * (S_wet / S_WET_REF)
    t_pitch = -W_PITCH * np.fmax(Cm, 0.0) / CM_REF
    t_trim  = -W_TRIM  * np.fabs(Cm)      / CM_REF

    score = t_ld + t_area + t_pitch + t_trim

    if verbose:
        print("\n══ SCORE BREAKDOWN ══════════════════════════════")
        print(f"  L/D term    (+) :  {float(t_ld):+.4f}   (L/D = {float(LD):.2f})")
        print(f"  Area term   (-) :  {float(t_area):+.4f}   (S_wet = {float(S_wet)*1e4:.0f} cm²)")
        print(f"  Pitch term  (-) :  {float(t_pitch):+.4f}   (max(Cm,0) = {max(float(Cm),0):.4f})")
        print(f"  Trim term   (-) :  {float(t_trim):+.4f}   (|Cm| = {abs(float(Cm)):.4f})")
        print(f"  ══ SCORE        :  {float(score):+.4f}")

    return score


# ══════════════════════════════════════════════════════════════════════════════
#  OPTIMISATION
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
    init_guess  = np.linspace(0.0, 0.015, N_SECTIONS),
    lower_bound = -0.04, upper_bound = 0.08,
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
    init_guess  = 5.0,
    lower_bound = 2.0, upper_bound = 14.0,
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
    spanwise_resolution  = 5,
    chordwise_resolution = 5,
)
aero = vlm.run()

# ── Derived quantities ────────────────────────────────────────────────────────
S_ref = 2.0 * np.sum(0.5 * (chords[:-1] + chords[1:]) * np.diff(y_le))
b     = 2.0 * SEMI_SPAN
AR    = b ** 2 / S_ref
q     = 0.5 * RHO * V_INF ** 2
L     = q * S_ref * aero["CL"]

# Wetted area ≈ 2 × planform area (top + bottom surface, no thickness correction)
S_wet = 2.0 * S_ref

CD_total = aero["CD"] + CD0

# ── Hard constraints ──────────────────────────────────────────────────────────
opti.subject_to(x_le[0] == 0.0)
opti.subject_to(y_le[0] == 0.0)
opti.subject_to(z_le[0] == 0.0)
opti.subject_to(y_le[-1] == SEMI_SPAN)
opti.subject_to(np.diff(y_le)   >= 0.03)    # min panel width
opti.subject_to(np.diff(chords) <= -0.004)  # monotone taper
opti.subject_to(np.diff(twists) <=  2.5)    # twist smoothness
opti.subject_to(np.diff(twists) >= -2.5)
opti.subject_to(AR >= 4.0)
opti.subject_to(AR <= 12.0)
opti.subject_to(S_ref >= 0.02)   # m²
opti.subject_to(S_ref <= 0.20)   # m²
opti.subject_to(L >= W_TARGET)   # lift ≥ 6 N
CD_induced_min = aero["CL"] ** 2 / (np.pi * 12.0)
opti.subject_to(aero["CD"] >= CD_induced_min)

# ── Objective ─────────────────────────────────────────────────────────────────
score = compute_score(aero["CL"], CD_total, aero["Cm"], S_wet)
opti.minimize(-score)

print("Starting optimisation...")
try:
    sol = opti.solve(
        max_iter = 100,
        options  = {
            "ipopt.tol":                1e-6,
            "ipopt.constr_viol_tol":    1e-6,
            "ipopt.acceptable_tol":     1e-4,
            "ipopt.acceptable_iter":    5,
            "ipopt.mu_strategy":        "adaptive",
            "ipopt.nlp_scaling_method": "gradient-based",
        },
    )
except RuntimeError:
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
print("═════════════════════════════════════════════════")

compute_score(sol_CL, sol_CD_total, sol_Cm, sol_S_wet, verbose=True)

# Draw the optimized wing
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