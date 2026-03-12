import aerosandbox as asb
import aerosandbox.numpy as np

n_sections = 5

# Main Wing Parameters
y_le = np.array([0.0, 0.1, 0.2, 0.3, 0.4])            # Spanwise sections: Y location
x_le = np.array([0.0, 0.01, 0.03, 0.06, 0.1])         # Sweep: X location
z_le = np.array([0.0, 0.005, 0.01, 0.02, 0.035])      # Dihedral: Z location
chords = np.array([0.15, 0.13, 0.1, 0.07, 0.03])      # Chord lengths
twists = np.array([2.0, 1.0, 0.0, -1.0, -2.0])        # Washout
airfoils = [
    asb.Airfoil("naca4415"),
    asb.Airfoil("naca4412"),
    asb.Airfoil("naca2412"),
    asb.Airfoil("naca0012"),
    asb.Airfoil("naca0010")
]
elevon_hinge_point = 0.75

# Wing Sections
sections = []
for i in range(n_sections):
    cs_list = []
    if i == 3: 
        cs_list = [
            asb.ControlSurface(
                name="elevon_pitch",
                hinge_point=elevon_hinge_point,
                symmetric=True,
                deflection=0.0 
            )]
    sections.append(
        asb.WingXSec(
            xyz_le=[x_le[i], y_le[i], z_le[i]],
            chord=chords[i],
            twist=twists[i],
            airfoil=airfoils[i],
            control_surfaces=cs_list
        )
    )
airplane = asb.Airplane(
    name="Parameterized Array Wing",
    xyz_ref=[0.05, 0, 0], # Center of Gravity
    wings=[
        asb.Wing(
            name="Main Wing",
            symmetric=True,
            xsecs=sections
        )
    ]
)

# airplane.draw()

# VLM Analysis
op_point = asb.OperatingPoint(
    velocity=25.0, # m/s
    alpha=5.0,     # degrees
)
vlm = asb.VortexLatticeMethod(
    airplane=airplane,
    op_point=op_point,
    spanwise_resolution=3,
    chordwise_resolution=5
)
aero = vlm.run()

# Aero Build-Up
deflected_airplane = airplane.with_control_deflections(control_surface_deflection_mappings={"elevon_pitch": -1.0})
aero_buildup = asb.AeroBuildup(
    airplane=deflected_airplane,
    op_point=op_point
)
buildup = aero_buildup.run()

# Results
print(f"CL:                {aero['CL']:.2f}")
print(f"CD:                {aero['CD']:.2f}")
print(f"Lift/Drag Ratio:   {aero['CL']/aero['CD']:.2f}")
print(f"Pitching moment:   {buildup['M_b'][1][0]:.2f} Nm")
# vlm.draw()