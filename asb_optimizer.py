import aerosandbox as asb
import aerosandbox.numpy as np

n_sections = 5

# Main Wing Parameters
y_le = np.array([0.0, 1.25, 2.5, 3.75, 5.0])          # Spanwise sections: Y location
x_le = np.array([0.0, 0.1, 0.3, 0.6, 1.0])            # Sweep: X location
z_le = np.array([0.0, 0.05, 0.1, 0.2, 0.35])          # Dihedral: Z location
chords = np.array([1.5, 1.3, 1.0, 0.7, 0.3])          # Chord lengths
twists = np.array([2.0, 1.0, 0.0, -1.0, -2.0])        # Washout
airfoils = [
    asb.Airfoil("naca4415"),
    asb.Airfoil("naca4412"),
    asb.Airfoil("naca2412"),
    asb.Airfoil("naca0012"),
    asb.Airfoil("naca0010")
]

# Wing Sections
sections = []
for i in range(n_sections):
    sections.append(
        asb.WingXSec(
            xyz_le=[x_le[i], y_le[i], z_le[i]],
            chord=chords[i],
            twist=twists[i],
            airfoil=airfoils[i]
        )
    )

airplane = asb.Airplane(
    name="Parameterized Array Wing",
    xyz_ref=[0.5, 0, 0], # Center of Gravity
    wings=[
        asb.Wing(
            name="Main Wing",
            symmetric=True,
            xsecs=sections
        )
    ]
)

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

# Results
print(f"Total Lift (L):   {aero['L']:.2f} N")
print(f"Total Drag (D):   {aero['D']:.2f} N")
print(f"Lift/Drag Ratio:  {aero['L']/aero['D']:.2f}")
vlm.draw()