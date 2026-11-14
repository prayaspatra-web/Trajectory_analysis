#!/usr/bin/env python3
"""
Description:
Computes ionic conductivity (S/m) and the Haven ratio for a 3-component ionic system
(1 cation species, 2 anion species) using linear regression on block-averaged
Mean Squared Displacement (MSD) and distinct-ion correlation curves.

Methodology:
  1. Fits linear slopes in the diffusive regime (5000 - 12500 ps) across 9 self,
     same-species distinct, and cross-species correlation datasets.
  2. Calculates Nernst-Einstein conductivity (sigma_NE) from self-diffusion slopes
     and total Onsager/Green-Kubo conductivity (sigma_total) including cross-correlations.
  3. Cross-validates the 9-term decomposition against a direct Einstein-Helfand
     collective dipole displacement fit ("eh_direct_blockavg.dat").

Output:
  Prints fitting statistics and conductivity breakdown to stdout and writes the full report to "ionic_conductivity.txt".
"""

import numpy as np
from scipy.stats import linregress

e  = 1.602176634e-19
kB = 1.380649e-23
T  = 298
z_plus = +0.8
z_an1  = -0.8
z_an2  = -0.8
N_plus = 1000
N_an1  = 250
N_an2  = 750
L_ANGSTROM = 74.0460
volume_ang3 = L_ANGSTROM ** 3
volume_m3   = volume_ang3 * 1e-30

files={"plus_self":"msd_cation_self_blockavg.dat","an1_self":"msd_anion1_self_blockavg.dat","an2_self":"msd_anion2_self_blockavg.dat","pp":"msd_cat_cat_blockavg.dat",
"an1_an1":"msd_an1_an1_blockavg.dat","an2_an2":"msd_an2_an2_blockavg.dat","plus_an1":"msd_cat_an1_blockavg.dat","plus_an2":"msd_cat_an2_blockavg.dat","an1_an2":"msd_an1_an2_blockavg.dat"}

T_MIN = 5000.0
T_MAX = 12500.0

slopes = {}
r2_values = {}

print("============================================================")
print("FITTING MSD / DISTINCT CORRELATION SLOPES")
print("============================================================")

for key, filename in files.items():

    data = np.loadtxt(filename, comments="#")

    t = data[:, 0]
    y = data[:, 1]

    mask = (t >= T_MIN) & (t <= T_MAX)

    if np.count_nonzero(mask) < 2:
        raise RuntimeError(
            f"Not enough points in fit window for {filename}"
        )

    result = linregress(t[mask], y[mask])

    slopes[key] = result.slope
    r2_values[key] = result.rvalue ** 2

    print(
        f"{key:12s} : "
        f"{result.slope: .8e} pm^2/ps "
        f"(R^2 = {result.rvalue**2:.6f})"
    )

CONVERSION = 1e-12

for key in slopes:
    slopes[key] *= CONVERSION

common = e**2 / (6.0 * kB * T * volume_m3)
sigma_plus = (
    common
    * N_plus
    * z_plus**2
    * slopes["plus_self"]
)

sigma_an1 = (
    common
    * N_an1
    * z_an1**2
    * slopes["an1_self"]
)

sigma_an2 = (
    common
    * N_an2
    * z_an2**2
    * slopes["an2_self"]
)

sigma_pp = (
    common
    * N_plus
    * z_plus**2
    * slopes["pp"]
)

sigma_an1_an1 = (
    common
    * N_an1
    * z_an1**2
    * slopes["an1_an1"]
)

sigma_an2_an2 = (
    common
    * N_an2
    * z_an2**2
    * slopes["an2_an2"]
)

sigma_plus_an1 = (
    common
    * z_plus
    * z_an1
    * np.sqrt(N_plus * N_an1)
    * slopes["plus_an1"]
)

sigma_plus_an2 = (
    common
    * z_plus
    * z_an2
    * np.sqrt(N_plus * N_an2)
    * slopes["plus_an2"]
)

sigma_an1_an2 = (
    common
    * z_an1
    * z_an2
    * np.sqrt(N_an1 * N_an2)
    * slopes["an1_an2"]
)

sigma_NE = sigma_plus + sigma_an1 + sigma_an2

sigma_total = (
    sigma_plus
    + sigma_an1
    + sigma_an2
    + sigma_pp
    + sigma_an1_an1
    + sigma_an2_an2
    + sigma_plus_an1
    + sigma_plus_an2
    + sigma_an1_an2
)

sigma_total_corrected = (
    sigma_plus
    + sigma_an1
    + sigma_an2
    + sigma_pp
    + sigma_an1_an1
    + sigma_an2_an2
    + 2.0 * sigma_plus_an1
    + 2.0 * sigma_plus_an2
    + 2.0 * sigma_an1_an2
)

haven_ratio = sigma_total_corrected / sigma_NE if sigma_NE != 0 else np.nan


# ============================================================
# PRINT RESULTS
# ============================================================

print()
print("============================================================")
print("IONIC CONDUCTIVITY")
print("============================================================")

print()
print("Self contributions:")
print("------------------------------------------------------------")

print(f"Cation self       : {sigma_plus: .8e} S/m")
print(f"Anion1 self       : {sigma_an1: .8e} S/m")
print(f"Anion2 self       : {sigma_an2: .8e} S/m")

print()
print("Same-species distinct contributions:")
print("------------------------------------------------------------")

print(f"Cation-Cation     : {sigma_pp: .8e} S/m")
print(f"Anion1-Anion1     : {sigma_an1_an1: .8e} S/m")
print(f"Anion2-Anion2     : {sigma_an2_an2: .8e} S/m")

print()
print("Cross-species contributions:")
print("------------------------------------------------------------")

print(f"Cation-Anion1     : {sigma_plus_an1: .8e} S/m")
print(f"Cation-Anion2     : {sigma_plus_an2: .8e} S/m")
print(f"Anion1-Anion2     : {sigma_an1_an2: .8e} S/m")

print()
print("============================================================")
print(f"NERNST-EINSTEIN CONDUCTIVITY   = {sigma_NE: .8e} S/m")
print(f"TOTAL CONDUCTIVITY (uncorrected) = {sigma_total: .8e} S/m")
print(f"TOTAL CONDUCTIVITY (corrected)   = {sigma_total_corrected: .8e} S/m")
print(f"HAVEN RATIO (sigma / sigma_NE)   = {haven_ratio: .4f}")
print("============================================================")

with open("ionic_conductivity.txt", "w") as f:

    f.write("Ionic Conductivity\n")
    f.write("==================\n\n")

    f.write(f"Temperature = {T} K\n")
    f.write(f"Box length  = {L_ANGSTROM} Angstrom\n")
    f.write(f"Volume      = {volume_m3:.8e} m^3\n\n")

    f.write("Charges:\n")
    f.write(f"z_plus = {z_plus}\n")
    f.write(f"z_an1  = {z_an1}\n")
    f.write(f"z_an2  = {z_an2}\n\n")

    f.write("Particle numbers:\n")
    f.write(f"N_plus = {N_plus}\n")
    f.write(f"N_an1  = {N_an1}\n")
    f.write(f"N_an2  = {N_an2}\n\n")

    f.write("Fit window:\n")
    f.write(f"{T_MIN} - {T_MAX} ps\n\n")

    f.write("Conductivity components:\n")
    f.write(f"Cation self       = {sigma_plus:.8e} S/m\n")
    f.write(f"Anion1 self       = {sigma_an1:.8e} S/m\n")
    f.write(f"Anion2 self       = {sigma_an2:.8e} S/m\n")
    f.write(f"Cation-Cation     = {sigma_pp:.8e} S/m\n")
    f.write(f"Anion1-Anion1     = {sigma_an1_an1:.8e} S/m\n")
    f.write(f"Anion2-Anion2     = {sigma_an2_an2:.8e} S/m\n")
    f.write(f"Cation-Anion1     = {sigma_plus_an1:.8e} S/m\n")
    f.write(f"Cation-Anion2     = {sigma_plus_an2:.8e} S/m\n")
    f.write(f"Anion1-Anion2     = {sigma_an1_an2:.8e} S/m\n\n")

    f.write(f"NERNST-EINSTEIN CONDUCTIVITY     = {sigma_NE:.8e} S/m\n")
    f.write(f"TOTAL CONDUCTIVITY (uncorrected) = {sigma_total:.8e} S/m\n")
    f.write(f"TOTAL CONDUCTIVITY (corrected)   = {sigma_total_corrected:.8e} S/m\n")
    f.write(f"HAVEN RATIO (sigma / sigma_NE)   = {haven_ratio:.4f}\n")


print()
print("Results saved to ionic_conductivity.txt")

EH_FILE = "eh_direct_blockavg.dat"

eh_data = np.loadtxt(EH_FILE, comments="#")

t_eh = eh_data[:, 0]
y_eh = eh_data[:, 1]

eh_mask = (t_eh >= T_MIN) & (t_eh <= T_MAX)

if np.count_nonzero(eh_mask) < 2:
    raise RuntimeError(
        f"Not enough points in fit window for {EH_FILE}"
    )

eh_fit = linregress(t_eh[eh_mask], y_eh[eh_mask])

eh_slope_pm2_ps = eh_fit.slope
eh_r2 = eh_fit.rvalue ** 2

eh_slope_m2_s = eh_slope_pm2_ps * CONVERSION

sigma_EH = common * eh_slope_m2_s

haven_ratio_EH = sigma_EH / sigma_NE if sigma_NE != 0 else np.nan

print()
print("============================================================")
print("DIRECT EINSTEIN-HELFAND FIT")
print("============================================================")
print(
    f"EH direct slope : {eh_slope_pm2_ps: .8e} pm^2/ps "
    f"(R^2 = {eh_r2:.6f})"
)
print(f"EH conductivity : {sigma_EH: .8e} S/m")
print(f"Haven Ratio (EH / NE) : {haven_ratio_EH: .4f}")

difference = sigma_EH - sigma_total_corrected

print()
print("============================================================")
print("EH VALIDATION")
print("============================================================")
print(f"Original 9-term total  : {sigma_total: .8e} S/m")
print(f"Corrected 9-term total : {sigma_total_corrected: .8e} S/m")
print(f"Direct EH conductivity : {sigma_EH: .8e} S/m")
print(f"EH - corrected 9-term  : {difference: .8e} S/m")

if sigma_EH != 0.0:
    relative_difference = (
        100.0 * abs(difference) / abs(sigma_EH)
    )
    print(
        f"Relative difference    : "
        f"{relative_difference:.6f} %"
    )

with open("ionic_conductivity.txt", "a") as f:

    f.write("\n")
    f.write("Direct Einstein-Helfand analysis:\n")
    f.write("---------------------------------\n")
    f.write(f"EH fit window = {T_MIN} - {T_MAX} ps\n")
    f.write(f"EH slope      = {eh_slope_pm2_ps:.8e} pm^2/ps\n")
    f.write(f"EH R^2        = {eh_r2:.8f}\n")
    f.write(f"EH conductivity = {sigma_EH:.8e} S/m\n")
    f.write(f"Haven Ratio (EH / NE) = {haven_ratio_EH:.4f}\n\n")

    f.write("EH validation:\n")
    f.write(
        f"Original 9-term total  = "
        f"{sigma_total:.8e} S/m\n"
    )
    f.write(
        f"Corrected 9-term total = "
        f"{sigma_total_corrected:.8e} S/m\n"
    )
    f.write(
        f"Direct EH conductivity = "
        f"{sigma_EH:.8e} S/m\n"
    )
    f.write(
        f"EH - corrected 9-term = "
        f"{difference:.8e} S/m\n"
    )

print()
print("Direct EH result appended to ionic_conductivity.txt")
