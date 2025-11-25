import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

# ----------------------
# User data
# ----------------------
lmax = np.array([0, 2, 4, 8, 16, 32], dtype=float)

assembly_time = np.array([1.547568, 1.5638285, 1.679423, 1.742959,
                          2.2370745, 3.8803145], dtype=float)
solver_time   = np.array([19.397185, 18.820575, 19.012055, 19.176555,
                          20.232415, 22.421565], dtype=float)
l2_error      = np.array([0.3705555254, 0.007470455881, 0.0002347503247,
                          3.748709047e-07, 5.528579505e-09, 5.530818461e-09], dtype=float)

# ----------------------
# Time scales (you asked to DIVIDE by these)
# ----------------------
T_neumann = 16.880775   # reference solver time scale
T_diff    = 1.0         # reference diffusion assembly time scale

# nondimensionalize by DIVISION (as you requested)
assembly_nd = assembly_time / T_diff     # here T_diff=1 -> same numerically, kept for clarity
solver_nd   = solver_time   / T_neumann  # divide by T_neumann to nondimensionalize

# ----------------------
# Plot style settings
# ----------------------
plt.rcParams.update({
    "text.usetex": False,               # use mathtext (safe without TeX)
    "mathtext.fontset": "dejavuserif",
    "font.family": "serif",
    "font.size": 11,
})

COLOR_SOLVER = "#005AA7"   # approved blue
COLOR_ASSEMB = "#C83737"   # approved red

FIGSIZE = (5.2, 3.6)   # width x height in inches (same height for both)
DPI = 300
MARKER_SIZE = 5
LINE_WIDTH = 1.2

xticks = lmax.tolist()  # [0,2,4,8,16,32]

# ----------------------
# Figure (b): assembly & solver time vs lmax (nondimensional times)
# ----------------------
fig1, ax1 = plt.subplots(figsize=FIGSIZE, dpi=DPI)

# solver: solid line (use blue)
ax1.plot(lmax, solver_nd, marker='o', markersize=MARKER_SIZE,
         linewidth=LINE_WIDTH, color=COLOR_SOLVER, label="Solver time")

# assembly: dashed line (use red)
ax1.plot(lmax, assembly_nd, marker='s', markersize=MARKER_SIZE,
         linewidth=LINE_WIDTH, color=COLOR_ASSEMB, linestyle='--', label="Assembly time")

ax1.set_xlabel(r"Maximum SH order $\ell_{\max}$")
ax1.set_ylabel("Time (nondimensional)")   # you can change label to "Time (s)" if you prefer
ax1.set_xticks(xticks)
ax1.grid(True, linestyle='--', linewidth=0.6, alpha=0.6)

# ticks inward on all sides; majors longer than minors
ax1.tick_params(axis='both', which='major', direction='in', length=6, top=True, right=True)
ax1.tick_params(axis='both', which='minor', direction='in', length=3, top=True, right=True)
ax1.xaxis.set_minor_locator(AutoMinorLocator(2))
ax1.yaxis.set_minor_locator(AutoMinorLocator(2))

# vertical (stacked) legend centered gently at top
ax1.legend(loc='upper center', bbox_to_anchor=(0.5, 0.92),
           ncol=1, fontsize=9, framealpha=0.95)

fig1.tight_layout(pad=0.2)
fig1.savefig("dtn_times_vs_lmax.png", dpi=DPI, bbox_inches='tight')
plt.show()

# ----------------------
# Figure (c): global relative L2 error vs lmax (log scale)
# ----------------------
fig2, ax2 = plt.subplots(figsize=FIGSIZE, dpi=DPI)

# Use black curve with markers as in your sample; keep modest linewidth/markers
ax2.loglog(lmax, l2_error, marker='o', markersize=MARKER_SIZE,
           linewidth=LINE_WIDTH, color='k', label=r"Global relative $L_2$ error")

ax2.set_xlabel(r"Maximum SH order $\ell_{\max}$")
ax2.set_ylabel(r"Global relative $L_2$ error")
ax2.set_xticks(xticks)
# keep x-axis tick labels as plain numbers (not scientific)
ax2.get_xaxis().set_major_formatter(plt.ScalarFormatter())
ax2.get_xaxis().tick_bottom()

# grid (both major & minor)
ax2.grid(True, which='both', linestyle='--', linewidth=0.6, alpha=0.6)

# ticks inward on all sides; major vs minor lengths
ax2.tick_params(axis='both', which='major', direction='in', length=6, top=True, right=True)
ax2.tick_params(axis='both', which='minor', direction='in', length=3, top=True, right=True)

# for log scale the minor-locator is automatic; ensure minor ticks exist between decades
# (Matplotlib provides them automatically for log scale)
# vertical stacked legend centered at top (single entry)
ax2.legend(loc='upper center', bbox_to_anchor=(0.5, 0.92),
           ncol=1, fontsize=9, framealpha=0.95)

fig2.tight_layout(pad=0.2)
fig2.savefig("dtn_l2error_vs_lmax.png", dpi=DPI, bbox_inches='tight')
plt.show()

print("Saved: dtn_times_vs_lmax.png, dtn_l2error_vs_lmax.png")

