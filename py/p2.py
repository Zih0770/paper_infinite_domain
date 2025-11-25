import matplotlib.pyplot as plt
import numpy as np

domain_sizes = np.array([10/7, 4, 6, 8, 10, 12, 14, 16], dtype=float)

neumann_solver   = np.array([
    17.005435, 19.075785, 18.983635, 19.070915,
    19.10659,  19.69043,  22.12822,  26.4945
])

dirichlet_solver = np.array([
    16.549225, 18.5285,   19.013215, 18.560565,
    18.878125, 20.220065, 22.096525, 26.27468
])

neumann_l2 = np.array([
    0.4151501674, 0.2756920147, 0.2819790013, 0.2849233729,
    0.2855453076, 0.2899858148, 0.293002253,  0.2953538709
])

dirichlet_l2 = np.array([
    0.1918708178, 0.007748085871, 0.002275172009, 0.0009569603642,
    0.000489294336, 0.0002831576029, 0.0001796384461, 0.0001198725187
])

xticks = [2, 4, 6, 8, 10, 12, 14, 16]

BLUE = "#005AA7"   # Dirichlet blue
RED  = "#C83737"   # Neumann red

plt.rcParams.update({
    "mathtext.fontset": "dejavuserif",
    "font.family": "serif",
    "font.size": 11,
})

FIG_WIDTH = 5.0   # inches
FIG_HEIGHT = 3.6  # inches

DPI = 300
MARKER_SIZE = 4
LINE_WIDTH = 1.2

# -------------------------
# Figure (b) Solver time
# -------------------------
fig1, ax1 = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=DPI)

ax1.plot(domain_sizes, dirichlet_solver,
         marker='o', markersize=MARKER_SIZE, linewidth=LINE_WIDTH,
         color=BLUE, label='Dirichlet')
ax1.plot(domain_sizes, neumann_solver,
         marker='s', markersize=MARKER_SIZE, linewidth=LINE_WIDTH,
         color=RED, label='Neumann')

ax1.set_xlabel(r'Domain size $b/a$')
ax1.set_ylabel('Solver time (s)')
ax1.set_xticks(xticks)
ax1.set_ylim(bottom=16)

ax1.grid(True, linestyle='--', linewidth=0.6, alpha=0.6)

ax1.tick_params(axis='both', which='both',
                direction='in', top=True, right=True, length=5)

ax1.legend(loc='upper center', bbox_to_anchor=(0.18, 0.95),
           ncol=1, fontsize=9, framealpha=0.95)

fig1.tight_layout(pad=0.15)
fig1.savefig('nd_times_vs_mesh.png', dpi=DPI, bbox_inches='tight')

plt.show()

# -------------------------
# Figure (c) Global relative L2 error
# -------------------------
fig2, ax2 = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=DPI)

ax2.plot(domain_sizes, dirichlet_l2,
         marker='o', markersize=MARKER_SIZE, linewidth=LINE_WIDTH,
         color=BLUE, label='Dirichlet')
ax2.plot(domain_sizes, neumann_l2,
         marker='s', markersize=MARKER_SIZE, linewidth=LINE_WIDTH,
         color=RED, label='Neumann')

ax2.set_xlabel(r'Domain size $b/a$')
ax2.set_ylabel(r'Global relative $L_2$ error')
ax2.set_xticks(xticks)
ax2.set_yscale('log')

ax2.grid(True, linestyle='--', linewidth=0.6, alpha=0.6)

ax2.tick_params(axis='both', which='both',
                direction='in', top=True, right=True, length=5)

ax2.yaxis.set_tick_params(which='minor', length=3)

ax2.legend(loc='upper center', bbox_to_anchor=(0.8, 0.95),
           ncol=1, fontsize=9, framealpha=0.95)

fig2.tight_layout(pad=0.15)
fig2.savefig('nd_l2_vs_mesh.png', dpi=DPI, bbox_inches='tight')

plt.show()

