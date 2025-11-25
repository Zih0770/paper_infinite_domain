import numpy as np
import matplotlib.pyplot as plt

# Use mathtext (no external LaTeX). Looks very similar to LaTeX but is local.
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 18,
    "mathtext.fontset": "stix",   # good-looking math fonts without external LaTeX
})

# Load DtN result (your uploaded file)
dtn = np.loadtxt("results/prem_phi_vs_r.dat")
r_dtn, phi_dtn = dtn[:,0], dtn[:,1]

spectral_data = np.loadtxt(
    "results/prem_phi_vs_r_spectral_high.dat",
    delimiter=";"
)
r_spec, phi_spec = spectral_data[:,0], spectral_data[:,1]

# Scale spectral so values match at r ~ 0
idx_spec_r0 = np.argmin(np.abs(r_spec - 0.0))
idx_dtn_r0  = np.argmin(np.abs(r_dtn  - 0.0))
#scale = phi_dtn[idx_dtn_r0] / phi_spec[idx_spec_r0]
scale = 1.0
phi_spec_scaled = phi_spec * scale

# Normalize radius to r/R with R = 1.2
R = 1.2
x_dtn = r_dtn / R
x_spec = r_spec / R

# Plot
plt.figure(figsize=(8,6))
plt.plot(x_dtn, phi_dtn,  '-', linewidth=2.2, label=r"DtN ($\ell_{\max}=32$)")
plt.plot(x_spec, phi_spec_scaled, '-', linewidth=2.2, label="Spectral benchmark")

plt.xlabel(r"$r/R$", fontsize=20)
plt.ylabel(r"$\phi(r)$", fontsize=20)
plt.tick_params(labelsize=14)
plt.legend(fontsize=14)
plt.grid(True, alpha=0.25)
plt.tight_layout()
plt.savefig("phi_vs_r_prem.png", dpi=300)
plt.close()
print("Saved phi_vs_r_prem.png")

