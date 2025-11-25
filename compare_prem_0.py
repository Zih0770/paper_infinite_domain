import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 18,
    "mathtext.fontset": "stix",   
})

dtn_100 = np.loadtxt("results/prem_phi_vs_r_100.dat")
r_dtn_100, phi_dtn_100 = dtn_100[:,0], dtn_100[:,1]
dtn_200 = np.loadtxt("results/prem_phi_vs_r_200.dat")
r_dtn_200, phi_dtn_200 = dtn_200[:,0], dtn_200[:,1]

spectral_data = np.loadtxt(
    "results/prem_phi_vs_r_spectral_high.dat",
    delimiter=";"
)
r_spec, phi_spec = spectral_data[:,0], spectral_data[:,1]

idx_spec_r0 = np.argmin(np.abs(r_spec - 0.0))
idx_dtn_r0_100  = np.argmin(np.abs(r_dtn_100  - 0.0))
scale = phi_dtn_100[idx_dtn_r0_100] / phi_spec[idx_spec_r0]
phi_spec_scaled = phi_spec * scale

# Plot
plt.figure(figsize=(8,6))
plt.plot(r_dtn_100,  phi_dtn_100, '-', linewidth=2.2, label=r"DtN ($h_{\mathrm{min}}=100\mathrm{km}$)")
#plt.plot(r_dtn_200,  phi_dtn_200, '-', linewidth=2.2, label=r"DtN ($h_{\mathrm{min}}=200\mathrm{km}$)")
plt.plot(r_spec, phi_spec_scaled, '-', linewidth=2.2, label="Spectral benchmark")

plt.xlabel(r"$r/R$", fontsize=20)
plt.ylabel(r"$\phi(r)$", fontsize=20)
plt.tick_params(labelsize=14)
plt.legend(fontsize=14)
plt.grid(True, alpha=0.25)
plt.tight_layout()
plt.savefig("phi_vs_r_prem.png", dpi=300)
plt.close()
print("Saved phi_vs_r_prem.png")

