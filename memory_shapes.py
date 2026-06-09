import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma

# 1. Define Time axis (Age of event: tn - tj)
# Starting slightly above 0 to avoid divide-by-zero in the power-law equation
t = np.linspace(0.01, 10, 1000) 
dt = t[1] - t[0]

# 2. Base Memory Parameters
S = 1.0         # Memory Strength
lam = 0.5       # Exponential decay rate (lambda)
tau = 7.0       # Memory length (cutoff)
tau_d = 2.0     # Delay / Latency period
alpha = 1.3     # Gamma distribution shape parameter
beta = 0.5      # Power-law fractional order (0 < beta < 1)

# 3. Define the Memory Kernels
# Eq 8: Exponential memory decay
M_exp = S * np.exp(-lam * t) * np.heaviside(tau - t, 1)

# Eq 9: Simple Shifted Heaviside (Latency without length cutoff)
M_shift = S * np.exp(-lam * (t - tau_d)) * np.heaviside(t - tau_d, 1)

# Eq 10: Bandpass Heaviside (Latency with strict length)
M_bandpass = S * np.exp(-lam * (t - tau_d)) * np.heaviside(t - tau_d, 1) * np.heaviside(tau - t, 1)

# Eq 11: Gamma distribution model (Smooth delay and peak)
M_gamma = S * (t**alpha) * np.exp(-lam * t) * np.heaviside(tau - t, 1)

# Eq 12: Power-Law model (Long-term fractional memory)
term1 = (t / dt)**beta
term2 = (np.maximum(t / dt - 1, 0))**beta
M_powerlaw = (dt**(beta - 1)) * ((term1 - term2) / gamma(beta + 1))

# 4. Plotting
fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(t, M_exp, label='Exponential Decay (Eq A1)', lw=2.5, color='#1f77b4')
ax.plot(t, M_shift, label='Shifted Heaviside (Eq A2)', lw=2.5, color='#ff7f0e')
ax.plot(t, M_bandpass, label='Bandpass Heaviside (Eq A3)', lw=2.5, color='#2ca02c', linestyle='--')
ax.plot(t, M_gamma, label='Gamma Distribution (Eq A4)', lw=2.5, color='#d62728', linestyle=':')
ax.plot(t, M_powerlaw, label='Power-Law (Eq A5)', lw=2.5, color='#9467bd')

# Mark the specific thresholds for clarity
ax.axvline(tau_d, color='gray', linestyle=':', label=r'Latency ($\tau_d$)')
ax.axvline(tau, color='gray', linestyle='-.', label=r'Length Cutoff ($\tau$)')

# Reverse the x-axis to represent "Time Before Present" properly
ax.invert_xaxis()

# Formatting and Limits
ax.set_ylim(-0.1, 2.5) # Cap Y-axis to prevent power-law from skewing the scale
ax.set_xlim(10, 0)     # Ensure X-axis goes from 10 on the left to 0 on the right

ax.set_xlabel('Time Before Present ($t_n - t_j$)', fontsize=12)
ax.set_ylabel('Memory Weight ($M_i$)', fontsize=12)
ax.set_title('Different Types of Memory Kernels', fontsize=14, pad=15)
ax.grid(True, alpha=0.3)
ax.legend(loc='upper left', framealpha=0.9)

# to save the figure, uncomment the line below
# Save as EPS
plt.savefig('memory_kernels.eps', format='eps', bbox_inches='tight')

# Save as PNG (dpi=300 ensures a high-resolution image)
plt.savefig('memory_kernels.png', format='png', dpi=300, bbox_inches='tight')

plt.tight_layout()
plt.show()

