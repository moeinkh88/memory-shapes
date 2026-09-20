"""
Figure 5 -- Numerical example: collective social behaviour under five memory
kernels, in fully discrete time.

Discrete-time version: the step size Delta t has been removed everywhere and
one step is read as one month.

Model (manuscript Eq. 3, discrete form):

    x(n) = x(0) + sum_{j=0}^{n-1} M(n,j) f(j, X_j)

with f(j, X_j) the social tension at month j -- the drive term the kernel
weights -- and M(n,j) the mnemonic function, which depends only on the INTEGER
lag l = n - j >= 1. Note the sum stops at j = n-1: a discrete system never
weights its own current step, so the kernels are defined from lag 1, not lag 0.

The engram is the single scandal at month 20, not the whole tension series:
the baseline is the steady state the perturbation interrupts. The tension
series is prescribed exogenously rather than computed from the system state,
since the illustration concerns only how each kernel processes a given drive.

Three changes relative to the continuous script, beyond deleting dt:

1. LAG INDEXING. Kernels are evaluated on l = 1, ..., L instead of on
   ages = 0, ..., n_steps-1. The memoryless baseline is therefore a unit
   spike at lag 1, i.e. x[n] = f[n-1]: a Markovian system reacts one step
   after the event, which is exactly what Eq. (1) says.

2. BURN-IN. The kernels are normalised over a fixed memory horizon L, and the
   simulation is run with L months of pre-history held at the baseline
   tension. Without it, early months only see a truncated piece of the kernel
   and every curve ramps up from below the baseline -- an edge artefact, most
   severe for the power law, which would otherwise start at roughly a tenth of
   the baseline and drift upward across the whole left half of the figure.
   With the burn-in, all five trajectories sit at exactly the baseline before
   the scandal, so the only thing the figure shows is the response to the
   perturbation.

3. POWER LAW. Eq. (8) with Delta t = 1 is simply
       M[l] = [l^beta - (l-1)^beta] / Gamma(beta + 1),
   with no (Delta t)^(beta-1) prefactor.

Each kernel is normalised to unit total mass over the horizon L, so every
channel spends the same total memory "budget" and the curves differ only in
how that budget is distributed across lags (shape, length, latency). This
fixes memory strength across channels; to show strength differences instead,
drop the normalisation and set S_k per kernel.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma as gamma_fn

# ------------------------------------------------- 1. Simulation parameters
n_steps = 100                    # months plotted
L = 100                          # memory horizon: longest lag retained (steps)
burn_in = L                      # pre-history at baseline, so lag L is always available
lags = np.arange(1, L + 1)       # l = 1, ..., L

# ---------------------------------------------------- 2. Social tension f(n)
baseline = 10.0                  # everyday social tension
scandal = 150.0                  # major scandal
event_month = 20

f_full = np.ones(burn_in + n_steps) * baseline
f_full[burn_in + event_month] = scandal

months = np.arange(n_steps)

# ------------------------------------------------- 3. Kernel parameters
lam = 0.15      # decay rate per month
tau_d = 10      # latency, in months
tau = 25        # memory length, in months (after the event)
alpha = 2.0     # Gamma shape parameter
beta = 0.5      # fractional order


def theta(x):
    """Heaviside step with Theta(0) = 1, as defined in the manuscript."""
    return np.heaviside(x, 1.0)


def normalise(w):
    """Unit total mass over the horizon L."""
    return w / np.sum(w)


# ------------------------------------------------------- 4. Memory kernels
# Memoryless (Markovian) baseline: all weight on the immediately previous step
W_memoryless = np.zeros(L)
W_memoryless[0] = 1.0                                                  # lag 1

W_exp = normalise(np.exp(-lam * lags))                                 # Eq 4

W_shift = normalise(np.exp(-lam * (lags - tau_d))
                    * theta(lags - tau_d))                             # Eq 5

W_band = normalise(np.exp(-lam * (lags - tau_d))
                   * theta(lags - tau_d) * theta(tau - lags))          # Eq 6

W_gamma = normalise((lags ** alpha) * np.exp(-lam * lags)
                    * theta(tau - lags))                               # Eq 7

W_pow = normalise((lags ** beta - (lags - 1.0) ** beta)
                  / gamma_fn(beta + 1.0))                              # Eq 8


# ------------------------------------- 5. Discrete convolution (Eq. 3 sum)
def apply_memory(f_hist, weights):
    """x[n] - x[0] = sum_{l=1}^{L} M[l] f[n-l], evaluated after the burn-in."""
    out = np.zeros(n_steps)
    for n in range(n_steps):
        m = n + burn_in
        past = f_hist[m - L:m][::-1]   # past[i] is the state at lag i+1
        out[n] = np.dot(weights, past)
    return out


resp_memoryless = apply_memory(f_full, W_memoryless)
resp_exp = apply_memory(f_full, W_exp)
resp_shift = apply_memory(f_full, W_shift)
resp_band = apply_memory(f_full, W_band)
resp_gamma = apply_memory(f_full, W_gamma)
resp_pow = apply_memory(f_full, W_pow)

# ---------------------------------------------------------------- 6. Plotting
fig, ax = plt.subplots(figsize=(12, 7))

y_top = 45.0

# Drive term f(n) and the memoryless response it produces
ax.bar(months, f_full[burn_in:], color='lightgray', alpha=0.7,
       label=r'Social tension $f(n)$ (drive term of Eq. 3)')
ax.step(months, resp_memoryless, where='mid', color='dimgray', lw=1.5,
        linestyle='--', label='Memoryless baseline (Markovian, one-step lag)')

# The scandal spike is far off the top of the axis; say so rather than clip it
ax.annotate('scandal (engram): %d' % scandal,
            xy=(event_month, y_top), xytext=(event_month + 1.5, y_top - 4),
            fontsize=9, color='black',
            arrowprops=dict(arrowstyle='-|>', color='black', lw=1.2))

# Event and threshold markers
ax.axvline(event_month, color='black', linestyle='--', alpha=0.8,
           label='Major scandal breaks (month %d)' % event_month)
ax.axvline(event_month + tau_d, color='orange', linestyle=':', alpha=0.8,
           label=r'Latency $\tau_d$ ends (month %d)' % (event_month + tau_d))
ax.axvline(event_month + tau, color='red', linestyle='-.', alpha=0.8,
           label=r'Memory length $\tau$ cutoff (month %d)' % (event_month + tau))

# Memory trajectories
ax.plot(months, resp_exp, label='Exponential (24-hour news cycle)',
        lw=3, color='#1f77b4')
ax.plot(months, resp_shift, label='Shifted latency (official report released)',
        lw=3, color='#ff7f0e')
ax.plot(months, resp_band, label='Bandpass (election-cycle weaponization)',
        lw=3, color='#2ca02c', linestyle='--')
ax.plot(months, resp_gamma, label='Gamma (slow-burn grassroots movement)',
        lw=3, color='#d62728')
ax.plot(months, resp_pow, label='Power law (generational grievance)',
        lw=3, color='#9467bd')

# Formatting
ax.set_xlim(0, 80)
ax.set_ylim(0, y_top)
ax.set_xlabel('Time (months)', fontsize=12)
ax.set_ylabel(r'Public outrage level $x(n)$', fontsize=12)
ax.set_title('Impact of memory type on social dynamics', fontsize=14, pad=15)
ax.grid(True, alpha=0.3)
ax.legend(loc='upper right', framealpha=0.9, fontsize=9)

plt.tight_layout()
plt.savefig('scandal_discrete.eps', format='eps', bbox_inches='tight')
plt.savefig('scandal_discrete.png', format='png', dpi=300, bbox_inches='tight')
plt.show()
