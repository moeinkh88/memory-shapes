"""
Figure 4 -- Discrete memory kernels.

Discrete-time version: the step size Delta t has been removed everywhere.
The kernel depends only on the integer lag

    l = n - j   (age of the past state, in steps),

so M is written M[l] rather than M(t_n, t_j).

Axis convention: lag increases to the LEFT, so the present sits at the right
edge, as in the original continuous figure ("time before present").

Kernels (manuscript numbering):
  Eq 4  exponential          M[l] = S exp(-lam l) Theta(tau - l)
  Eq 5  shifted Heaviside    M[l] = S exp(-lam (l - tau_d)) Theta(l - tau_d)
  Eq 6  bandpass Heaviside   M[l] = S exp(-lam (l - tau_d)) Theta(l - tau_d) Theta(tau - l)
  Eq 7  Gamma                M[l] = S l^alpha exp(-lam l) Theta(tau - l)
  Eq 8  fractional power law M[l] = [l^beta - (l-1)^beta] / Gamma(beta + 1)

Eq 8 is the product-rectangle weight with Delta t = 1. Dropping the
(Delta t)^(beta-1) prefactor matters: in the continuous script that factor was
0.01^(-0.5) = 10, so the power-law curve was scaled by the plotting resolution
rather than by beta, which is why the y-axis had to be capped at 2.5.

Lag 0: the kernels are drawn from l = 0 so that the functional form is visible
at zero age -- S for the exponential, 0 for the delayed kernels, which is the
"organic latency" of the Gamma kernel (Section 4.2). Note that Eq. (3) sums
over j = 0, ..., n-1, so the dynamics only ever evaluate l >= 1; l = 0 is shown
as the limiting value of the kernel, not as a weight the system applies. The
fractional weights of Eq. (8) are defined only for l >= 1 (they are increments
between consecutive lags), so that curve starts at l = 1; setting it to 0 at
l = 0 would wrongly suggest the power law ignores the recent past, when in fact
it weights it most heavily.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma as gamma_fn

# ---------------------------------------------------------------- 1. Lag axis
L = 12                             # longest lag shown (in steps)
lags = np.arange(0, L + 1)         # l = 0, 1, ..., L
lags_pow = np.arange(1, L + 1)     # Eq 8 weights exist only for l >= 1

# ------------------------------------------------------ 2. Kernel parameters
S = 1.0        # memory strength (amplitude)
lam = 0.5      # decay rate per step
tau = 7        # memory length, in steps
tau_d = 2      # latency / delay, in steps
alpha = 1.3    # Gamma shape parameter
beta = 0.5     # fractional order, 0 < beta < 1


def theta(x):
    """Heaviside step with Theta(0) = 1, as defined in the manuscript."""
    return np.heaviside(x, 1.0)


# ------------------------------------------------------- 3. Discrete kernels
M_exp = S * np.exp(-lam * lags) * theta(tau - lags)                      # Eq 4

M_shift = S * np.exp(-lam * (lags - tau_d)) * theta(lags - tau_d)        # Eq 5

M_band = (S * np.exp(-lam * (lags - tau_d))
          * theta(lags - tau_d) * theta(tau - lags))                     # Eq 6

M_gamma = S * (lags ** alpha) * np.exp(-lam * lags) * theta(tau - lags)  # Eq 7

M_pow = ((lags_pow ** beta - (lags_pow - 1.0) ** beta)
         / gamma_fn(beta + 1.0))                                         # Eq 8

# ---------------------------------------------------------------- 4. Plotting
fig, ax = plt.subplots(figsize=(10, 6))

style = dict(lw=1.2, alpha=0.55)   # thin guide lines: only integer lags exist

# Eq 6 is Eq 5 multiplied by the length cutoff, so the two coincide for
# tau_d <= l <= tau. The bandpass is drawn dashed with open markers so that
# both stay visible in the overlapping range.
series = [
    (lags,     M_exp,   'Exponential (Eq 4)',          '#1f77b4', 'o', '-',  True),
    (lags,     M_shift, 'Shifted Heaviside (Eq 5)',    '#ff7f0e', 's', '-',  True),
    (lags,     M_band,  'Bandpass Heaviside (Eq 6)',   '#2ca02c', '^', '--', False),
    (lags,     M_gamma, 'Gamma (Eq 7)',                '#d62728', 'D', '-',  True),
    (lags_pow, M_pow,   'Fractional power law (Eq 8)', '#9467bd', 'v', '-',  True),
]

for x, values, label, colour, marker, linestyle, filled in series:
    ax.plot(x, values, color=colour, marker=marker,
            ms=7 if filled else 11,
            markerfacecolor=colour if filled else 'none',
            markeredgewidth=1.6, linestyle=linestyle,
            label=label, **style)

# The fractional kernel has no weight at l = 0: the underlying continuous
# kernel (t_n - t_j)^(beta-1) diverges there. The guide below traces that
# approach between l = 1 and l = 0, scaled to meet the Eq. (8) weight at
# l = 1, and runs off the top of the axis. No marker is drawn at l = 0,
# since no finite value exists to plot.
l_fine = np.linspace(0.02, 1.0, 400)
ax.plot(l_fine, M_pow[0] * l_fine ** (beta - 1.0),
        color='#9467bd', **style)

# Threshold markers
ax.axvline(tau_d, color='gray', linestyle=':',
           label=r'Latency $\tau_d$ = %d steps' % tau_d)
ax.axvline(tau, color='gray', linestyle='-.',
           label=r'Memory length $\tau$ = %d steps' % tau)

# Formatting: lag increases to the left, present at the right edge
ax.set_xticks(lags)
ax.set_xlim(L + 0.5, -0.5)
ax.set_ylim(-0.05, 1.3)
ax.set_xlabel(r'Time before present: lag $\ell = n - j$ (steps)', fontsize=12)
ax.set_ylabel(r'Memory weight $M^{(k)}(n,j)$', fontsize=12)
ax.set_title('Different types of memory kernels', fontsize=14, pad=15)
ax.grid(True, alpha=0.3)
ax.legend(loc='upper left', framealpha=0.9)

plt.tight_layout()
plt.savefig('memory_kernels_discrete.eps', format='eps', bbox_inches='tight')
plt.savefig('memory_kernels_discrete.png', format='png', dpi=300,
            bbox_inches='tight')
plt.show()
