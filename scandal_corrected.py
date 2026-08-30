import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma

# 1. Simulation Parameters
n_steps = 100           
t = np.arange(n_steps)  
dt = 1.0                

# 2. The "Engrams" (Everyday tension = 10, Major Scandal = 150)
social_events = np.ones(n_steps) * 10
event_month = 20
social_events[event_month] = 150 

# 3. Scaled Memory Parameters (Stretched for 100 months)
lam = 0.15      # Exponential decay rate
tau_d = 10      # Latency / Delay period (10 months)
tau = 25        # Length Cutoff / Forgetting Time (25 months AFTER the event)
alpha = 2.0     # Gamma shape parameter
beta = 0.5      # Power-law fractional order

# Pre-calculate event ages (0 to 99)
ages = np.arange(n_steps)

# 4. Define and Normalize the Memory Kernels
# NEW: Memoryless (Markovian Baseline)
W_memoryless = np.zeros(n_steps)
W_memoryless[0] = 1.0  # 100% weight to the present moment, 0% to the past

W_exp = np.exp(-lam * ages)
W_exp /= np.sum(W_exp)

W_shift = np.exp(-lam * (ages - tau_d)) * np.heaviside(ages - tau_d, 1)
W_shift /= np.sum(W_shift)

W_band = np.exp(-lam * (ages - tau_d)) * np.heaviside(ages - tau_d, 1) * np.heaviside(tau - ages, 1)
W_band /= np.sum(W_band)

W_gamma = (ages**alpha) * np.exp(-lam * ages) * np.heaviside(tau - ages, 1)
W_gamma /= np.sum(W_gamma)

term1 = (ages / dt)**beta
term2 = (np.maximum(ages / dt - 1, 0))**beta
W_pow = (dt**(beta - 1)) * ((term1 - term2) / gamma(beta + 1))
W_pow /= np.sum(W_pow)

# 5. Apply Memory to the Events (Discrete Convolution)
def apply_memory(events, weights):
    response = np.zeros(n_steps)
    for n in range(n_steps):
        # Sum of past events multiplied by the memory weight for their specific age
        response[n] = np.sum(events[:n+1] * weights[n::-1])
    return response

resp_memoryless = apply_memory(social_events, W_memoryless)
resp_exp = apply_memory(social_events, W_exp)
resp_shift = apply_memory(social_events, W_shift)
resp_band = apply_memory(social_events, W_band)
resp_gamma = apply_memory(social_events, W_gamma)
resp_pow = apply_memory(social_events, W_pow)

# 6. Plotting
fig, ax = plt.subplots(figsize=(12, 7))

# Plot Baseline and Event
ax.bar(t, social_events, color='lightgray', alpha=0.5, label='Daily Social Tension (Memoryless baseline)')

# Event and Threshold Vertical Lines
ax.axvline(x=event_month, color='black', linestyle='--', alpha=0.8, label='Major Scandal Breaks (Month 20)')
ax.axvline(x=event_month + tau_d, color='orange', linestyle=':', alpha=0.8, label=r'Latency Ends (Month 30)')
ax.axvline(x=event_month + tau, color='red', linestyle='-.', alpha=0.8, label=r'Forgetting Time / Cutoff $\tau$ (Month 45)')

# Memory Trajectories
ax.plot(t, resp_exp, label='Exponential (24-Hour News Cycle)', lw=3, color='#1f77b4')
ax.plot(t, resp_shift, label='Shifted Latency (Official Report Released)', lw=3, color='#ff7f0e')
ax.plot(t, resp_band, label='Bandpass (Election Cycle Weaponization)', lw=3, color='#2ca02c')
ax.plot(t, resp_gamma, label='Gamma (Slow-Burn Grassroots Movement)', lw=3, color='#d62728')
ax.plot(t, resp_pow, label='Power-Law (Generational Grievance)', lw=3, color='#9467bd')

# Formatting
ax.set_xlim(0, 80)
ax.set_ylim(0, 40) # Adjusted slightly so the memoryless spike is visible but doesn't squash the other curves too much
ax.set_xlabel('Time (Months)', fontsize=12)
ax.set_ylabel('Public Outrage Level', fontsize=12)
ax.set_title('Impact of Memory Types on Social Dynamics', fontsize=14, pad=15)
ax.grid(True, alpha=0.3)
ax.legend(loc='upper right', framealpha=0.9, fontsize=9)

plt.tight_layout()
plt.show()