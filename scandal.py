import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma

# 1. Simulation Parameters
n_steps = 100           # Total months to simulate
t = np.arange(n_steps)  # Discrete time steps
dt = 1.0                # Step size

# 2. Create the "Engrams" (The Social Event / Scandal)
# Baseline social tension of 10, but a massive scandal breaks at Month 20
social_events = np.ones(n_steps) * 10
social_events[20] = 150 

# 3. Base Memory Parameters
S = 1.0         # Memory Strength
lam = 0.15      # Exponential decay rate (lambda)
tau = 25.0      # Memory length (cutoff)
tau_d = 10.0    # Delay / Latency period (10 months)
alpha = 2.0     # Gamma distribution shape parameter
beta = 0.5      # Power-law fractional order (0 < beta < 1)

def calculate_public_response(events, kernel_type):
    """Calculates the system state (public response) at each time step using discrete convolution."""
    response = np.zeros(n_steps)
    
    # Start from step 1 to avoid looking back at time 0
    for n in range(1, n_steps):
        tj = np.arange(n)   # Past time steps up to n
        age = n - tj        # Time before present (tn - tj)
        
        # Apply the specific collective memory functions (Kernels)
        if kernel_type == 'exponential':
            weights = S * np.exp(-lam * age)
            
        elif kernel_type == 'shifted':
            weights = S * np.exp(-lam * (age - tau_d)) * np.heaviside(age - tau_d, 1)
            
        elif kernel_type == 'bandpass':
            weights = S * np.exp(-lam * (age - tau_d)) * np.heaviside(age - tau_d, 1) * np.heaviside(tau - age, 1)
            
        elif kernel_type == 'gamma':
            # Normalized slightly to fit the visual scale of the others
            weights = (S * (age**alpha) * np.exp(-lam * age) * np.heaviside(tau - age, 1)) / 10 
            
        elif kernel_type == 'powerlaw':
            term1 = (age / dt)**beta
            term2 = (np.maximum(age / dt - 1, 0))**beta
            weights = (dt**(beta - 1)) * ((term1 - term2) / gamma(beta + 1))
        
        # Calculate current state: sum of past events weighted by collective memory
        response[n] = np.sum(weights * events[:n])
        
    return response

# 4. Run the simulations
resp_exp      = calculate_public_response(social_events, 'exponential')
resp_shift    = calculate_public_response(social_events, 'shifted')
resp_bandpass = calculate_public_response(social_events, 'bandpass')
resp_gamma    = calculate_public_response(social_events, 'gamma')
resp_powerlaw = calculate_public_response(social_events, 'powerlaw')

# 5. Plotting the Society's Response over Time
fig, ax = plt.subplots(figsize=(12, 7))

# Plot the underlying event (The Scandal)
ax.bar(t, social_events, color='lightgray', alpha=0.5, label='Social Events (Engrams)')
ax.axvline(x=20, color='black', linestyle='--', alpha=0.7, label='Major Scandal (Month 20)')

# Plot the social memory trajectories
ax.plot(t, resp_exp, label='Exponential (24-Hour News Cycle)', lw=3, color='#1f77b4')
ax.plot(t, resp_shift, label='Shifted Latency (Official Report Released)', lw=3, color='#ff7f0e')
ax.plot(t, resp_bandpass, label='Bandpass (Election Cycle Weaponization)', lw=3, color='#2ca02c', linestyle='--')
ax.plot(t, resp_gamma, label='Gamma (Slow-Burn Grassroots Movement)', lw=3, color='#d62728', linestyle=':')
ax.plot(t, resp_powerlaw, label='Power-Law (Generational Grievance)', lw=3, color='#9467bd')

# Formatting
ax.set_xlim(0, 100)
ax.set_ylim(0, 800)
ax.set_xlabel('Time (Months)', fontsize=12)
ax.set_ylabel('Public Response / Outrage Level', fontsize=12)
ax.set_title('Impact of Collective Memory Types on Social Dynamics', fontsize=14, pad=15)
ax.grid(True, alpha=0.3)
ax.legend(loc='upper right', framealpha=0.9)

plt.tight_layout()
plt.show()