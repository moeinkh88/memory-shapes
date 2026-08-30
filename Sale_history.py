import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma

# 1. Simulation Parameters
n_steps = 100           # Total months to simulate
t = np.arange(n_steps)  # Discrete time steps
dt = 1.0                # Step size

# 2. Create the "Engrams" (Past Experience / Sales Data)
# Baseline sales of 10, but a massive viral success at Month 20
sales_history = np.ones(n_steps) * 10
sales_history[20] = 150 

# 3. Base Memory Parameters
S = 1.0         # Memory Strength
lam = 0.15      # Exponential decay rate
tau = 25.0      # Memory length (cutoff)
tau_d = 10.0    # Delay / Latency period (10 months)
alpha = 2.0     # Gamma distribution shape parameter
beta = 0.5      # Power-law fractional order (0 < beta < 1)

def calculate_memory_state(events, kernel_type):
    """Calculates the system state at each time step using discrete convolution."""
    state = np.zeros(n_steps)
    
    # Start from step 1 to avoid looking back at time 0
    for n in range(1, n_steps):
        tj = np.arange(n)   # Past time steps up to n
        age = n - tj        # Time before present (tn - tj)
        
        # Apply the specific mnemonic function (Memory Kernels)
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
        
        # Calculate current state: sum of past events weighted by memory
        # This represents Eq. 3 from the mathematical model
        state[n] = np.sum(weights * events[:n])
        
    return state

# 4. Run the simulations
state_exp      = calculate_memory_state(sales_history, 'exponential')
state_shift    = calculate_memory_state(sales_history, 'shifted')
state_bandpass = calculate_memory_state(sales_history, 'bandpass')
state_gamma    = calculate_memory_state(sales_history, 'gamma')
state_powerlaw = calculate_memory_state(sales_history, 'powerlaw')

# 5. Plotting the System's Response over Time
fig, ax = plt.subplots(figsize=(12, 7))

# Plot the underlying event (The Viral Spike)
ax.bar(t, sales_history, color='lightgray', alpha=0.5, label='Past Sales (Engrams)')
ax.axvline(x=20, color='black', linestyle='--', alpha=0.7, label='Viral Success (Month 20)')

# Plot the memory trajectories
ax.plot(t, state_exp, label='Exponential (Immediate Reaction)', lw=3, color='#1f77b4')
ax.plot(t, state_shift, label='Shifted Latency (Delayed Reaction)', lw=3, color='#ff7f0e')
ax.plot(t, state_bandpass, label='Bandpass (Strict Window)', lw=3, color='#2ca02c', linestyle='--')
ax.plot(t, state_gamma, label='Gamma (Nostalgic Peak)', lw=3, color='#d62728', linestyle=':')
ax.plot(t, state_powerlaw, label='Power-Law (Infinite Drag)', lw=3, color='#9467bd')

# Formatting
ax.set_xlim(0, 80)
ax.set_ylim(0, 660)
ax.set_xlabel('Time (Months)', fontsize=12)
ax.set_ylabel('System State (Current Marketing Budget)', fontsize=12)
ax.set_title('Impact of Different Memory Types on System Dynamics', fontsize=14, pad=15)
ax.grid(True, alpha=0.3)
ax.legend(loc='upper right', framealpha=0.9)

plt.tight_layout()
plt.show()