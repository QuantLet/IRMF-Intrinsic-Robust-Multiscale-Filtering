import numpy as np
import matplotlib.pyplot as plt

# Time grid
n = 1000
t = np.linspace(0, 1, n)

# Signal
x = 1.2 * np.sin(2 * np.pi * t) + 0.5 * np.sin(6 * np.pi * t)

# Plot
plt.figure(figsize=(8, 3.8), facecolor='none')
plt.plot(t, x, color='#1f77b4', linewidth=3)

plt.xlabel(r'$t$', fontsize=18)
plt.ylabel(r'$X_t$', fontsize=18)

plt.xlim(0, 1)

ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_facecolor('none')

plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

plt.tight_layout()
plt.savefig("Signal_12sin2pi_plus_05sin6pi.png",
            dpi=300,
            transparent=True,
            bbox_inches='tight')
plt.show()
