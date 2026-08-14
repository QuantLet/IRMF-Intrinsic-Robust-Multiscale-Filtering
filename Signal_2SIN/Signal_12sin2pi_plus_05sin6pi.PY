import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------
# Mixed-frequency sine signal
# X(t) = 1.2 sin(2πt) + 0.5 sin(6πt)
# -------------------------------------------------

# Number of observations
n = 2000

# Time grid
t = np.linspace(0, 1, n)

# Signal
X = 1.2 * np.sin(2 * np.pi * t) \
  + 0.5 * np.sin(6 * np.pi * t)

# -------------------------------------------------
# Plot
# -------------------------------------------------

plt.figure(figsize=(10, 3.8), facecolor='none')

plt.plot(
    t,
    X,
    color='#1f77b4',
    linewidth=4
)

plt.xlabel(r'$t$', fontsize=20)
plt.ylabel(r'$X_t$', fontsize=20)

plt.xlim(0, 1)

ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_facecolor('none')

plt.xticks(fontsize=18)
plt.yticks(fontsize=18)

plt.tight_layout()

plt.savefig(
    "MixedFrequencySine_n2000.png",
    dpi=400,
    transparent=True,
    bbox_inches="tight"
)

plt.show()
