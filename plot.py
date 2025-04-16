"""
NEXT:
SMOOTH NEWTONIAN ENHANCED RELATIVISTIC PARTICLE STEP APPROXIMATION METHOD
for every step inbetween some delta steps, use newtonian physics to efficiently approximate the next calculated position for optimization
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
from matplotlib.patches import Circle

# Configuration
frames_per_iteration = (
    1  # Larger number = faster simulation. More frames are skipper per render.
)

with open("out.json", "r") as f:
    sim = json.load(f)

M = sim["M"]
n_steps = sim["n_steps"]
n_particles = sim["n_particles"]

plt.style.use("dark_background")

fig, ax = plt.subplots()

plt.xlim(-200, 200)
plt.ylim(-200, 200)

ax.set_aspect("equal", adjustable="box")

scat = ax.scatter([], [], 0.1, "white")
circle = Circle((0, 0), radius=2 * M, color="red", fill=False)
ax.add_patch(circle)

data = np.load("out.npz")
rs = data["r"]
phis = data["phi"]

rs = rs.reshape(n_steps, n_particles)
phis = phis.reshape(n_steps, n_particles)


def update(frame):
    frame = frame * frames_per_iteration

    if frame >= n_steps:
        frame = 0

    r = rs[frame]
    phi = phis[frame]

    x = r * np.cos(phi)
    y = r * np.sin(phi)

    data = np.stack([x, y]).T
    scat.set_offsets(data)


ani = animation.FuncAnimation(
    fig=fig, func=update, frames=n_steps // frames_per_iteration, interval=0
)
ani.save("out.gif", fps=30)
plt.show()
