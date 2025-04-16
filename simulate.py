import json
import random
import math
import numpy as np
from numba import njit, prange
import time

# Configuration options
n_particles = 1000
n_steps = 600
step_size = 10
M = 4  # Mass of the black hole
save_accuracy = "low"  # Accuracy/number of decimals when saving to npz
initial_conditions = "star"  # star, stream, disk
star_radius = 50
star_offset_x = 50
star_offset_y = 50

# Initialize an empty array for our particles
particles = np.zeros(
    n_particles,
    dtype=[("r", "f8"), ("p", "f8"), ("phi", "f8"), ("L", "f8"), ("dead", "b1")],
)


def generate_stream():
    for i in range(n_particles):
        r = 10 * M + i / 5
        L = math.sqrt((M * r**2) / (r - 3 * M)) - 1
        particles[i] = (r, 0, random.random(), L, False)


# Generate random points within a circle at a certain position away from the black hole
def generate_star():
    for i in range(n_particles):
        # Generate random point around star with offset, directly in polar coordinates
        r = math.sqrt(random.random()) * star_radius
        theta = random.random() * 2 * math.pi

        x = star_offset_x + r * math.cos(theta)
        y = star_offset_y + r * math.sin(theta)

        # Convert to polar
        r = math.hypot(x, y)
        phi = math.atan2(y, x)

        L = math.sqrt(abs((M * r**2) / (r - 3 * M))) - 1
        particles[i] = (r, 0, phi, L, False)


def initialize_particles():
    if initial_conditions == "star":
        generate_star()
    else:
        generate_stream()


@njit
def rk4_system(f, y, h):
    k1 = f(y[0], y[1], y[2], y[3])
    k2 = f(y[0] + h / 2 * k1[0], y[1] + h / 2 * k1[1], y[2] + h / 2 * k1[2], y[3])
    k3 = f(y[0] + h / 2 * k2[0], y[1] + h / 2 * k2[1], y[2] + h / 2 * k2[2], y[3])
    k4 = f(y[0] + h * k3[0], y[1] + h * k3[1], y[2] + h * k3[2], y[3])

    return (
        y[0] + h / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]),
        y[1] + h / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]),
        y[2] + h / 6 * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2]),
    )


@njit
def particle_dynamics(r, phi, p, L):
    dr = p
    dp = -M / (r**2) + L**2 * (1 / (r**3) - 3 * M / (r**4))
    dphi = L / r**2

    return np.array([dr, dphi, dp])


@njit(parallel=True)
def update_particles(particles):
    for i in prange(len(particles)):
        if not particles[i]["dead"]:
            y = np.array(
                [
                    particles[i]["r"],
                    particles[i]["phi"],
                    particles[i]["p"],
                    particles[i]["L"],
                ]
            )
            r, phi, p = rk4_system(particle_dynamics, y, step_size)

            if r < 2 * M:
                particles[i]["dead"] = True
            else:
                particles[i]["r"] = r
                particles[i]["phi"] = phi
                particles[i]["p"] = p


def main():
    initialize_particles()
    states = []

    start_time = time.time()

    for _ in range(n_steps):
        update_particles(particles)
        state = [
            {
                "r": particles[i]["r"],
                "phi": particles[i]["phi"],
            }
            for i in range(n_particles)
        ]
        states.append(state)

    end_time = time.time()

    print(f"Simulation took {round(end_time - start_time, 3)} seconds")

    with open("out.json", "w") as f:
        json.dump({"M": M, "n_steps": n_steps, "n_particles": n_particles}, f)

    nDigits = (
        save_accuracy == "high"
        and 8
        or save_accuracy == "medium"
        and 5
        or save_accuracy == "low"
        and 4
    )
    np.savez_compressed(
        "out.npz",
        r=[[round(particle["r"], nDigits) for particle in state] for state in states],
        phi=[
            [round(particle["phi"], nDigits) for particle in state] for state in states
        ],
    )


if __name__ == "__main__":
    main()
