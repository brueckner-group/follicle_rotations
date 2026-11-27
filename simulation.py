"""
@author: Leo Flehinghaus
Date: 27.11.2026

Simulation of active particles with spring-like
interactions and polarity alignment on a sphere.
Allows neighbour exchange by repeated voronoi tessellations.
All cells connected with the same rest length l0.
"""

import numpy as np
from scipy.spatial import SphericalVoronoi
from collections import defaultdict
from datetime import datetime
import csv


# ── Initialisation ────────────────────────────────────────────────────────────

def initialise_positions(N, R):
    """Initialise particle positions on the sphere (N×3 array)."""
    indices = np.arange(0, N, dtype=float) + 0.5
    phi     = np.arccos(1 - 2 * indices / N)
    theta   = np.pi * (1 + 5**0.5) * indices
    pts     = np.column_stack([np.cos(theta) * np.sin(phi),
                               np.sin(theta) * np.sin(phi),
                               np.cos(phi)])
    return R * pts


def initialise_polarities(N, positions, R, initial_polarisation='random'):
    """
    Initialise tangent polarities (N×3 array).
    Vectors are projected onto the tangent plane right away.
    """
    if initial_polarisation == 'random':
        pol = np.random.normal(0, 1, size=(N, 3))
    else:
        pol = np.ones((N, 3))
    # Project onto tangent plane of the spherec 
    pol -= (np.sum(pol * positions, axis=1) / R**2)[:, None] * positions
    pol /= np.linalg.norm(pol, axis=1)[:, None] # Possible division by zero but unlikely
    return pol


# ── Neighbour detection ───────────────────────────────────────────────────────

def find_neighbours(positions, R, threshold=1e-3):
    """
    Compute neighbours via Voronoi tessellation.
    Returns list-of-lists, or None on failure.
    """
    try:
        sv = SphericalVoronoi(positions, radius=R, center=[0, 0, 0],
                              threshold=threshold)
        sv.sort_vertices_of_regions()

        region_to_indices = defaultdict(list)
        for idx, region_list in enumerate(sv.regions):
            for region in region_list:
                region_to_indices[region].append(idx)

        neighbours = []
        for i in range(len(positions)):
            nbr_set = set()
            for region in sv.regions[i]:
                for k in region_to_indices[region]:
                    if k != i:
                        nbr_set.add(k)
            neighbours.append(list(nbr_set))
        return neighbours

    except ValueError:
        print("Voronoi tessellation failed – keeping previous neighbours.")
        return None


# ── Forces ───────────────────────────────────────────────────────

def calculate_forces(positions, i_idx, j_idx, epsilon_attraction, epsilon_repulsion, l0):
    """
    Compute elastic forces between neighbouring particles.
    Vectorised over all pairs simultaneously.
    """
    N      = len(positions)
    forces = np.zeros((N, 3))

    # Calculate all distances between neighbours
    # Starts with the neighbours from cell 0 
    # No distances are double counted
    dvec  = positions[j_idx] - positions[i_idx]           # Shape is (NumberNeighbours, 3)
    dist  = np.linalg.norm(dvec, axis=1, keepdims=True)   # (NumberNeighbours, 1)
    unit  = dvec / dist                                   # (NumberNeighbours, 3)

    stretch   = dist.squeeze() - l0                       # (NumberNeighbours,) (simple array containing all distances)
    eps       = np.where(stretch < 0, epsilon_repulsion, epsilon_attraction) # Define regions of repulsion and attraction
    force_vec = (eps * stretch)[:, None] * unit           # (NumberNeighbours, 3) (eps * stretch = (N,)) (eps*stretch[:,None] = (N,1))
    # Add at all neighbour indices the corresponding forces
    np.add.at(forces, i_idx,  force_vec)
    np.add.at(forces, j_idx, -force_vec)
    return forces


# ── Torques ──────────────────────────────────────────────────────

def calculate_torques(positions, polarities, i_idx, j_idx, alignment_strength,
                      noise_amplitude, dt, R):
    #
    #Compute alignment torques on polarities.
    #Vectorised over neighbours for each particle.
    #
    N      = len(polarities)
    torques = np.zeros((N, 3))

    pos_i  = positions[i_idx]                                        # (P, 3)
    pos_j  = positions[j_idx]                                        # New
    pol_i  = polarities[i_idx]                                       # (P, 3)
    pol_j  = polarities[j_idx]                                       # (P, 3)
  
    # i sees (pol_i × pol_j) · pos_i / |pos_i|
    # j sees (pol_j × pol_i) · pos_j / |pos_j| = -cross_ij · pos_j / |pos_j|
    cross_ij  = np.cross(pol_i, pol_j)                               # (P, 3)
    phidot_i  = np.sum(cross_ij * pos_i, axis=1) / R  #  (P,)
    phidot_j  = np.sum(-cross_ij * pos_j, axis=1) / R #  (P,)

    # Sum contributions per particle i
    phidot = np.zeros(N)
    np.add.at(phidot, i_idx, phidot_i)
    np.add.at(phidot, j_idx, phidot_j)
    phidot *= alignment_strength 

    # Scalar noise per particle
    noise   = noise_amplitude * np.random.normal(0, 1, size=N) / np.sqrt(dt)
    phidot += noise

    # Torque vectors
    cross_rp   = np.cross(positions, polarities)                # (N, 3)
    torques    = (phidot / R)[:, None] * cross_rp               # (N, 3)
    return torques


# ── Main simulation ───────────────────────────────────────────────────────────

def simulation(N, rho, v0, epsilon_attraction, epsilon_repulsion,
               alignment_strength, noise_amplitude,
               dt, samples, steps, save_dir,
               direct_return=True, initial_polarisation='random',
               steps_for_voronoi = 10, save_transitions = False,
               init_steps = 0):
    """Propagate active particles in time using forward-Euler integration."""
    # Save the simulation with a current time stamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename  = save_dir + f"simulation_{timestamp}"

    # Initialise the cells
    R          = np.sqrt(N / (4 * np.pi * rho))
    positions  = initialise_positions(N, R)
    polarities = initialise_polarities(N, positions, R,
                                       initial_polarisation=initial_polarisation)
    # Save simulation parameters
    info_headers = ['N', 'rho', 'v0', 'epsilon_attraction', 'epsilon_repulsion',
                    'beta', 'sigma', 'dt', 'samples', 'steps']
    info_file = save_dir + f"information_{timestamp}"
    with open(info_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(info_headers)
        writer.writerow([N, rho, v0, epsilon_attraction, epsilon_repulsion,
                         alignment_strength, noise_amplitude, dt, samples, steps])
    print(f"Simulation details saved to '{info_file}'.")

    neighbours = find_neighbours(positions, R)
    old_neighbours = neighbours
    i_idx, j_idx = zip(*[(i, j)
                      for i, nbrs in enumerate(neighbours)
                      for j in nbrs if j > i])
    i_idx = np.array(i_idx)
    j_idx = np.array(j_idx)
    dvec  = positions[j_idx] - positions[i_idx]     
    dist  = np.linalg.norm(dvec, axis=1)
    l0 = np.mean(dist) 

    if save_transitions:
        transition_file = save_dir + f"transitions_{timestamp}"
        T1_transition_list = []
        junction_list = []

    all_data = []
    # Simulation
    for sample in range(init_steps):
        if (sample % 1000 == 0 ):
            print(f"Sample {sample + 1} of {init_steps}")   
        if (sample % steps_for_voronoi == 0): 
            new_neighbours = find_neighbours(positions, R)
            if new_neighbours is not None:    
                neighbours = new_neighbours
                i_idx, j_idx = zip(*[(i, j)
                    for i, nbrs in enumerate(neighbours)
                    for j in nbrs if j > i]) # New
                i_idx = np.array(i_idx)
                j_idx = np.array(j_idx)
        forces  = calculate_forces(positions, i_idx,j_idx,
                                    epsilon_attraction, epsilon_repulsion, l0)
        torques = calculate_torques(positions, polarities, i_idx,j_idx,
                                    alignment_strength, noise_amplitude, dt, R)
        # Integrate polarities
        polarities += dt * torques
        polarities /= np.linalg.norm(polarities, axis=1)[:, None]
        # Integrate positions (project force onto tangent plane)
        total_forces = v0 * polarities + forces
        positions   += dt * (total_forces
                                - np.sum(total_forces * positions, axis=1)[:, None]
                                * positions / R**2)
        # Project positions back onto sphere
        positions /= np.linalg.norm(positions, axis=1)[:, None]
        positions *= R
        # Project polarities back onto tangent plane
        polarities -= (np.sum(polarities * positions, axis=1)[:, None]
                        / R**2 * positions)
        polarities /= np.linalg.norm(polarities, axis=1)[:, None]

    for sample in range(samples):
        if (sample % 100 == 0 ):
            print(f"Sample {sample + 1} of {samples}")   

        for step in range(steps):
            if (step % steps_for_voronoi == 0): 
                new_neighbours = find_neighbours(positions, R)
                if new_neighbours is not None:    
                    neighbours = new_neighbours
                    i_idx, j_idx = zip(*[(i, j)
                      for i, nbrs in enumerate(neighbours)
                      for j in nbrs if j > i]) # New
                    i_idx = np.array(i_idx)
                    j_idx = np.array(j_idx)

            forces  = calculate_forces(positions, i_idx,j_idx,
                                       epsilon_attraction, epsilon_repulsion, l0)
            torques = calculate_torques(positions, polarities, i_idx,j_idx,
                                        alignment_strength, noise_amplitude, dt, R)

            # Integrate polarities 
            polarities += dt * torques
            polarities /= np.linalg.norm(polarities, axis=1)[:, None]

            # Integrate positions (project force onto tangent plane)
            total_forces = v0 * polarities + forces
            positions   += dt * (total_forces
                                 - np.sum(total_forces * positions, axis=1)[:, None]
                                 * positions / R**2)

            # Project positions back onto sphere
            positions /= np.linalg.norm(positions, axis=1)[:, None]
            positions *= R

            # Project polarities back onto tangent plane
            polarities -= (np.sum(polarities * positions, axis=1)[:, None]
                           / R**2 * positions)
            polarities /= np.linalg.norm(polarities, axis=1)[:, None]

        if save_transitions:
            transitions = 0
            for i in range(len(neighbours)):
                # here we assume that a cell can only exchange one neighbour at once
                if neighbours[i] != old_neighbours[i]:
                    transitions += 1/4
            T1_transition_list.append(transitions)# normalisation later for debugging / len(i_idx))
            junction_list.append(len(i_idx))
            old_neighbours = neighbours
        all_data.append(np.hstack((positions, polarities)))

    np.save(filename, all_data)
    if save_transitions:
        np.save(transition_file, T1_transition_list)
    if direct_return:
        if save_transitions:
            return np.array(all_data), T1_transition_list, junction_list
        else:
            return np.array(all_data)
        

if __name__ == "__main__":
    N = 400 
    rho = 0.008 
    v0 = 1.0
    epsilon_attraction = 0.8 
    epsilon_repulsion  = 0.8
    alignment_strength = 0.1
    noise_amplitude    = 0.7 
    dt      = 1e-2
    samples = 1000
    steps   = 500 
    save_dir = '.'
    simulation(N, rho, v0, epsilon_attraction, epsilon_repulsion,
               alignment_strength, noise_amplitude,
               dt, samples, steps, save_dir,
               direct_return=False, initial_polarisation='random',
               steps_for_voronoi = 1, save_transitions = False,
               init_steps = samples * steps)