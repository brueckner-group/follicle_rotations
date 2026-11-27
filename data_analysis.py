"""
@author: Leo Flehinghaus
Date: 27.11.2026

Functions for data analysis and plotting
"""

import os
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import ListedColormap
from scipy.spatial import SphericalVoronoi


def list_files_in_folder(folder_path):
    try:
        with os.scandir(folder_path) as it:
            files = [
                entry.name for entry in it
                if entry.name.endswith('.npy') and entry.is_file()
            ]
        return sorted(files)
    except FileNotFoundError:
        print(f"Folder '{folder_path}' not found.")
        return []
    except PermissionError:
        print(f"Permission denied to access '{folder_path}'.")
        return []
    
def calc_ang_vel(x, y, z, vx, vy, vz):
    r = np.column_stack((x, y, z))        # shape (N, 3)
    v = np.column_stack((vx, vy, vz))     # shape (N, 3)
    # r^2 for each particle
    r2 = np.sum(r**2, axis=1)             # shape (N,)
    # Compute M_i = r^2 I - r r^T
    I = np.eye(3)
    M = r2[:, None, None] * I - r[:, :, None] * r[:, None, :]  # shape (N, 3, 3)
    # Mean inertia tensor
    M_mean = M.mean(axis=0)               # shape (3, 3)
    # Angular momentum vectors
    L = np.cross(r, v)                    # shape (N, 3)
    L_mean = L.mean(axis=0)               # shape (3,)
    # Solve M_mean ω = L_mean 
    omega = np.linalg.solve(M_mean, L_mean)
    return omega

def calculate_velocity_flucutations(x,y,z,vx,vy,vz):
    omega = calc_ang_vel(x, y, z, vx, vy, vz)
    v = np.column_stack((vx, vy, vz))
    v_abs = np.sqrt(vx**2 + vy**2 + vz**2)
    r = np.column_stack((x, y, z))
    v_rot = np.cross(omega, r)
    delta_vx = vx - v_rot[:, 0]
    delta_vy = vy - v_rot[:, 1]
    delta_vz = vz - v_rot[:, 2]
    delta_v_abs = np.sqrt(delta_vx**2 + delta_vy**2 + delta_vz**2)
    return delta_v_abs, v_abs# / v_abs

def _edges_from_voronoi(points, R, threshold=1e-3):
    """Return the set of undirected neighbour pairs (i, j) with i < j."""
    sv = SphericalVoronoi(points, R, [0.0, 0.0, 0.0], threshold=threshold)
    sv.sort_vertices_of_regions()
    region_to_cells = defaultdict(list)
    for cell_idx, verts in enumerate(sv.regions):
        for v in verts:
            region_to_cells[v].append(cell_idx)
    edges = set()
    for cells in region_to_cells.values():
        for a in cells:
            for b in cells:
                if a < b:
                    edges.add((a, b))
    return edges

def _debounce(series, min_dwell):
    """Collapse runs shorter than `min_dwell` into the preceding state.
    Example with min_dwell=2:   1 1 0 1 1 1  ->  1 1 1 1 1 1
                                a single 0 flanked by 1s is dropped.
    """
    out = series.copy()
    n = len(out)
    i = 0
    while i < n:
        j = i
        while j < n and out[j] == out[i]:
            j += 1
        run_len = j - i
        if i > 0 and run_len < min_dwell:
            out[i:j] = out[i - 1]
        i = j
    return out

def calculate_T1_rate(positions, acquisition_time, min_dwell=1,
                      return_breakdown=False):
    """
    T1 transition rate per junction per minute on a sphere.

    Counts T1 events from the *edge* time-series:
        T1 events = (total number of edge flips) / 2
    so that simultaneous multi-neighbour exchanges are counted correctly.

    Parameters
    ----------
    positions : ndarray, shape (t, N, 3)
        Cell positions on a sphere, one frame per row of axis 0.
    acquisition_time : float
        Total acquisition time in minutes (the time spanned by the t frames).
    min_dwell : int, default 1
        Minimum number of consecutive frames the new edge state must
        persist to be counted. `min_dwell=1` counts every flip (including
        flicker that reverses on the next frame). Set to 2+ to suppress
        reversals that revert within `min_dwell` frames.
    return_breakdown : bool, default False
        If True, return (rate, info_dict) where info_dict contains the raw
        T1 count, mean junction count, and per-frame junction counts.

    Returns
    -------
    rate : float
        T1 events per junction per minute.
    """
    t_steps, N, _ = positions.shape
    R = float(np.linalg.norm(positions[0, 0]))

    # For every pair that is ever a neighbour, store its 0/1 time-series.
    pair_series = defaultdict(lambda: np.zeros(t_steps, dtype=np.int8))
    junctions_per_frame = np.zeros(t_steps, dtype=int)

    for t in range(t_steps):
        edges = _edges_from_voronoi(positions[t], R)
        junctions_per_frame[t] = len(edges)
        for pair in edges:
            pair_series[pair][t] = 1

    # Optionally drop transient flips before counting.
    if min_dwell > 1:
        for key in list(pair_series.keys()):
            pair_series[key] = _debounce(pair_series[key], min_dwell)

    # Each T1 = 1 edge lost + 1 edge gained = 2 flips in the pair_series.
    flips = 0
    for series in pair_series.values():
        flips += int(np.abs(np.diff(series)).sum())
    t1_events = flips / 2.0

    mean_junctions = float(junctions_per_frame.mean())
    rate = t1_events / (mean_junctions * acquisition_time)

    if return_breakdown:
        return rate, {
            "t1_events": t1_events,
            "mean_junctions": mean_junctions,
            "junctions_per_frame": junctions_per_frame,
        }
    return rate

def plot_trajectories(data_trajectory, betas_trajectory):
    # Plot trajectories of different coupling strengths
    # Input are lists of the trajectory data, and the corresponding beta values

    fig = plt.figure(figsize=(10, 8))
    gs = GridSpec(3, 6,figure=fig,hspace=0.38,wspace=0.35, 
                  top=0.93, bottom=0.06,left=0.06, right=0.97,)

    # Row 1
    view = np.array([1, -1, 1], dtype=float)
    view /= np.linalg.norm(view)
    tmin = 9700

    w3d, h3d = 0.45, 0.45          # width & height of each 3D box (tweak freely)
    origins   = [0.01, 0.32, 0.62]  # left edge of each panel

    trim_bottom = 20
    trim_top  = 45
    shift = int(256 / 3)

    hsv_colors   = plt.get_cmap("hsv", 256)(np.linspace(0, 1, 256))
    hsv_flipped  = hsv_colors[::-1]                           # invert the direction
    hsv_shifted  = np.roll(hsv_flipped, shift, axis=0)        # roll to start at green
    hsv_trimmed  = hsv_shifted[trim_bottom:256 - trim_top]                   # cut the wrap edge
    cmap         = ListedColormap(hsv_trimmed)

    for col, (data, _) in enumerate(zip(data_trajectory, betas_trajectory)):
        ax = fig.add_axes([origins[col], 0.57, w3d, h3d], projection="3d")

        total_lengths = np.zeros(400)
        for j in range(400):
            traj    = data[tmin:, j, :3]
            deltas  = np.diff(traj, axis=0)
            total_lengths[j] = np.nansum(np.linalg.norm(deltas, axis=1))

        # normalise lengths to [0, 1] for the colormap
        l_min, l_max = total_lengths.min(), total_lengths.max()
        norm_lengths = (total_lengths - l_min) / (l_max - l_min + 1e-12)
        norm = plt.Normalize(vmin = 20, vmax = 60) 

        # --- second pass: plot each trajectory with its length-based color ---
        for j in range(400):
            traj    = data[tmin:, j, :3]
            visible = (traj @ view) > 0
            traj_p  = np.where(visible[:, None], traj, np.nan)
            color = cmap(norm_lengths[j])  # one single color per trajectory
            ax.plot(traj_p[:, 0], traj_p[:, 1], traj_p[:, 2],
                    lw=0.8, alpha=1, color=color)

        ax.axis("off")

    #  colorbar in its own axes, placed below the 3D panels
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar_ax = fig.add_axes([0.3, 0.57, 0.5,  0.02])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
    cbar.set_ticks([])
    cbar_ax.text(-0.01, 0.5, "Track length", va="center", ha="right", transform=cbar_ax.transAxes, fontsize=32)
    cbar_ax.text( 1.01, 0.5, "$20 \\sim 60 \\, \\mu\\mathrm{m}$",    va="center", ha="left",  transform=cbar_ax.transAxes, fontsize=32)

    fig.text(0.25, 0.62, "$\\beta = {:.1}$".format(betas_trajectory[0]), ha="center", va="bottom", fontsize=32, fontweight="bold")
    fig.text(0.55, 0.62, "$\\beta = {:.1}$".format(betas_trajectory[1]), ha="center", va="bottom", fontsize=32, fontweight="bold")
    fig.text(0.85, 0.62, "$\\beta = {}$".format(round(betas_trajectory[2])), ha="center", va="bottom", fontsize=32, fontweight="bold")

    #plt.savefig("trajectories.pdf", bbox_inches="tight")
 