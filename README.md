# This directory contains the code for the modelling part in the paper "A supracellular desmosome-keratin network drives epithelial flocking and 3D rotation through alignment percolation" by He et al. (2026).

Authors: Leo Flehinghaus and David Brückner

-------------------------------------------------

## Content:

- simulation.py : Code for simulating active migration on a spherical surface
  - Developed in python 3.11. Dependencies: NumPy, SciPy
  - Output: Two .txt files. One lists the simulation settings and the other the positions and polarities at each recorded time step. 
- data_analysis.py : Relevant functions in analysing the simulations
  - Dependencies: NumPy, SciPy, Matplotlib
  - Angular velocity, velocity fluctuations, T1 transition rate, plotting of trajectories
 
---------------------------------------------------  

## Model of collective cell migration on a spherical surface 

We model cells as self-propelled particles on a spherical surface. Each cell is connected to its nearest neighbours via elastic springs, so that the force balance reads in the overdamped limit

$$ \zeta \vec{v}_i = f_0 \hat{P}_i + \vec{F}_i, $$
$$ \vec{F}_i = - K \sum_{\langle I, j \rangle} \left(r_{ij} - l_0 \right) \hat{r}_{ij}. $$

$K$ denotes the spring constant and $\zeta$ the substrate friction coefficient. $l_0$ is the common equilibrium length that remains constant throughout the simulation. The nearest neighbours are identified by repeated Delauney triangulations. 
Polarity vectors are confined to the local tangent plane and align align according to 

$$ \frac{\mathrm{d}\hat{P}_i}{\mathrm{d}t} = \left(\beta \sum_{\langle I, j \rangle} \left(\hat{P}_i \times \hat{P}_j\right) \cdot \hat{r}_i + \eta_i\right) (\hat{r}_i \times \hat{P}_i),$$

where the $\hat{r}$ symbols denote unit vectors. $\beta$ is the alignment strength and $\eta$ is white Gaussian noise. Distances are calculated in Cartesian coordinates, which are comparable to spherical coordinates for a sufficiently large number of cells. 


### Numerical integration

Initially, the cells are placed with random polarisation on a Fibonacci lattice, and the system is pre-equilibrated before trajectories are recorded. 

The equations of motion are integrated with the Euler-Maruyama scheme with the following procedure:
- Estimate the closest neighbours by computing the Delaunay triangulation,
- Compute forces and torques,
- Integrate and normalise the polarity vectors. All vectors are in the local tangent plane,
- Subtract the normal force from the forces to keep the cells on the correct radial distance,
- Integrate the position vectors with the updated polarity vectors,
- Project the polarity vectors back onto the tangent plane at the new positions.


### Simulation parameters

Typical parameters found to match the experimental observation are 
- Cell density $\rho = 008$,
- Number of cells $N = 400$,
- The self-propulsion velocity is set to 1 in a dimensionless form,
- Integration time $\Delta t = 10 ^{-2}$,
- Spring constant $K = 0.8$,
- Noise amplitude $\sigma = 0.7$,
- Alignment strength  $\beta = 0.1$.

-----------------------------------------------------

## Data analysis

The velocity of the cells is computed as

$$ \vec{v}_i = \mathrm{d} \vec{r} / \mathrm{d}t $$

and can be decomposed as the combination of a global rotation and a residual velocity

$$ \vec{v}_i = \vec{\omega} \times \vec{v}_i + \delta \vec{v}_i. $$

We estimate the angular velocity such that $\langle \delta \vec{v}\rangle = 0$. 

We compute the T1-transition rate by counting the edge flips while ignoring immediate flips back to the previous configuration.  

