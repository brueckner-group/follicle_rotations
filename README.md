# This directory contains the code for the modelling part in the paper "A supracellular desmosome-keratin network drives epithelial flocking and 3D rotation through alignment percolation" by He et al. (2026).

-------------------------------------------------

## Content:

- simulation.py : Code for simulating active migration on a spherical surface
  - Developed in python 3.11. Dependencies: NumPy, SciPy
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

$$ \frac{\mathrm{d}\hat{P}_i}{\mathrm{d}t} = \left(\beta \sum_{\langle I, j \rangle} \left(\hat{P}_i \times \hat{P}_j\right) \cdot \hat{r}_i + \eta_i\right) (\hat{r}_i \times \hat{P}_i,$$

where the $\hat{r}$ symbols denote unit vectors. $\beta$ is the alignment strength and $\eta$ is white Gaussian noise. 
