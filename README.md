# This directory contains the code for the modelling part in the paper "A supracellular desmosome-keratin network drives epithelial flocking and 3D rotation through alignment percolation" by He et al. (2026).

-------------------------------------------------

We model cells as self-propelled particles on a spherical surface. Each cell is connected to its nearest neighbours via elastic springs, so that the force balance reads in the overdamped limit

$$ \vec{v} $$

where the neighbours are identified with a Delauney triangulation. We implemented polarity alignment
