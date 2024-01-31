import numpy as np


# Function to count particles in nuclei
def count_particles_in_nuclei(labeled_particles, labeled_nuclei, num_nuclei):
    particle_counts = np.zeros(num_nuclei, dtype=int)
    for i in range(1, num_nuclei + 1):
        nuclei_mask = labeled_nuclei == i
        # Count the number of unique particles in the nucleus
        unique_particles = np.unique(labeled_particles[nuclei_mask])
        # Exclude the background label (0)
        particle_counts[i - 1] = len(unique_particles[unique_particles > 0])
    return particle_counts
