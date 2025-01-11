# -*- coding: utf-8 -*-
# @Author: Georg C. Ganzenmueller, Albert-Ludwigs Universitaet Freiburg, Germany
# @Date:   2025-01-06 20:37:42
# @Last Modified by:   Georg C. Ganzenmueller, Albert-Ludwigs Universitaet Freiburg, Germany
# @Last Modified time: 2025-01-06 20:37:44
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate

def compute_displacement(frame1, frame2):
    """
    Compute displacement between two consecutive frames using cross-correlation.
    
    Args:
    - frame1, frame2: 1D numpy arrays representing consecutive frames (images).
    
    Returns:
    - displacement: The displacement of the feature between frame1 and frame2.
    """
    # Compute cross-correlation
    correlation = correlate(frame2, frame1, mode='full')
    
    # Find the index of the maximum correlation (displacement)
    displacement_index = np.argmax(correlation) - (len(frame1) - 1)
    
    return displacement_index

def compute_strain(displacements, dx):
    """
    Compute strain from displacement data.
    
    Args:
    - displacements: Array of displacements over time.
    - dx: Spatial step (distance between two adjacent points).
    
    Returns:
    - strain: The strain calculated as the spatial derivative of displacement.
    """
    # Calculate strain as the numerical derivative of displacement
    strain = np.gradient(displacements, dx)
    return strain

# Example usage:
# Assume you have a sequence of 1D frames (arrays) stored in 'frames'
frames = [np.sin(np.linspace(0, 2*np.pi, 100) + i) for i in range(10)]  # Example frames

# Compute displacements between consecutive frames
displacements = []
for i in range(len(frames) - 1):
    displacement = compute_displacement(frames[i], frames[i+1])
    displacements.append(displacement)

# Assume a constant spatial step dx between adjacent points
dx = 0.1  # Example spatial resolution
strain = compute_strain(displacements, dx)

# Plot the results
plt.figure(figsize=(12, 6))

# Displacement plot
plt.subplot(1, 2, 1)
plt.plot(displacements, label='Displacement')
plt.title('Displacement over time')
plt.xlabel('Frame index')
plt.ylabel('Displacement (pixels)')
plt.legend()

# Strain plot
plt.subplot(1, 2, 2)
plt.plot(strain, label='Strain', color='r')
plt.title('Strain over time')
plt.xlabel('Frame index')
plt.ylabel('Strain')
plt.legend()

plt.tight_layout()
plt.show()
