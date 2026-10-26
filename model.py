"""
Rectified Flow from Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - interpolate_linear
def interpolate_linear(x0, x1, t):
    """Return the linear interpolant from noise at t=0 to data at t=1."""
    # TODO: Return the linear interpolant from noise at t=0 to data at t=1.
    return (1-t)*x0 + t*x1

