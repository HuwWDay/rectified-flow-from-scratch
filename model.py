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

# Step 2 - target_velocity
def target_velocity(x0, x1):
    """Return the constant straight-path target velocity from noise x0 to data x1."""
    # TODO: Return the constant straight-path target velocity along the interpolant...
    return x1 - x0

