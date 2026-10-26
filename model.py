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

# Step 3 - sample_uniform_time
def sample_uniform_time(batch_size, seed):
    """Draw a column of times uniformly from the unit interval."""
    # TODO: Draw a column of times uniformly from the unit interval...
    torch.manual_seed(seed)
    return torch.rand((batch_size, 1))

# Step 4 - sample_gaussian_noise
def sample_gaussian_noise(x, seed):
    """Return standard normal noise matching the shape of a given tensor using a given seed."""
    # TODO: Return standard normal noise matching a tensor using a given seed...
    torch.manual_seed(seed)
    return torch.randn_like(x)

# Step 5 - make_flow_batch
def make_flow_batch(x1, seed):
    """Given data x1, sample noise and time and return xt, target velocity, and t."""
    # TODO: Implement `make_flow_batch` to assemble one flow-matching training triple from a batch of data....
    batch, dim = x1.shape
    x0 = sample_gaussian_noise(x1, seed)
    v = target_velocity(x0, x1)
    t = sample_uniform_time(batch, seed)
    xt = interpolate_linear(x0, x1, t)
    return xt, v, t

