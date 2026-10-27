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

# Step 6 - flow_matching_loss
def flow_matching_loss(v_pred, v_target):
    """Score a predicted velocity against the straight-path target."""
    # TODO: Implement `flow_matching_loss` to score predicted velocities against a target.
    return torch.mean((v_pred-v_target)**2)

# Step 7 - sinusoidal_time_embedding
import torch


def sinusoidal_time_embedding(t: torch.Tensor, embed_dim: int) -> torch.Tensor:
    """Map each time in t to a sin-cos Fourier feature vector of length embed_dim.

    Args:
        t: Float tensor of shape (batch,) or (batch, 1).
        embed_dim: An even positive integer.

    Returns:
        Float tensor of shape (batch, embed_dim) on the same device and dtype as t.
    """
    half = embed_dim // 2

    # Ensure t has shape (batch, 1) for broadcasting across the frequency dimension
    if t.ndim == 1:
        t = t.unsqueeze(1)

    # Compute frequencies: freqs[i] = exp(-log(10000) * i / half)
    # Shape: (half,)
    indices = torch.arange(half, device=t.device, dtype=t.dtype)
    freqs = torch.exp(-torch.log(torch.tensor(10000.0, device=t.device, dtype=t.dtype)) * indices / half)

    # Broadcast multiplication: (batch, 1) * (half,) -> (batch, half)
    args = t * freqs

    # Concatenate [sin, cos] along the feature dimension -> (batch, embed_dim)
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

# Step 8 - init_velocity_mlp
import math
import torch


def init_velocity_mlp(in_dim: int, hidden_dim: int, time_embed_dim: int, seed: int) -> dict[str, torch.Tensor]:
    """Return a seeded parameter dict for a two-hidden-layer velocity MLP.

    Args:
        in_dim: Dimension of the spatial state x.
        hidden_dim: Number of units in each hidden layer.
        time_embed_dim: Dimension of the sinusoidal time embedding (even integer).
        seed: Random seed for parameter initialization.

    Returns:
        dict containing keys 'W1', 'b1', 'W2', 'b2', 'W3', 'b3' with float32
        tensors requiring gradients.
    """
    torch.manual_seed(seed)

    # Layer dimensions (n_in, n_out)
    dims = [
        (in_dim + time_embed_dim, hidden_dim),  # Layer 1
        (hidden_dim, hidden_dim),                # Layer 2
        (hidden_dim, in_dim),                    # Layer 3
    ]

    params = {}
    for i, (n_in, n_out) in enumerate(dims, start=1):
        # Glorot normal initialization: std = sqrt(2 / (n_in + n_out))
        scale = math.sqrt(2.0 / (n_in + n_out))
        w = (torch.randn(n_in, n_out, dtype=torch.float32) * scale).requires_grad_(True)
        
        # Zero bias (torch.zeros does not advance the RNG)
        b = torch.zeros(n_out, dtype=torch.float32, requires_grad=True)

        params[f"W{i}"] = w
        params[f"b{i}"] = b

    return params

# Step 9 - velocity_mlp_forward
import torch
import torch.nn.functional as F


def velocity_mlp_forward(x: torch.Tensor, t, params: dict, time_embed_dim: int) -> torch.Tensor:
    """Return a velocity of the same shape as x from states, times, and MLP params.

    Args:
        x: Float tensor of shape (batch, in_dim).
        t: Scalar or float tensor broadcastable to (batch, 1).
        params: Parameter dict containing 'W1', 'b1', 'W2', 'b2', 'W3', 'b3'.
        time_embed_dim: Feature dimension for the sinusoidal time embedding.

    Returns:
        Predicted velocity v with shape (batch, in_dim).
    """
    batch_size = x.shape[0]

    # Convert t to tensor if scalar, match x's device and dtype, and broadcast/reshape to (batch, 1)
    if not isinstance(t, torch.Tensor):
        t = torch.tensor(t, device=x.device, dtype=x.dtype)
    t = t.view(-1, 1).expand(batch_size, 1)

    # Compute sinusoidal time embedding -> (batch, time_embed_dim)
    t_embed = sinusoidal_time_embedding(t, time_embed_dim)

    # Concatenate spatial state and time embedding -> (batch, in_dim + time_embed_dim)
    h = torch.cat([x, t_embed], dim=-1)

    # Layer 1 + ReLU
    h = F.relu(h @ params["W1"] + params["b1"])

    # Layer 2 + ReLU
    h = F.relu(h @ params["W2"] + params["b2"])

    # Output Layer (linear) -> (batch, in_dim)
    v = h @ params["W3"] + params["b3"]

    return v

# Step 10 - make_mixture_dataset
import torch


def make_mixture_dataset(
    centers: torch.Tensor, n_per_component: int, std: float, seed: int
) -> torch.Tensor:
    """Build a fixed-seed 2-D Gaussian mixture from component centers.

    Args:
        centers: Float tensor of shape (n_components, 2).
        n_per_component: Number of samples to draw per component.
        std: Standard deviation of each isotropic Gaussian component.
        seed: Random seed for reproducibility.

    Returns:
        Tensor of shape (n_components * n_per_component, 2) of dtype torch.float32.
    """
    torch.manual_seed(seed)

    n_components = centers.shape[0]
    samples = []

    for k in range(n_components):
        # Draw standard normal samples Z ~ N(0, I) for component k
        z = torch.randn(n_per_component, 2, dtype=torch.float32, device=centers.device)
        component_samples = centers[k].to(dtype=torch.float32) + std * z
        samples.append(component_samples)

    return torch.cat(samples, dim=0)

# Step 11 - flow_train_step
import torch


def flow_train_step(params, data, batch_size, lr, time_embed_dim, seed):
    """Draw a minibatch of data, compute the flow-matching loss through the velocity MLP,
    and take one manual SGD update.
    """
    # 1. Seed RNG for index sampling
    torch.manual_seed(seed)

    # 2. Sample row indices and extract minibatch
    n = data.shape[0]
    indices = torch.randint(0, n, (batch_size,))
    minibatch = data[indices]

    # 3. Build flow interpolant batch: make_flow_batch returns (xt, v_target, t)
    xt, ut, t = make_flow_batch(minibatch, seed + 1)

    # 4. Predict velocity with MLP forward pass
    pred = velocity_mlp_forward(xt, t, params, time_embed_dim)

    # 5. Compute flow-matching loss and record pre-update float value
    loss = flow_matching_loss(pred, ut)
    loss_value = float(loss.item())

    # 6. Zero out any existing gradients
    for p in params.values():
        if p.grad is not None:
            p.grad.zero_()

    # 7. Backpropagate
    loss.backward()

    # 8. In-place SGD update
    with torch.no_grad():
        for p in params.values():
            p.sub_(lr * p.grad)

    return params, loss_value

# Step 12 - train_rectified_flow
def train_rectified_flow(params, data, n_steps, batch_size, lr, time_embed_dim, seed):
    """Repeat flow_train_step for n_steps and return updated params plus the loss history."""
    # TODO: train_rectified_flow runs a short training loop for the velocity MLP...
    train_loss = []

    for step in range(n_steps):
        params, loss_value = flow_train_step(params, data, batch_size, lr, time_embed_dim, seed = seed + step)
        train_loss.append(loss_value)
        
    return params, train_loss

# Step 13 - linspace_timesteps
def linspace_timesteps(n_steps):
    # TODO: Return n_steps plus one evenly spaced times from 0 to 1 inclusive.
    return torch.linspace(0, 1, n_steps+1)

