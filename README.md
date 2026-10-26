# rectified-flow-from-scratch
Build a complete rectified-flow / linear interpolant flow-matching generative model in pure PyTorch. Instead of a long discrete reverse chain that predicts noise, you learn a velocity field that transports Gaussian noise to data along straight paths, then sample by integrating an ODE with Euler and Heun steps.
