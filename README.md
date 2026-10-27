# Rectified Flow from Scratch

Build a complete rectified-flow / linear interpolant flow-matching generative model in pure PyTorch. Instead of a long discrete reverse chain that predicts noise, you learn a velocity field that transports Gaussian noise to data along straight paths, then sample by integrating an ODE with Euler and Heun steps.

## How to run

```bash
python scaffold.py
```

## Steps

- [x] **1.** interpolate_linear
- [x] **2.** target_velocity
- [x] **3.** sample_uniform_time
- [x] **4.** sample_gaussian_noise
- [x] **5.** make_flow_batch
- [x] **6.** flow_matching_loss
- [x] **7.** sinusoidal_time_embedding
- [x] **8.** init_velocity_mlp
- [x] **9.** velocity_mlp_forward
- [x] **10.** make_mixture_dataset
- [x] **11.** flow_train_step
- [x] **12.** train_rectified_flow
- [x] **13.** linspace_timesteps
- [x] **14.** euler_step
- [x] **15.** euler_sample
- [x] **16.** heun_step
- [x] **17.** heun_sample
- [x] **18.** sample_quality_mse
- [x] **19.** make_reflow_pairs

---

Built on Deep-ML.
