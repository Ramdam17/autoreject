# AutoReject GPU Benchmark Report

**Date:** 2026-04-11 14:38
**Machine:** ng11104
**CPU:** AMD EPYC 7413 24-Core Processor
**GPU:** NVIDIA A100-SXM4-40GB (CUDA)
**Python:** 3.12.4
**PyTorch:** 2.11.0
**MNE:** 1.12.0
**Metal:** False
**CuPy:** True
**Git:** 787efc2

## ds000117_face

Data: 1179 epochs x 70 channels x 276 times
  *Face recognition sub-16, runs [3, 4, 5, 6]*

### Performance

| Backend | Wall time (ms) | vs CPU | Peak GPU (MB) |
|---------|---------------|--------|---------------|
| numpy_cpu | 190361.8 | 1.0x | - |
| torch_gpu | 9704.9 | 19.6x | 5064.1 |
| torch_gpu_argmin | 7586.0 | 25.1x | 5065.3 |
| cuda_kernel | ERROR | - | - |

### Accuracy (vs CPU reference)

| Backend | Thresh match | Mean diff | Consensus | n_interpolate |
|---------|-------------|---------|-----------|---------------|
| cuda_kernel | nan% | nan% | DIFF | DIFF |
| torch_gpu_argmin | 0.0% | 5.08% | match | match |
| torch_gpu | 0.0% | 0.27% | match | match |

## ds002778_parkinson

Data: 97 epochs x 32 channels x 1536 times
  *Parkinson resting-state sub-pd14 ses-off*

### Performance

| Backend | Wall time (ms) | vs CPU | Peak GPU (MB) |
|---------|---------------|--------|---------------|
| numpy_cpu | 27713.8 | 1.0x | - |
| torch_gpu | 1126.2 | 24.6x | 379.1 |
| torch_gpu_argmin | 429.7 | 64.5x | 379.1 |
| cuda_kernel | 585.2 | 47.4x | 224.1 |

### Accuracy (vs CPU reference)

| Backend | Thresh match | Mean diff | Consensus | n_interpolate |
|---------|-------------|---------|-----------|---------------|
| cuda_kernel | 0.0% | 2.91% | match | match |
| torch_gpu_argmin | 0.0% | 2.91% | match | match |
| torch_gpu | 0.0% | 0.00% | match | match |

## mne_sample_eeg

Data: 25 epochs x 60 channels x 211 times
  *MNE sample auditory-visual, EEG only*

### Performance

| Backend | Wall time (ms) | vs CPU | Peak GPU (MB) |
|---------|---------------|--------|---------------|
| numpy_cpu | 17024.0 | 1.0x | - |
| torch_gpu | 1694.1 | 10.0x | 29.4 |
| torch_gpu_argmin | 430.0 | 39.6x | 29.4 |
| cuda_kernel | 429.9 | 39.6x | 22.2 |

### Accuracy (vs CPU reference)

| Backend | Thresh match | Mean diff | Consensus | n_interpolate |
|---------|-------------|---------|-----------|---------------|
| cuda_kernel | 0.0% | 2.94% | match | match |
| torch_gpu_argmin | 0.0% | 2.94% | match | match |
| torch_gpu | 0.0% | 0.00% | match | match |

## scale_10min

Data: 300 epochs x 64 channels x 1000 times

### Performance

| Backend | Wall time (ms) | vs CPU | Peak GPU (MB) |
|---------|---------------|--------|---------------|
| numpy_cpu | 103003.8 | 1.0x | - |
| torch_gpu | 3692.7 | 27.9x | 1272.8 |
| torch_gpu_argmin | 2195.0 | 46.9x | 1273.1 |
| cuda_kernel | 2921.2 | 35.3x | 865.5 |

### Accuracy (vs CPU reference)

| Backend | Thresh match | Mean diff | Consensus | n_interpolate |
|---------|-------------|---------|-----------|---------------|
| cuda_kernel | 0.0% | 17.20% | match | match |
| torch_gpu_argmin | 0.0% | 17.20% | match | match |
| torch_gpu | 0.0% | 0.00% | match | match |

## scale_128ch

Data: 300 epochs x 128 channels x 1000 times

### Performance

| Backend | Wall time (ms) | vs CPU | Peak GPU (MB) |
|---------|---------------|--------|---------------|
| numpy_cpu | 207296.6 | 1.0x | - |
| torch_gpu | 11798.1 | 17.6x | 2537.5 |
| torch_gpu_argmin | 8816.7 | 23.5x | 2538.1 |
| cuda_kernel | 10259.3 | 20.2x | 1723.7 |

### Accuracy (vs CPU reference)

| Backend | Thresh match | Mean diff | Consensus | n_interpolate |
|---------|-------------|---------|-----------|---------------|
| cuda_kernel | 0.0% | 21.38% | match | match |
| torch_gpu_argmin | 0.0% | 21.38% | match | match |
| torch_gpu | 0.0% | 0.00% | match | match |

## scale_20min

Data: 600 epochs x 64 channels x 1000 times

### Performance

| Backend | Wall time (ms) | vs CPU | Peak GPU (MB) |
|---------|---------------|--------|---------------|
| numpy_cpu | 192701.0 | 1.0x | - |
| torch_gpu | 7002.8 | 27.5x | 2922.3 |
| torch_gpu_argmin | 5344.9 | 36.1x | 2922.9 |
| cuda_kernel | 7050.9 | 27.3x | 1723.3 |

### Accuracy (vs CPU reference)

| Backend | Thresh match | Mean diff | Consensus | n_interpolate |
|---------|-------------|---------|-----------|---------------|
| cuda_kernel | 0.0% | 16.06% | match | match |
| torch_gpu_argmin | 0.0% | 16.06% | match | match |
| torch_gpu | 0.0% | 0.00% | match | match |

## scale_32ch

Data: 300 epochs x 32 channels x 1000 times

### Performance

| Backend | Wall time (ms) | vs CPU | Peak GPU (MB) |
|---------|---------------|--------|---------------|
| numpy_cpu | 54524.8 | 1.0x | - |
| torch_gpu | 1416.8 | 38.5x | 645.8 |
| torch_gpu_argmin | 668.6 | 81.5x | 645.9 |
| cuda_kernel | 1059.8 | 51.4x | 440.6 |

### Accuracy (vs CPU reference)

| Backend | Thresh match | Mean diff | Consensus | n_interpolate |
|---------|-------------|---------|-----------|---------------|
| cuda_kernel | 0.0% | 17.27% | DIFF | DIFF |
| torch_gpu_argmin | 0.0% | 17.27% | DIFF | DIFF |
| torch_gpu | 0.0% | 0.00% | match | match |

## scale_5min

Data: 150 epochs x 64 channels x 1000 times

### Performance

| Backend | Wall time (ms) | vs CPU | Peak GPU (MB) |
|---------|---------------|--------|---------------|
| numpy_cpu | 58840.8 | 1.0x | - |
| torch_gpu | 2737.2 | 21.5x | 625.0 |
| torch_gpu_argmin | 1341.3 | 43.9x | 625.2 |
| cuda_kernel | 1708.5 | 34.4x | 444.3 |

### Accuracy (vs CPU reference)

| Backend | Thresh match | Mean diff | Consensus | n_interpolate |
|---------|-------------|---------|-----------|---------------|
| cuda_kernel | 0.0% | 18.22% | match | match |
| torch_gpu_argmin | 0.0% | 18.22% | match | match |
| torch_gpu | 0.0% | 0.00% | match | match |

## scale_64ch

Data: 300 epochs x 64 channels x 1000 times

### Performance

| Backend | Wall time (ms) | vs CPU | Peak GPU (MB) |
|---------|---------------|--------|---------------|
| numpy_cpu | 103946.8 | 1.0x | - |
| torch_gpu | 3681.6 | 28.2x | 1272.8 |
| torch_gpu_argmin | 2199.2 | 47.3x | 1273.1 |
| cuda_kernel | 2923.5 | 35.6x | 865.5 |

### Accuracy (vs CPU reference)

| Backend | Thresh match | Mean diff | Consensus | n_interpolate |
|---------|-------------|---------|-----------|---------------|
| cuda_kernel | 0.0% | 17.20% | match | match |
| torch_gpu_argmin | 0.0% | 17.20% | match | match |
| torch_gpu | 0.0% | 0.00% | match | match |

## scale_clean_10pct

Data: 300 epochs x 64 channels x 1000 times

### Performance

| Backend | Wall time (ms) | vs CPU | Peak GPU (MB) |
|---------|---------------|--------|---------------|
| numpy_cpu | 106441.7 | 1.0x | - |
| torch_gpu | 3553.3 | 30.0x | 1272.8 |
| torch_gpu_argmin | 2068.0 | 51.5x | 1273.1 |
| cuda_kernel | 2805.8 | 37.9x | 865.5 |

### Accuracy (vs CPU reference)

| Backend | Thresh match | Mean diff | Consensus | n_interpolate |
|---------|-------------|---------|-----------|---------------|
| cuda_kernel | 0.0% | 15.89% | match | match |
| torch_gpu_argmin | 0.0% | 15.89% | match | match |
| torch_gpu | 0.0% | 0.01% | match | match |

## scale_highdensity_128ch

Data: 300 epochs x 128 channels x 1000 times

### Performance

| Backend | Wall time (ms) | vs CPU | Peak GPU (MB) |
|---------|---------------|--------|---------------|
| numpy_cpu | 241214.6 | 1.0x | - |
| torch_gpu | 19908.7 | 12.1x | 2601.5 |
| torch_gpu_argmin | 16868.8 | 14.3x | 2601.5 |
| cuda_kernel | 18332.4 | 13.2x | 2602.6 |

### Accuracy (vs CPU reference)

| Backend | Thresh match | Mean diff | Consensus | n_interpolate |
|---------|-------------|---------|-----------|---------------|
| cuda_kernel | 0.0% | 21.38% | match | match |
| torch_gpu_argmin | 0.0% | 21.38% | match | match |
| torch_gpu | 0.0% | 0.00% | match | match |

## scale_noisy_30pct

Data: 300 epochs x 64 channels x 1000 times

### Performance

| Backend | Wall time (ms) | vs CPU | Peak GPU (MB) |
|---------|---------------|--------|---------------|
| numpy_cpu | 101842.0 | 1.0x | - |
| torch_gpu | 3688.8 | 27.6x | 1272.8 |
| torch_gpu_argmin | 2187.3 | 46.6x | 1273.1 |
| cuda_kernel | 2924.0 | 34.8x | 865.5 |

### Accuracy (vs CPU reference)

| Backend | Thresh match | Mean diff | Consensus | n_interpolate |
|---------|-------------|---------|-----------|---------------|
| cuda_kernel | 0.0% | 17.20% | match | match |
| torch_gpu_argmin | 0.0% | 17.20% | match | match |
| torch_gpu | 0.0% | 0.00% | match | match |

## smoke_32ch

Data: 120 epochs x 32 channels x 250 times

### Performance

| Backend | Wall time (ms) | vs CPU | Peak GPU (MB) |
|---------|---------------|--------|---------------|
| numpy_cpu | 15226.5 | 1.0x | - |
| torch_gpu | 945.2 | 16.1x | 78.3 |
| torch_gpu_argmin | 218.2 | 69.8x | 78.3 |
| cuda_kernel | 250.8 | 60.7x | 42.7 |

### Accuracy (vs CPU reference)

| Backend | Thresh match | Mean diff | Consensus | n_interpolate |
|---------|-------------|---------|-----------|---------------|
| cuda_kernel | 0.0% | 12.91% | match | match |
| torch_gpu_argmin | 0.0% | 12.91% | match | match |
| torch_gpu | 0.0% | 0.00% | match | match |

## var_32ch

Data: 300 epochs x 32 channels x 1000 times

### Performance

| Backend | Wall time (ms) | vs CPU | Peak GPU (MB) |
|---------|---------------|--------|---------------|
| numpy_cpu | 51368.4 | 1.0x | - |
| torch_gpu | 1417.0 | 36.3x | 645.8 |
| torch_gpu_argmin | 666.0 | 77.1x | 645.9 |
| cuda_kernel | 1086.0 | 47.3x | 440.6 |

### Accuracy (vs CPU reference)

| Backend | Thresh match | Mean diff | Consensus | n_interpolate |
|---------|-------------|---------|-----------|---------------|
| cuda_kernel | 0.0% | 12.36% | DIFF | DIFF |
| torch_gpu_argmin | 0.0% | 12.36% | DIFF | DIFF |
| torch_gpu | 0.0% | 0.00% | match | match |

## var_64ch

Data: 300 epochs x 64 channels x 1000 times

### Performance

| Backend | Wall time (ms) | vs CPU | Peak GPU (MB) |
|---------|---------------|--------|---------------|
| numpy_cpu | 100841.8 | 1.0x | - |
| torch_gpu | 3682.1 | 27.4x | 1272.8 |
| torch_gpu_argmin | 2186.1 | 46.1x | 1273.1 |
| cuda_kernel | 2923.5 | 34.5x | 865.5 |

### Accuracy (vs CPU reference)

| Backend | Thresh match | Mean diff | Consensus | n_interpolate |
|---------|-------------|---------|-----------|---------------|
| cuda_kernel | 0.0% | 17.99% | match | match |
| torch_gpu_argmin | 0.0% | 17.99% | match | match |
| torch_gpu | 0.0% | 0.04% | match | match |

## var_mne_sample

Data: 25 epochs x 60 channels x 211 times
  *MNE sample auditory-visual, EEG only*

### Performance

| Backend | Wall time (ms) | vs CPU | Peak GPU (MB) |
|---------|---------------|--------|---------------|
| numpy_cpu | 14729.7 | 1.0x | - |
| torch_gpu | 1673.2 | 8.8x | 29.4 |
| torch_gpu_argmin | 461.5 | 31.9x | 29.4 |
| cuda_kernel | 460.1 | 32.0x | 22.2 |

### Accuracy (vs CPU reference)

| Backend | Thresh match | Mean diff | Consensus | n_interpolate |
|---------|-------------|---------|-----------|---------------|
| cuda_kernel | 0.0% | 1.27% | match | match |
| torch_gpu_argmin | 0.0% | 1.27% | match | match |
| torch_gpu | 0.0% | 0.00% | match | match |

## Figures

![accuracy_scatter](../figures/accuracy_scatter.png)

![memory_usage](../figures/memory_usage.png)

![timing_comparison](../figures/timing_comparison.png)

![validation_summary](../figures/validation_summary.png)

## Optimizations Explored

| Optimization | Hotspot | Isolated speedup | Status |
|-------------|---------|-----------------|--------|
| Metal fused threshold-CV kernel | batched_cv_loss (22.6%) | 4-14x vs PyTorch (small-medium) | Implemented |
| Batched consensus scoring (einsum) | cv_scoring_loop (27.6%) | 3.7-6.4x | Implemented |
| GPU argmin (replace bayes_opt) | bayesian_opt (11.5%) | Exact solution, deterministic | Implemented |
| Median topk | cv_median (16.4%) | 0.5x (slower) | Abandoned |
| Batched interpolation | per_epoch_interp (16.6%) | 1.1-1.3x (marginal) | Marginal |

## Key Finding

Bayesian optimization is redundant in the GPU pipeline: all CV losses are pre-computed in batch, so the GP surrogate models a fully known function. `argmin` gives the exact minimum (deterministic) instead of a stochastic approximation, while eliminating CPU<->GPU round-trips.

*Ref: Jas et al. (2017) 'Candidate thresholds using Bayesian optimization' — motivation was computational efficiency, not methodological.*
