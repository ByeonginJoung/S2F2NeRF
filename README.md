# Stable Surface Regularization for Fast Few-Shot NeRF

This repository provides a reproduced version for stable surface regularization for fast few-shot nerf, which is published in 3DV 2024.

### Environments

```
python 3.9.12
pytorch 1.11.0
numpy 1.22.3
open3d 0.15.2
matplotlib 3.5.2
scikit-image 0.19.2
tqdm
lpips
```

To enable gradient upon grid, we are using [smooth_sampler](https://github.com/tymoteuszb/smooth-sampler), as same as [Go-Surf](https://github.com/JingwenWang95/go-surf).