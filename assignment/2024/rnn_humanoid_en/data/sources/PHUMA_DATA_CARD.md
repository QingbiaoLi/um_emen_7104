---
license: apache-2.0
task_categories:
- robotics
- reinforcement-learning
tags:
- humanoid
- locomotion
- dataset
- retargeting
- physics-simulation
size_categories:
- 100K<n<1M
---

# PHUMA: Physically Reliable Humanoid Locomotion Dataset

> [!WARNING]
> **This repository does NOT contain the complete PHUMA dataset.**
> 
> Due to licensing restrictions, motions derived from **LAFAN1** and **LocoMuJoCo** are excluded from this release. To obtain the full PHUMA dataset, please follow the
instructions in our [GitHub repository](https://github.com/DAVIAN-Robotics/PHUMA).

This repository provides physically reliable humanoid locomotion dataset, PHUMA.

PHUMA leverages large-scale human motion data while overcoming physical artifacts through careful data curation and physics-constrained retargeting to create a high-quality humanoid locomotion dataset.

For detailed results, implementation notes, and videos, please visit our [paper](https://arxiv.org/abs/2510.26236), [project page](https://davian-robotics.github.io/PHUMA/) and [GitHub repository](https://github.com/DAVIAN-Robotics/PHUMA).

## Download and Setup

The dataset is provided as a compressed file. To use it:

```bash
# Download data.zip from this repository
# Then extract it:
unzip data.zip
```

This will create a `data/` directory with all the motion data.

## Dataset Structure

The dataset contains retargeted data for 2 different humanoids:

```
data/
├── g1/          # Humanoid configuration g1
└── h1_2/        # Humanoid configuration h1_2
```

## Data Format

Each `.npy` file in the dataset follows a consistent structure:

```python
{
    'root_trans': (num_frames, 3),      # Root translation (x, y, z)
    'root_ori': (num_frames, 4),        # Root orientation quaternion (x, y, z, w)
    'dof_pos': (num_frames, num_dof),   # Degrees of freedom positions for all joints
    'fps': fps                          # Frame rate (frames per second)
}
```

### Field Descriptions

- **root_trans**: Root joint translation in 3D space `(x, y, z)` for each frame
- **root_ori**: Root joint orientation as quaternion `(x, y, z, w)` for each frame
- **dof_pos**: Joint positions for all degrees of freedom across frames
- **fps**: Frame rate of the motion sequence

## Citation

If you find this dataset useful in your research, please cite our paper:

```bibtex
@article{lee2025phuma,
    title={PHUMA: Physically-Grounded Humanoid Locomotion Dataset}, 
    author={Kyungmin Lee and Sibeen Kim and Youngdo Lee and Minho Park and Hyunseung Kim and Dongyoon Hwang and Donghu Kim and Hojoon Lee and Jaegul Choo},
    journal={arXiv preprint arXiv:2510.26236},
    year={2025}
}
```
