# Dataset Selection and Task Scope

Verification date: 2026-10-30. This experiment does not use NGSIM or replace trajectories from the actual data files with artificially generated Bezier curves.

## The two candidate datasets

| Aspect | Kaggle Humanoid Robot Navigation | PHUMA (selected) |
| --- | --- | --- |
| Original link | https://www.kaggle.com/datasets/zara2099/humanoid-robot-navigation-dataset | https://arxiv.org/html/2510.26236v3 |
| Availability | CSV downloaded and inspected: 1,000 rows and 13 columns | Public official Hugging Face release; a fixed revision can be retrieved |
| Contents | Navigation status, terrain, energy consumption, stability, Bezier control points, etc. | Per-frame root position, orientation, joint positions, and frame rate for G1/H1-2 |
| Suitability for continuous position forecasting | Lacks per-frame actual positions, sampling intervals, and reliable episode boundaries needed for this task | `root_trans` and `fps` directly support history/future windows |
| Suitable simple task | Navigation outcome classification or score regression; provenance needs further verification | Short-term 2D root trajectory forecasting for G1 walking motions |
| Publisher's license declaration | CC0 | Hugging Face data card: Apache-2.0; upstream source restrictions still apply |

The actual Kaggle CSV fields are: `time_step, robot_id, terrain_type, obstacle_count, obstacle_distance, bezier_control_points, planned_trajectory_length, mpc_adjustment, gait_stability_index, center_of_mass_offset, energy_consumption, navigation_status, navigation_score`.

The schema verification record and two sample rows are in `data/sources/kaggle_schema_verified.json`. The presence of `time_step` and `robot_id` alone does not establish that rows represent continuous motion at regular sampling intervals. Bezier control points describe a planned path; they cannot be treated as observed positions along an executed trajectory.

## PHUMA sources

- Paper: Lee et al., *PHUMA: Physically Reliable Humanoid Locomotion Dataset*, arXiv:2510.26236v3.
- Official code: https://github.com/DAVIAN-Robotics/PHUMA
- Project page: https://davian-robotics.github.io/PHUMA/
- Official data: https://huggingface.co/datasets/DAVIAN-Robotics/PHUMA
- Pinned revision: `48b0f5982969104ab041461e594a8b8c36db8ba5`.
- Original fields: `root_trans (T,3)`, `root_ori (T,4)`, `dof_pos (T,D)`, and `fps`.

PHUMA processes human motion data using physical constraints and retargets the motions to humanoid robots. The trajectories in these public files are **robot reference motions**, not execution logs measured by sensors on a physical robot. The paper primarily addresses motion imitation and control. This assignment instead defines a supervised trajectory forecasting task; it does not reproduce the paper's main experiments.

## A small, reproducible subset

Select motions under `data/g1/idea400/` whose filenames contain `walking` and match `number_|hand_pockets|arm_swing|clapping|salute|waving|pointing`. These are walking motions accompanied by simple upper-body actions. Other robots, sources, and more complex motion categories are excluded. The selection does not imply a strict label of straight-line, constant-speed walking.

The subset contains 295 segments, 169 source motions grouped by name, and 29,856 original frames at 30 Hz. Taking every third frame produces 10,006 frames at 10 Hz. XYZ positions are retained, while the model uses only horizontal XY coordinates. The approximately 111 KiB NPZ is embedded in both notebooks, avoiding the need for a data account or the complete archive of approximately 3.4 GB. `prepare_phuma.py` can reconstruct the same subset from the pinned official release using HTTP Range requests.

Selection does not depend on prediction error. Each original file's path, SHA-256, CRC, frame rate, and length are recorded in `data/manifest.json`. The hashes of all 295 cached original files were checked, and the NPZ coordinates were verified to exactly match the original `root_trans[::3]`. Classroom NPZ loading uses `allow_pickle=False`; the reconstruction tool permits only the NumPy types needed by the original NPY dictionaries.

Source names are grouped after removing `_clip...` / `_chunk...` suffixes. Splitting occurs before windowing, and separate segments are never concatenated. The default window requires at least 30 frames, leaving 157 usable source motions: 109 for training, 23 for validation, and 25 for testing. Their window counts are 794, 172, and 191, respectively. The remaining 12 source motions do not meet the length requirement.

## Limits on interpretation

1. Splitting by source name prevents chunks of the same source motion from appearing in different sets. Reliable actor IDs are unavailable, so actor-independent evaluation cannot be claimed, and correlations between differently named motions cannot be ruled out.
2. Neighboring windows overlap. The report includes both averages over windows and averages giving equal weight to each source motion. All windows must not be treated as independent samples when calculating confidence intervals.
3. No additional smoothing or interpolation is applied here. PHUMA's upstream pipeline already includes motion processing and physical optimization. The results measure the predictability of processed reference trajectories, not noisy online sensor measurements from a physical robot.
4. XY forecasts omit whole-body joints, orientation, obstacles, contacts, and control inputs. They cannot directly serve as a robot controller or guarantee dynamic feasibility. The deterministic RNN also does not model multiple possible future intentions.
5. The current result uses one fixed split and seed for classroom purposes. Generalization across scenes, sources, robots, seeds, and longer horizons requires separate evaluation.

## Licensing and citation

The package retains `data/sources/PHUMA_DATA_CARD.md`, `PHUMA_LICENSE.txt`, and `PHUMA_REPOSITORY_README.md`. The official data card explicitly states that the public release excludes restricted motions derived from LAFAN1 and LocoMuJoCo. This subset uses only the IDEA400 portion of the public archive. Apache-2.0 is the publisher's declaration for that data repository; it does not remove rights associated with the original human motion sources. The paper's CC BY 4.0 license is not the dataset license. Review the upstream Motion-X / IDEA400 terms before further public redistribution or commercial use.

```bibtex
@article{lee2025phuma,
  title={PHUMA: Physically Reliable Humanoid Locomotion Dataset},
  author={Lee, Kyungmin and Kim, Sibeen and Lee, Youngdo and Park, Minho
          and Kim, Hyunseung and Hwang, Dongyoon and Kim, Donghu
          and Lee, Hojoon and Choo, Jaegul},
  journal={arXiv preprint arXiv:2510.26236},
  year={2025}
}
```
