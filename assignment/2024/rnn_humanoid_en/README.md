# RNN Humanoid Trajectory Prediction Teaching Package

**Start with `humanoid_rnn_instructor.ipynb` (complete executable solution) or `humanoid_rnn_student.ipynb` (student exercise).** Both notebooks include the same PHUMA G1 subset and shared code. Uploading a single notebook to Colab is sufficient; no additional files or Kaggle/Hugging Face account are required to obtain the data.

## Running the notebooks

At https://colab.research.google.com/, select **Upload notebook**, upload the appropriate `.ipynb` file, choose a Python 3 CPU runtime, and run the cells in order. The first cell installs dependencies only if required libraries are missing. In the student version, complete the 13 TODO blocks before running the checks and training cells. The unfilled version intentionally raises `NotImplementedError`.

For local Jupyter, Python 3.12 is recommended. Run `python -m pip install -r requirements-notebook.txt`, then open the notebook with `jupyter notebook`. Execution creates `humanoid_rnn_workspace/` in the current directory and a separate output directory for each training run. The instructor notebook retains plots and outputs from an actual execution. Rerunning it trains from scratch and does not depend on saved weights.

The English notebooks preserve the executable code and, for the instructor version, the recorded outputs of the verified Chinese originals. Only teaching text and documentation references have been translated. The original notebooks were tested in independent local Jupyter kernels; cloud execution in Colab has not been tested directly.

Script entry points are also available:

```bash
python -m pip install -r requirements.txt
python check_assignment.py --module solution
python solution.py --epochs 100 --output outputs/new_run
# After completing student.py:
python check_assignment.py --module student
python student.py --epochs 100 --output outputs/student_run
```

## Model and task

The input consists of 20 historical XY positions sampled at 10 Hz, at times -1.9 to 0 seconds. The output consists of 10 future XY positions at +0.1 to +1.0 seconds. Features combine positions centered on the last observation with finite-difference velocities, giving shape `(B,20,4)`. Normalization statistics are computed from the training set only. A manually implemented Elman RNN encodes the history; its final hidden state produces 20 coordinate residuals in one step to correct a constant-velocity forecast fitted to historical positions. A manually implemented GRU-style gated RNN is provided for comparison.

Defaults: hidden size 32, AdamW learning rate 0.001, batch size 128, 100 epochs, seed 42, MSE loss, and gradient norm clipping at 1.0. Checkpoints are selected by validation ADE, followed by final test evaluation. Training, validation, and test sets are split by source motion; overlapping windows must not be randomly distributed across sets.

## Results from a verified run

Python 3.12, CPU, PyTorch 2.15.0, NumPy 2.5.4, and 100 epochs per model:

| Model | Test ADE (m) | Test FDE (m) |
| --- | ---: | ---: |
| Last-position baseline | 0.3554 | 0.6207 |
| Constant-velocity baseline | 0.1842 | 0.3754 |
| Elman RNN + constant-velocity residual | **0.1146** | **0.2392** |
| Gated RNN + constant-velocity residual | 0.1192 | 0.2444 |

Elman reduces average displacement error by approximately 37.8% relative to the constant-velocity baseline, with an endpoint error of approximately 23.9 cm at one second. The gated model does not outperform Elman on this small dataset; this result is retained for student analysis. These metrics measure distance error, not classification accuracy: ADE averages Euclidean distance over future time steps, while FDE measures Euclidean distance at the final predicted time step.

These are short-term 2D prediction results on retargeted reference motions and should not be interpreted as real-robot navigation accuracy. Hardware and dependency versions may produce numerical differences. Full metrics, averages giving equal weight to each source motion, training records, and checkpoints are in `outputs/verified_cpu/`. The instructor notebook also contains outputs from its independent execution.

## Files

- `ASSIGNMENT.md`: tasks, equations, tensor shapes, a 100-point rubric, and submission requirements.
- `DATASETS.md`: verification of both candidates, selection rationale, versions, licensing, and task scope.
- `solution.py` / `student.py`: corresponding script versions; the student version contains no answer implementations.
- `trajectory_utils.py`: shared grouped splits, normalization, experiment orchestration, saving, and plotting.
- `check_assignment.py`: public checks for shapes, gradients, metrics, training, and split isolation.
- `predict.py`: predicts future absolute positions from a trained checkpoint and historical XY positions.
- `prepare_phuma.py`: reconstructs the subset from the pinned official release; classroom execution does not require this step.
- `build_notebooks.py` / `verify_notebooks.py`: instructor tools for rebuilding notebooks and verifying answer correspondence and execution.
- `package_release.py`: creates separate instructor and student ZIP packages.
- `translation_verification.json`: records code, output, and data equivalence checks for the English edition.
- `data/`: a lightweight dataset requiring no pickle loading, a per-file provenance manifest, and source documentation.

## Relationship to the reference assignment

The reference file available in the workspace is `emen_7029_7104_2026_asg1_hw_torch.py`, a linear/polynomial regression assignment. This material follows its teaching format: explain tasks and marks, provide shared data and plotting, leave key implementation steps as TODOs, and require independent testing of generalization. The learning task is replaced with sequential trajectory prediction. The `f/g/initial_recurrent_state/scan` interfaces from the existing RNN draft are also retained to support manual implementation of recurrence.

You may distribute `humanoid_rnn_student.ipynb` directly. The student ZIP also includes instructions, scripts, data, and public checks, but excludes instructor solutions, weights, and the executed instructor notebook. The instructor ZIP includes the complete solution and verified results.
