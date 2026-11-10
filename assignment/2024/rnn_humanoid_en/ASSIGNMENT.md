# Predicting Humanoid Trajectories with RNNs (100 Points)

Goal: manually implement RNNs on the PHUMA Unitree G1 walking subset to predict the next one second of root trajectory from historical planar positions. After completing the 13 TODO blocks, the student notebook should provide the same functionality as the complete instructor version. Data loading, grouped splits, normalization, validation-based model selection, and plotting are provided.

This assignment uses the interfaces `f(h,x)`, `g(h)`, `initial_recurrent_state(batch_size)`, and `scan`. To learn recurrence and gradient propagation, do not call `nn.RNN/GRU/LSTM` or their Cell variants, and do not override the model's `forward` method. You may use `nn.Linear`, tensor operations, and optimizers.

## 1. Data and time conventions

The data come from the [official public PHUMA release](https://huggingface.co/datasets/DAVIAN-Robotics/PHUMA). They are retargeted robot reference motions, not execution logs from a physical robot. The subset includes simple G1 walking motions with accompanying upper-body actions. Horizontal XY coordinates from `root_trans` are in meters. Sampling every third frame reduces the original 30 Hz rate to 10 Hz. Do not modify the fixed dataset or select test examples based on prediction error.

- `B`: batch size; `O=20`: number of observed points; `H=10`: number of future points.
- Historical times are `-1.9,-1.8,...,0` seconds, spanning 1.9 seconds and commonly described as approximately two seconds of history.
- Future times are `0.1,0.2,...,1.0` seconds; `dt=0.1`.
- Input dimension `D_x=4`, hidden dimension `D_h=32`, and output dimension `D_y=2H=20`.
- Clips/chunks from the same source motion stay in the same split. Split source motions before windowing; windows must not cross segment boundaries.
- Estimate means and standard deviations from the training set only. Inputs, velocities, baselines, and inference must not use future ground truth.

## 2. Tasks and grading

| Task | Points | TODO identifiers |
| --- | ---: | --- |
| Historical position and velocity features | 10 | `features` |
| Elman RNN parameters, update, output, and initial state | 20 | `elman_init / elman_f / elman_g / elman_h0` |
| Temporal scan that preserves gradients | 15 | `scan` |
| Multi-step residual forecasting | 10 | `predict` |
| One-epoch training loop | 15 | `train` |
| ADE / FDE in meters | 10 | `metrics` |
| Custom gated RNN | 15 | `custom_init / custom_f / custom_g / custom_h0` |
| Experimental results and interpretation | 5 | Report |

## A. Historical features, 10 points

`make_features(observed,dt)` takes input of shape `(B,O,2)` and returns `(B,O,4)`. The first two channels contain each historical position minus the last observed position. The last two channels contain differences between consecutive historical positions divided by `dt`. Copy the second velocity entry into the first entry. The full history is available when forecasting begins, so this copy does not use information from the prediction interval.

Preserve the time dimension when selecting the last observed position to allow broadcasting. Do not normalize in this function; the shared code applies normalization using training-set statistics only.

## B. Elman RNN, 20 points

Implement `h_t = tanh(W_x x_t + b_x + W_h h_(t-1))` and `g(h)=W_o h+b_o`.

Use `Linear(d_x,d_h)` for the input mapping, `Linear(d_h,d_h,bias=False)` for the recurrent mapping, and `Linear(d_h,d_y)` for the output mapping. Store `self.d_h`, initialize the recurrent weights orthogonally, and initialize both output weights and output bias to zero. This makes the initial forecast equal to the constant-velocity baseline.

`f` returns `(B,D_h)`, and `g` returns `(B,2H)`. The initial hidden state is a zero tensor of shape `(B,D_h)` whose device and dtype match the model parameters.

## C. scan, 15 points

The input `xs` has shape `(B,O,D_x)`. Iterate over time dimension 1, calling `f` to update the hidden state at each step, and return `(B,O,D_h)`. A Python loop over time is allowed; process the entire batch together at each step. Do not call `detach()`, convert to NumPy, or reconstruct tensors in ways that break the computation graph. The final hidden state must retain gradients to earlier inputs.

## D. Multi-step forecasting, 10 points

The provided helper fits velocity `v_hat` by least squares over the last six historical positions, which span 0.5 seconds. It produces the constant-velocity forecast relative to the last observation: `baseline[k] = v_hat * (k+1) * dt`.

In `predict(rnn,xs,baseline,output_scale)`, initialize the hidden state, call `scan`, take the last hidden state, apply `g`, and reshape the output to `(B,H,2)`. The forecast is:

`prediction = baseline + output_scale * residual`

`output_scale=1.0` represents a fixed one-meter scale, not a parameter estimated from the test set. Return displacement relative to the last observed point, in meters. To obtain absolute positions, add the last observed position. Predict all future steps in one pass, without teacher forcing or feeding future ground-truth positions back into the model.

## E. One training epoch, 15 points

For each `(xs,baseline,target)` batch in `train_one_epoch`, enable training mode and move the tensors to the specified device. Clear gradients, predict, compute `mean(((prediction-target)/output_scale)^2)`, backpropagate, clip the gradient norm to 1.0, and update the optimizer.

Return the epoch's mean loss weighted by the number of samples, as a Python float. The final batch may be smaller, so do not average batch losses with equal weights. The shared experiment code uses AdamW and saves the best model according to validation ADE at each epoch. The test set is used for final reporting.

## F. ADE / FDE, 10 points

First calculate `distance[b,k] = sqrt((pred_x-target_x)^2+(pred_y-target_y)^2)`. Then return `ADE[b]=mean_k(distance[b,k])` and `FDE[b]=distance[b,H-1]`. Both tensors have shape `(B,)` and units of meters. Coordinate RMSE is not a substitute for ADE.

## G. Custom gated RNN, 15 points

Keep the same interfaces. You may implement the following GRU-style recurrence manually, where `*` denotes elementwise multiplication:

```text
r = sigmoid(W_xr x + b_r + W_hr h)
z = sigmoid(W_xz x + b_z + W_hz h)
n = tanh(W_xn x + b_n + r * (W_hn h))
h_new = z * h + (1-z) * n
```

Each input and recurrent mapping produces `3D_h` values, split into three groups along the last dimension. The recurrent mapping has no bias. Initialize each of the three recurrent weight blocks orthogonally and initialize the output head to zero. Other recurrent architectures are allowed if you explain their equations and respect the shared interfaces. Keep data splits, input information, and training budgets fixed when comparing with Elman. A gated model is not required to outperform Elman.

## H. Experiments and submission, 5 points

Submit the completed student notebook with its execution outputs and a short analysis. If using scripts, submit `student.py` and the experimental outputs. Include at least the model equations, training configuration, validation curves, ADE/FDE for the baselines and both RNNs, test trajectory plots, and one failure case with possible explanations.

Explain why source motions are grouped before splitting, why normalization uses only training data, why an RNN may outperform constant velocity, and why a more complex model may not perform better. Choose hyperparameters using the validation set. Do not repeatedly tune against test results after evaluating the test set. Failure cases may support interpretation, but must not be used to remove test data after observing errors.

A suggested classroom target for the fixed split is **ADE <= 0.15 m and FDE <= 0.30 m, with better performance than the constant-velocity baseline**. Allow reasonable variation across dependency versions and hardware. Grading should emphasize implementation correctness, sound experimental practice, and analysis rather than one specific decimal value. The reference results use robot reference motions and do not establish real-robot deployment accuracy.

After completing the Elman components, set `MODELS` to `"elman"` in the notebook for an intermediate experiment. Set it back to `"both"` after completing all tasks. After modifying a function definition, rerun its cell before rerunning the checks and training. The unfilled student's `NotImplementedError` identifies the TODO that needs to be completed.
