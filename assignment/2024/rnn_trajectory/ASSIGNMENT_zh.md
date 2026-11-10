# Assignment 5：用 RNN 预测车辆未来轨迹

目标：使用 NGSIM 真实车辆位置，预测未来 3 秒的二维轨迹。你将手写 Elman RNN 和 `scan`，完成训练与评估，并设计一个改进的递归模型。

本作业沿用参考作业的 `f`、`g`、`initial_recurrent_state` 接口。作为本次教学要求，请勿调用 `torch.nn.RNN/GRU/LSTM` 及其 `Cell`，也不要在模型中覆盖 `forward`。可以使用 `nn.Linear`、`nn.Parameter`、张量运算、激活函数和优化器。你只需要修改 `student.py` 中的 TODO。

## 数据与预测设定

数据由 [USDOT/FHWA NGSIM](https://data.transportation.gov/Automobiles/Next-Generation-Simulation-NGSIM-Vehicle-Trajectories/8ect-6jqj) 提供，已附在 `data/` 中；`manifest.json` 记录来源与查询。单位从英尺转换成米，由 10 Hz 降采样为 5 Hz。程序先按车辆划分训练、验证、测试集合，再构造窗口。

- `O=15` 个历史点：`t=-2.8,-2.6,...,0` 秒。
- `H=15` 个未来点：`t=0.2,0.4,...,3.0` 秒。
- `B` 是 batch size，`D_h=64` 是隐状态维度，`D_x=4`，`D_y=2H=30`。
- 不能用真实未来位置构造输入、速度、归一化参数或预测基线。
- `trajectory_utils.py` 已提供按车辆分割、训练集特征标准化、数据加载、验证集选模型、最终测试与绘图。

程序已经提供一个合理基线：对最后 1 秒的历史位置拟合速度，假定未来保持此速度。RNN 负责根据历史修正这个基线。两个模型均使用此基线，保证比较公平。

## 任务与评分，共 100 分

| 任务 | 分值 | TODO 标识 |
| --- | ---: | --- |
| 构造历史特征 | 10 | `features` |
| Elman RNN 参数、更新、输出、初始状态 | 20 | `elman_init / elman_f / elman_g / elman_h0` |
| 实现保留梯度的时间扫描 | 15 | `scan` |
| 从最后隐状态预测所有未来位置 | 10 | `predict` |
| 单个 epoch 训练循环 | 15 | `train` |
| 正确计算米制 ADE 与 FDE | 10 | `metrics` |
| 自定义 RNN | 15 | `custom_init / custom_f / custom_g / custom_h0` |
| 完整实验、曲线与误差分析 | 5 | 报告 |

### A. 历史特征，10 分

`make_features(observed, dt)` 输入 `(B,O,2)` 的位置，输出 `(B,O,4)`：

1. 前两维为历史每个位置减去最后一个已观测位置。
2. 后两维为相邻历史点的位置差除以 `dt`，单位米/秒。
3. 第一个历史点没有前一帧，使用第二个历史点的速度填入第一项。
4. 这里只返回物理量特征；标准化由公共程序完成。

提示：计算最后观测点时保留时间维度，方便广播。整条历史序列在发起预测时都已观测，因此第三步不使用未知未来。

### B. Elman RNN，20 分

实现公式：

`h_t = tanh(W_x x_t + b_x + W_h h_(t-1))`

`g(h) = W_o h + b_o`

用 `nn.Linear(d_x,d_h)` 作为输入映射，递归映射不加偏置，输出映射为 `nn.Linear(d_h,d_y)`。递归权重使用正交初始化，输出权重与偏置初始化为零；这使初始预测等于匀速基线。保存 `self.d_h`，不要依赖文件级全局变量。

- `f(h,x)`：`(B,D_h),(B,D_x) -> (B,D_h)`。
- `g(h)`：`(B,D_h) -> (B,2H)`。
- `initial_recurrent_state(batch_size)`：返回 `(B,D_h)` 零张量，设备和 dtype 应与模型参数相同。

### C. scan，15 分

`scan(f,h,xs)` 的输入 `xs` 为 `(B,O,D_x)`。沿时间维度依次更新隐状态，返回 `(B,O,D_h)`。

允许使用 Python 循环；batch 维度应由张量运算一起处理。不能在循环中 `detach()`、转 NumPy 或创建破坏梯度的新张量。最后时间步的损失应当能反向传播至早期时间步。

### D. 多步预测，10 分

`predict(rnn,xs,baseline,output_scale)` 返回 `(B,H,2)` 的未来相对位置：

1. 生成初始隐状态，对历史调用 `scan`。
2. 取最后隐状态，调用 `g` 得到 `(B,2H)`。
3. 将输出变形为 `(B,H,2)`，乘固定尺度 `output_scale=10.0`。
4. 将残差与米制匀速基线相加。

这是直接多步预测，函数没有未来真值参数。所有未来步一次生成，无需把真实未来点回填给 RNN。

### E. 训练，15 分

`train_one_epoch` 遍历公共 DataLoader，每个 batch 为 `(xs,baseline,target)`。

- 切换为训练模式，将三个张量移到 `device`。
- 清梯度，计算预测。
- 损失：`mean(((prediction-target)/output_scale)^2)`，对 batch、时间、坐标三个维度求平均。
- 反向传播，对参数做梯度范数裁剪 `max_norm=1.0`，更新优化器。
- 返回按样本数加权的 epoch 平均损失，类型为 Python float。最后一个 batch 可能较小，不能对 batch loss 直接等权平均。

不要在此函数中读取测试集。公共程序会每个 epoch 评估验证集，选择最小验证 ADE 的 checkpoint，最后计算测试成绩。

### F. ADE/FDE，10 分

对每个窗口 `b` 的未来时间 `k`，先计算二维欧氏距离：

`distance[b,k] = sqrt((pred_x-target_x)^2 + (pred_y-target_y)^2)`

`ADE[b] = mean_k(distance[b,k])`

`FDE[b] = distance[b,H-1]`

返回两个形状均为 `(B,)` 的张量。注意先计算每个时刻的欧氏距离，再对时间平均；这与“坐标平方误差取平均后开方”不同。

### G. 自定义 RNN，15 分

保留 `f/g/initial_recurrent_state` 接口，设计改进结构。可手写 GRU 形式的门控更新：

`r = sigmoid(W_xr x + b_r + W_hr h)`

`z = sigmoid(W_xz x + b_z + W_hz h)`

`n = tanh(W_xn x + b_n + r * (W_hn h))`

`h_new = z * h + (1-z) * n`

`*` 表示逐元素乘法。可以用一个输出为 `3D_h` 的 Linear 映射，再沿最后维度切为三组。门控模型同样使用零初始化输出头；递归矩阵的三组权重分别做正交初始化。

也可尝试其他递归结构。比较时固定数据划分、窗口长度、训练预算和输入信息，只根据验证集选择超参数。报告改进模型是否优于 Elman，并解释结果；不要求每个随机种子下门控模型都必然更优。

### H. 实验与报告，5 分

至少包含：模型结构说明、随机种子、验证曲线、两个 RNN 与匀速基线的 ADE/FDE、若干测试轨迹图，以及一个失败案例分析。说明训练集归一化、按车辆划分和禁止真实未来输入各解决了什么问题。

参考运行的 Elman 测试 ADE 约 0.98 米，门控版本约 0.95 米。教学精度目标可取 **ADE <= 1.10 米且 FDE <= 2.35 米**，并优于同次实验的匀速基线；依赖版本差异应允许合理浮动。成绩应综合实现正确性、实验规范和分析，不靠反复查看测试集调参以达到某个小数值。

## 运行与提交

推荐 Python 3.12，在本目录执行：

```bash
python -m pip install -r requirements.txt

# 完成基础部分后，只检查 Elman
python check_assignment.py --module student --model elman

# 所有 TODO 完成后
python check_assignment.py --module student
python student.py --epochs 40 --output outputs/student
```

未填空时出现 `NotImplementedError` 是预期行为。公开检查通过仅说明部分实现性质正确，仍需完成真实数据实验。

输出包含最优权重、`metrics.json`、`splits.json`、`learning_curves.png`、`trajectories.png`、`error_by_horizon.png` 和测试集预测。推理示例：

```bash
python predict.py --module student \
  --checkpoint outputs/student/gated_best.pt \
  --history data/example_history.npy \
  --output outputs/student/example_forecast.npy
```

提交：补齐后的 `student.py`、实验配置与指标 JSON、结果图和简短报告。请保留原始分割和数据来源；不要提交修改测试集、归一化时读取未来标签、或从答案文件导入实现的版本。
