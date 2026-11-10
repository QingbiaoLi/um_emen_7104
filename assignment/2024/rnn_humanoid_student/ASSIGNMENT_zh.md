# 用 RNN 预测人形机器人未来轨迹（100 分）

目标：在 PHUMA 的 Unitree G1 行走子集上，手写 RNN，使用历史平面位置预测未来 1 秒的根节点轨迹。完成学生 notebook 的 13 个 TODO 后，它应与完整教师版具有相同功能。公共数据读取、分组划分、标准化、验证集选择与绘图均已提供。

本作业使用 `f(h,x)`、`g(h)`、`initial_recurrent_state(batch_size)` 和 `scan` 接口。为学习递归与梯度传播，不调用 `nn.RNN/GRU/LSTM` 及其 Cell，也不覆盖模型的 `forward`；可以使用 `nn.Linear`、张量运算和优化器。

## 1. 数据与时间定义

数据来自 [PHUMA 官方公开发布](https://huggingface.co/datasets/DAVIAN-Robotics/PHUMA)，为重定向的机器人参考动作，而非实机执行日志。采用 G1 的简单行走/伴随上肢动作，原始 `root_trans` 的水平 XY 单位为米，30 Hz 每三帧取一帧成为 10 Hz。不要修改固定数据或通过预测误差筛选测试样本。

- `B`：batch size；`O=20`：历史点数；`H=10`：未来点数。
- 历史时刻为 `-1.9,-1.8,...,0` 秒，覆盖 1.9 秒间隔，通常称约 2 秒历史。
- 未来时刻为 `0.1,0.2,...,1.0` 秒；`dt=0.1`。
- 特征维数 `D_x=4`，隐层 `D_h=32`，输出维数 `D_y=2H=20`。
- 同一源动作的 clips/chunks 留在同一集合；先划分源动作再滑窗，窗口不能跨片段。
- 仅训练集估计均值、标准差；输入、速度、基线和推理均不读取未来真值。

## 2. 任务与评分

| 任务 | 分值 | TODO 标识 |
| --- | ---: | --- |
| 历史位置和速度特征 | 10 | `features` |
| Elman RNN 参数、更新、输出、初始状态 | 20 | `elman_init / elman_f / elman_g / elman_h0` |
| 保留梯度的时间扫描 | 15 | `scan` |
| 未来多步残差预测 | 10 | `predict` |
| 一个 epoch 的训练循环 | 15 | `train` |
| 米制 ADE / FDE | 10 | `metrics` |
| 自定义门控 RNN | 15 | `custom_init / custom_f / custom_g / custom_h0` |
| 实验结果与解释 | 5 | 报告 |

## A. 历史特征，10 分

`make_features(observed,dt)`：输入 `(B,O,2)`，输出 `(B,O,4)`。前两维为每个历史位置减去最后观测位置；后两维为相邻历史位置差除以 `dt`。第一项速度复制第二项速度。整段历史在预测时已经可见，因此此复制不使用预测区间的信息。

保留最后观测点的时间维度以便广播；不要在此函数中标准化，公共代码会只用训练集统计量完成标准化。

## B. Elman RNN，20 分

实现 `h_t = tanh(W_x x_t + b_x + W_h h_(t-1))`，以及 `g(h)=W_o h+b_o`。

输入映射用 `Linear(d_x,d_h)`，递归映射用 `Linear(d_h,d_h,bias=False)`，输出映射用 `Linear(d_h,d_y)`。保存 `self.d_h`，递归权重正交初始化，输出权重和偏置置零，使初始预测等于匀速基线。

`f` 返回 `(B,D_h)`，`g` 返回 `(B,2H)`。初始隐状态为 `(B,D_h)` 的零张量，其 device 和 dtype 必须与参数一致。

## C. scan，15 分

输入 `xs` 为 `(B,O,D_x)`。沿时间维度 1 逐步调用 `f` 更新隐状态，返回 `(B,O,D_h)`。允许 Python 时间循环；每一步同时处理整个 batch。不能 `detach()`、转 NumPy 或重建破坏计算图的张量。最后隐状态应保留通往早期输入的梯度。

## D. 多步预测，10 分

公共函数以最后 6 个历史位置（0.5 秒跨度）最小二乘拟合速度 `v_hat`，给出相对末帧的匀速预测 `baseline[k] = v_hat * (k+1) * dt`。

`predict(rnn,xs,baseline,output_scale)` 先初始化隐状态并 `scan`，取最后状态，经 `g` 输出后 reshape 为 `(B,H,2)`。预测公式：

`prediction = baseline + output_scale * residual`

`output_scale=1.0` 表示固定的一米尺度，不是从测试集估计出的参数。返回相对最后观测点的米制位移；绝对位置须再加最后观测位置。所有未来步一次生成，无 teacher forcing，也不把真实未来点回填给模型。

## E. 单轮训练，15 分

`train_one_epoch` 每次读入 `(xs,baseline,target)`，切换训练模式并将张量移到指定 device。清空梯度，预测，计算 `mean(((prediction-target)/output_scale)^2)`，反向传播，梯度范数裁剪到 1.0，再更新优化器。

返回按样本数加权的 epoch 平均损失（Python float）。最后一个 batch 可能更小，不能直接等权平均各 batch loss。公共代码使用 AdamW，每轮只根据验证 ADE 保存最佳模型；测试集用于最后报告。

## F. ADE / FDE，10 分

先计算 `distance[b,k] = sqrt((pred_x-target_x)^2+(pred_y-target_y)^2)`，再返回 `ADE[b]=mean_k(distance[b,k])` 和 `FDE[b]=distance[b,H-1]`。两者形状均为 `(B,)`，单位米。不能用坐标 RMSE 替代 ADE。

## G. 自定义门控 RNN，15 分

保持相同接口，可按以下 GRU 形式手写门控递归（`*` 为逐元素乘法）：

```text
r = sigmoid(W_xr x + b_r + W_hr h)
z = sigmoid(W_xz x + b_z + W_hz h)
n = tanh(W_xn x + b_n + r * (W_hn h))
h_new = z * h + (1-z) * n
```

输入和递归映射各输出 `3D_h` 维，沿最后一维分为三组。递归映射不加偏置，对三组递归权重分别正交初始化，输出头同样置零。允许提出其他递归结构，但需要说明公式并遵守公共接口。与 Elman 比较时固定数据划分、输入信息和训练预算；不要求门控模型必然优于 Elman。

## H. 实验与提交，5 分

提交完成的学生 notebook（保留运行输出）及简短分析；若使用脚本则提交 `student.py` 和实验输出。报告至少包含：模型公式、训练配置、验证曲线、基线和两个 RNN 的 ADE/FDE、测试轨迹图、一个失败案例及可能原因。

解释为什么按源动作分组、为什么只用训练集标准化、为什么 RNN 可能优于匀速、为什么更复杂模型未必更好。改变超参数时依据验证集选择；测试集评估后不再反复据此调参。测试图中的失败案例可用于解释，不能用于事后筛除测试数据。

参考教学目标：同一固定划分下 **ADE <= 0.15 米、FDE <= 0.30 米，并优于匀速基线**。依赖版本与硬件允许合理浮动；评分以实现正确、实验规范和分析为主，不把某个小数值作为唯一标准。参考结果来自机器人参考动作，不能据此宣称实机部署精度。

先完成 Elman 后可将 notebook 的 `MODELS` 改为 `"elman"` 进行阶段性实验；全部完成后设回 `"both"`。修改函数定义后要重新运行对应单元，再运行检查与训练。原始学生版的 `NotImplementedError` 提醒你完成相应 TODO。
