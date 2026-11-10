# 数据集选择与任务边界

核验日期：2026-10-30。本实验未使用 NGSIM，也未使用人工生成的 Bézier 曲线代替真实数据文件中的轨迹。

## 两个候选数据集

| 项目 | Kaggle Humanoid Robot Navigation | PHUMA（本实验采用） |
| --- | --- | --- |
| 原始链接 | https://www.kaggle.com/datasets/zara2099/humanoid-robot-navigation-dataset | https://arxiv.org/html/2510.26236v3 |
| 获取情况 | 实际下载并检查 CSV，1,000 行、13 列 | 官方 Hugging Face 公开发布，可固定版本获取 |
| 数据内容 | 导航状态、地形、能耗、稳定性、Bézier 控制点等 | G1/H1-2 的逐帧根节点位置、朝向、关节位置、帧率 |
| 连续位置预测适用性 | 缺少逐帧实际位置、采样间隔和可靠的 episode 边界，不适合直接构造本任务 | `root_trans` 和 `fps` 可直接构造历史/未来窗口 |
| 适合的简单任务 | 导航结果分类或分数回归；需另行核验数据来源 | G1 行走动作的根节点二维短期预测 |
| 发布方许可声明 | CC0 | Hugging Face 数据卡：Apache-2.0；同时保留上游来源限制 |

Kaggle CSV 的实际字段：`time_step, robot_id, terrain_type, obstacle_count, obstacle_distance, bezier_control_points, planned_trajectory_length, mpc_adjustment, gait_stability_index, center_of_mass_offset, energy_consumption, navigation_status, navigation_score`。
字段核验记录和两行样本见 `data/sources/kaggle_schema_verified.json`。`time_step` 与 `robot_id` 的存在本身不能证明各行构成等间隔的连续运动。Bézier 控制点描述规划路径，不能当成已执行轨迹的观测点。

## PHUMA 来源

- 论文：Lee et al., *PHUMA: Physically Reliable Humanoid Locomotion Dataset*, arXiv:2510.26236v3。
- 官方代码：https://github.com/DAVIAN-Robotics/PHUMA
- 项目页：https://davian-robotics.github.io/PHUMA/
- 官方数据：https://huggingface.co/datasets/DAVIAN-Robotics/PHUMA
- 本次固定 revision：`48b0f5982969104ab041461e594a8b8c36db8ba5`。
- 原始结构：`root_trans (T,3)`、`root_ori (T,4)`、`dof_pos (T,D)`、`fps`。

PHUMA 将人类动作数据经过物理约束处理并重定向到人形机器人。公开文件中的这些轨迹是**机器人参考动作**，不能描述为实机传感器测得的执行日志。论文的主要任务是动作模仿与控制；这里另行设计的是有监督轨迹预测教学实验，不是复现论文主实验。

## 可复现的小型子集

选取 `data/g1/idea400/` 中文件名包含 `walking` 且匹配 `number_|hand_pockets|arm_swing|clapping|salute|waving|pointing` 的动作。内容为行走及简单上肢动作，排除其他机器人、其他来源以及复杂运动类别。不是严格的“直线匀速行走”标签。

共 295 个片段、169 个按名称归并的源动作、29,856 个原始 30 Hz 帧。每隔 3 帧采样一次，得到 10 Hz 的 10,006 帧；保存 XYZ，模型只用水平 XY。NPZ 约 111 KiB，内置于两个 notebook，无需账号或下载完整约 3.4 GB 压缩包。`prepare_phuma.py` 可通过 HTTP Range 从固定官方版本重新提取同一子集。

筛选不依赖预测误差。每个原始文件的路径、SHA-256、CRC、帧率及长度均记录在 `data/manifest.json`。已逐一核对缓存的 295 个原始文件哈希，并验证 NPZ 坐标与原始 `root_trans[::3]` 完全一致。读取课堂 NPZ 使用 `allow_pickle=False`；重建工具仅允许原始 NPY 字典所需的 NumPy 类型。

以去掉 `_clip...` / `_chunk...` 后缀的源名称分组，先划分再滑窗；不同片段不拼接。默认窗口需要至少 30 帧，故最终有 157 个源动作可用：训练 109、验证 23、测试 25；对应窗口数 794、172、191。其余 12 个源动作不满足长度要求。

## 结果解释的限制

1. 按源名称隔离可避免同一源动作的分块进入不同集合；缺少可靠的演员 ID，不能声称演员身份完全独立，也不能排除不同名称动作间的相关性。
2. 相邻滑窗有重叠。报告同时提供逐窗口平均和逐源动作等权平均，不能把所有窗口当作独立样本来计算置信区间。
3. 我们没有新增平滑或插值；PHUMA 上游本身包含动作处理与物理优化。此结果反映已处理参考轨迹的可预测性，不代表带噪实机在线传感器表现。
4. XY 预测不包含全身关节、朝向、障碍物、接触和控制输入，不能直接用于机器人控制或保证动力学可行性。未来意图的多解性也没有被确定性 RNN 建模。
5. 当前是一个固定划分、固定种子的课堂结果。跨场景、跨来源、跨机器人、多种子与更长预测时域需要独立评估。

## 许可与引用

保留 `data/sources/PHUMA_DATA_CARD.md`、`PHUMA_LICENSE.txt` 和 `PHUMA_REPOSITORY_README.md`。官方数据卡明确说明公开包不包含受限制的 LAFAN1 与 LocoMuJoCo 衍生动作；本子集仅取公开包中的 IDEA400 来源。Apache-2.0 是发布方对该数据仓库的声明，不意味着可以忽略原始人体动作来源的权利或把论文的 CC BY 4.0 当成数据许可。进一步公开分发或商业使用时应核对上游 Motion-X / IDEA400 条款。

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
