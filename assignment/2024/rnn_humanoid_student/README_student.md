# 人形机器人轨迹预测：学生版

推荐单独打开 `humanoid_rnn_student.ipynb`。上传到 https://colab.research.google.com/ 后选择 Python 3 CPU runtime；数据、来源清单及公共代码都已内置。完成 13 个 TODO，再运行检查、训练与结果单元。第一次运行前请先阅读 notebook 的任务说明。

无需下载整个 PHUMA，不需要 Kaggle 或 Hugging Face 账号。你将使用 PHUMA G1 行走参考动作，根据约 2 秒历史预测未来 1 秒 XY 根节点位置。任务要求、评分和提交方式见 `ASSIGNMENT_zh.md`；数据适用性和许可见 `DATASETS_zh.md`。

本地 Jupyter 安装：

```bash
python -m pip install -r requirements-notebook.txt
jupyter notebook humanoid_rnn_student.ipynb
```

如果选择脚本形式，只修改 `student.py` 中的 TODO：

```bash
python -m pip install -r requirements.txt
python check_assignment.py --module student --model elman
python student.py --model elman --epochs 100 --output outputs/my_elman
# 全部完成后：
python check_assignment.py --module student
python student.py --epochs 100 --output outputs/my_run
```

脚本与 notebook 是同一作业的两种入口。选择一种完成即可，修改不会自动同步到另一种。公共函数和检查无需修改。新的实验使用新的输出目录。

本包不含教师答案或预训练权重。未完成 TODO 时出现 `NotImplementedError` 是预期行为。全部完成后，使用训练/验证集开发模型，最后报告测试 ADE/FDE、轨迹图和你的分析。
