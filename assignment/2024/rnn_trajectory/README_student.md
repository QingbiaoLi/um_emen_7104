# RNN 车辆轨迹预测：学生包

请先阅读 `ASSIGNMENT_zh.md`。本包包含真实 NGSIM 车辆轨迹、公共工具和 13 处留空的 `student.py`，不含教师答案或训练完成的参考权重。

推荐 Python 3.12，在本目录运行：

```bash
python -m pip install -r requirements.txt
```

完成 `student.py` 中的 TODO 后：

```bash
python check_assignment.py --module student
python student.py --epochs 40 --output outputs/student
```

数据已附在 `data/ngsim_i80.csv.gz` 中，安装依赖后可离线训练。结果保存到 `outputs/student/`。未填空时的 `NotImplementedError` 是预期行为；不要改动公共数据划分与评估逻辑。

若要匹配参考实验的依赖版本，可使用 `python -m pip install -r requirements-tested.txt`。输入历史与预测说明、评分标准及提交要求见作业文档。
