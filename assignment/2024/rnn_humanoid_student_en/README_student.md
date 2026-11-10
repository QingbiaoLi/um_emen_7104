# Humanoid Trajectory Prediction: Student Edition

Start with `humanoid_rnn_student.ipynb`. Upload it to https://colab.research.google.com/ and select a Python 3 CPU runtime. The data, provenance manifest, and shared code are included. Complete the 13 TODO blocks, then run the checks, training, and results cells. Read the task descriptions in the notebook before your first run.

You do not need to download the entire PHUMA dataset or obtain a Kaggle/Hugging Face account to access the bundled data. Using PHUMA G1 walking reference motions, you will predict the next one second of root XY positions from approximately two seconds of history. See `ASSIGNMENT.md` for requirements, grading, and submission instructions, and `DATASETS.md` for dataset suitability and licensing.

For local Jupyter:

```bash
python -m pip install -r requirements-notebook.txt
jupyter notebook humanoid_rnn_student.ipynb
```

If you choose the script workflow, edit only the TODO blocks in `student.py`:

```bash
python -m pip install -r requirements.txt
python check_assignment.py --module student --model elman
python student.py --model elman --epochs 100 --output outputs/my_elman
# After completing all tasks:
python check_assignment.py --module student
python student.py --epochs 100 --output outputs/my_run
```

The scripts and notebook are two entry points for the same assignment. Choose one; edits do not automatically synchronize between them. You do not need to modify the shared functions or checks. Use a new output directory for each experiment.

This package contains no instructor solutions or pretrained weights. A `NotImplementedError` is expected until the corresponding TODO is completed. Develop your model using the training and validation sets, then report test ADE/FDE, trajectory plots, and your analysis.
