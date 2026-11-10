"""Build standalone teaching notebooks from the shared, tested Python sources."""
import ast
import base64
import hashlib
import io
from pathlib import Path
import re
import zipfile
import nbformat as nbf
ROOT = Path(__file__).resolve().parent

def function_sources(filename):
    source = (ROOT / filename).read_text()
    lines = source.splitlines()
    return {node.name: '\n'.join(lines[node.lineno - 1:node.end_lineno]) for node in ast.parse(source).body if isinstance(node, (ast.FunctionDef, ast.ClassDef))}

def build(student):
    kind = 'student' if student else 'instructor'
    notebook = nbf.v4.new_notebook()
    notebook.metadata = {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}, 'language_info': {'name': 'python', 'version': '3.12'}, 'colab': {'name': f'humanoid_rnn_{kind}.ipynb', 'provenance': []}}
    cells = notebook.cells

    def md(source):
        cells.append(nbf.v4.new_markdown_cell(source.strip()))

    def code(source, cell_id, collapsed=False):
        cell = nbf.v4.new_code_cell(source.strip())
        cell.id = cell_id
        if collapsed:
            cell.metadata.update({'jupyter': {'source_hidden': True}, 'cellView': 'form'})
        cells.append(cell)
    title = 'Student Exercise' if student else 'Complete Instructor Solution'
    md(f"# RNN Humanoid Trajectory Prediction - {title}\n\nEMEN 7029 / 7104 - PyTorch - PHUMA / Unitree G1\n\n**Task: use 20 historical XY positions to predict the next 10 positions (one second).**\nUse the open PHUMA G1 walking subset to implement an Elman RNN and a gated RNN manually, and compare them with a constant-velocity baseline.\nThis task predicts 2D root trajectories from robot **reference motions**; it does not use real-robot execution logs or perform whole-body control.\n\nUpload this notebook to [Google Colab](https://colab.research.google.com/) and select a Python 3 CPU runtime.\nThe data and shared code are embedded, so this single notebook is sufficient; no data-service account is required to obtain the subset.\nFor local use, Python 3.12 and Jupyter are recommended. Execution creates `humanoid_rnn_workspace/` in the current directory.\n\n{('Complete the 13 TODO blocks below, then run the notebook from top to bottom. Do not modify the shared code. Unfilled blocks intentionally raise NotImplementedError.' if student else 'Run all cells from top to bottom to retrain. Saved outputs come from an actual CPU execution; execution does not depend on those outputs.')}\n\nThe teaching format follows the reference assignment: task description -> TODO implementation -> visual verification -> generalization analysis.\n")
    md('## 0. Environment\n\nSmall models are trained on the CPU by default. This cell does not access the network when the required libraries are already installed; it installs missing libraries automatically.\nThe full experiment trains each model for 100 epochs. Runtime depends on your machine.\n')
    code('import importlib.util\nimport os\nfrom pathlib import Path\nimport subprocess\nimport sys\n\nrequirements = {"numpy": "numpy>=1.26,<3", "torch": "torch>=2.2,<3",\n                "matplotlib": "matplotlib>=3.8,<4"}\nmissing = [spec for name, spec in requirements.items()\n           if importlib.util.find_spec(name) is None]\nif missing:\n    subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])\nROOT = Path.cwd() / "humanoid_rnn_workspace"\nROOT.mkdir(exist_ok=True)\nos.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))\nos.environ.setdefault("XDG_CACHE_HOME", str(ROOT / ".cache"))\nimport numpy as np\nimport torch\nfrom torch import nn\nfrom torch.utils.data import DataLoader, TensorDataset\nfrom IPython.display import Image, Markdown, display\ntorch.set_default_dtype(torch.float32)\nprint("Python:", sys.version.split()[0], "PyTorch:", torch.__version__, "NumPy:", np.__version__)\nprint("Workspace:", ROOT)', 'environment')
    md((ROOT / 'DATASETS.md').read_text().replace('# Dataset Selection and Task Scope', '## 1. Dataset selection and task scope', 1) + '\n\nThe collapsed cell below only unpacks public data, provenance records, and license text. It contains no model answers and requires no editing.')
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in sorted((ROOT / 'data').rglob('*')):
            if path.is_file() and 'raw_cache' not in path.parts:
                archive.write(path, str(path.relative_to(ROOT)))
    payload = stream.getvalue()
    encoded = base64.b64encode(payload).decode()
    digest = hashlib.sha256(payload).hexdigest()
    code(f'# @title Restore the bundled public motion subset and provenance\nimport base64\nimport hashlib\nimport io\nimport json\nimport zipfile\n\nDATA_B64 = "{encoded}"\narchive_bytes = base64.b64decode(DATA_B64)\nassert hashlib.sha256(archive_bytes).hexdigest() == "{digest}"\nwith zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:\n    for name in archive.namelist():\n        destination = (ROOT / name).resolve()\n        if not destination.is_relative_to(ROOT.resolve()):\n            raise ValueError("Unsafe archive path")\n        content = archive.read(name)\n        if destination.exists() and destination.read_bytes() != content:\n            raise FileExistsError(f"Existing file differs: {{destination}}; use a new workspace")\n        destination.parent.mkdir(parents=True, exist_ok=True)\n        destination.write_bytes(content)\nDATA = ROOT / "data" / "phuma_g1_walking.npz"\nmanifest = json.loads((DATA.parent / "manifest.json").read_text())\nassert hashlib.sha256(DATA.read_bytes()).hexdigest() == manifest["sha256_npz"]\nprint("Data:", manifest["dataset"], manifest["robot"])\nprint("Sequences / source groups / frames:", manifest["num_sequences"],\n      manifest["num_source_groups"], manifest["frames_10hz"])\nprint("Revision:", manifest["revision"])\ndel DATA_B64, archive_bytes', 'bundled-data', collapsed=True)
    assignment = (ROOT / 'ASSIGNMENT.md').read_text()
    intro, tasks = assignment.split('## A.', 1)
    md(intro.replace('# Predicting Humanoid Trajectories with RNNs (100 Points)', '## 2. Model design and grading', 1))
    md('## 3. Shared experiment code (provided)\n\nPipeline: split by source motion -> create windows within each segment -> normalize using training statistics -> fit a historical constant-velocity baseline -> predict RNN residuals -> select checkpoints by validation ADE -> evaluate on the test set.\n\nThese functions provide the experiment infrastructure. Complete the model and training functions below.\nThe shared experiment runner calls your implementation of `train_one_epoch`; it does not bypass it.\n')
    utils = (ROOT / 'trajectory_utils.py').read_text()
    utils = utils.replace('ROOT = Path(__file__).resolve().parent', 'ROOT = Path.cwd() / "humanoid_rnn_workspace"')
    code(utils, 'provided-infrastructure', collapsed=True)
    code('tracks, provenance = read_tracks(DATA)\nsplits = split_tracks(tracks, seed=42, minimum_length=30)\nfor name, ids in splits.items():\n    observed, future, _ = make_windows(tracks, ids, 20, 10, 2)\n    print(name, "source recordings:", len(ids), "windows:", len(observed),\n          "history:", tuple(observed.shape), "future:", tuple(future.shape))\nassert not (set(splits["train"]) & set(splits["test"]))\nassert not (set(splits["val"]) & set(splits["test"]))\nassert not (set(splits["train"]) & set(splits["val"]))\n\n# Only a training trajectory is used for this preliminary data visualization.\nimport matplotlib.pyplot as plt\nobserved, future, _ = make_windows(tracks, splits["train"], 20, 10, 2)\nfig, ax = plt.subplots(figsize=(6, 4))\nax.plot(observed[0, :, 0], observed[0, :, 1], "o-", label="Observed: -1.9 to 0 s")\nconnected = torch.cat((observed[0, -1:], future[0]))\nax.plot(connected[:, 0], connected[:, 1], "o-", label="Future: +0.1 to +1.0 s")\nax.set(xlabel="Root x (m)", ylabel="Root y (m)", title="PHUMA G1: one training window")\nax.axis("equal")\nax.legend()\nax.grid(alpha=0.2)\nfig.tight_layout()\nexample_path = ROOT / "training_example.png"\nfig.savefig(example_path, dpi=140)\nplt.close(fig)\ndisplay(Image(filename=str(example_path)))', 'inspect-data')
    sections = re.split('(?=^## [A-H]\\.)', '## A.' + tasks, flags=re.M)
    descriptions = {s[3]: s.strip() for s in sections if s.startswith('## ')}
    sources = function_sources(f"{(kind if student else 'solution')}.py")
    ordered = [('A', 'make_features'), ('B', 'ElmanRNN'), ('C', 'scan'), ('D', 'predict'), ('E', 'train_one_epoch'), ('F', 'displacement_errors'), ('G', 'MyRNN')]
    for letter, name in ordered:
        md(descriptions[letter])
        code(sources[name], f'task-{name.lower()}')
    md('## 4. Implementation checks\n\nAfter completing A-F, set `MODELS="elman"`. After completing G, set `MODELS="both"`.\nThe public checks cover physical units, tensor shapes, gradients across time, batch independence, whether training reduces loss, and split isolation.\nThe small artificial tensors used here only check function correctness. Subsequent training and evaluation use PHUMA data.\n')
    code('MODELS = "both"  # "elman", "gated", or "both"\nEPOCHS = 100\nDEVICE = "cpu"  # Set "cuda" only with a GPU runtime.\n', 'experiment-settings')
    check_source = function_sources('check_assignment.py')['check']
    code(check_source + '\n\nfrom types import SimpleNamespace\nimplementation = SimpleNamespace(make_features=make_features, ElmanRNN=ElmanRNN,\n    MyRNN=MyRNN, scan=scan, predict=predict, train_one_epoch=train_one_epoch,\n    displacement_errors=displacement_errors)\nassert MODELS in ("both", "elman", "gated")\nfor model_name in (["elman", "gated"] if MODELS == "both" else [MODELS]):\n    check(implementation, model_name)', 'public-checks', collapsed=True)
    md('## 5. Training from scratch and final evaluation\n\nThe default is 100 epochs. Both models start with zero-initialized output heads, use the same baseline and inputs, and select their checkpoints independently using validation data.\nTest metrics are calculated only after training and model selection. Do not repeatedly select hyperparameters based on test results.\nEach execution of this cell creates a new output directory without overwriting earlier experiments.\n')
    code('from datetime import datetime\nRUN = ROOT / "outputs" / datetime.now().strftime("run_%Y%m%d_%H%M%S_%f")\nrun_experiment({"elman": ElmanRNN, "gated": MyRNN}, make_features,\n    predict, train_one_epoch, displacement_errors,\n    argv=["--data", str(DATA), "--output", str(RUN), "--epochs", str(EPOCHS),\n          "--model", MODELS, "--device", DEVICE, "--seed", "42"])\nreport = json.loads((RUN / "metrics.json").read_text())', 'train-and-evaluate')
    code('table = "| Model | Test ADE (m) | Test FDE (m) | Macro-source ADE (m) |\\n"\ntable += "| --- | ---: | ---: | ---: |\\n"\nfor name, metric in report["test"].items():\n    macro = report["test_macro_source"][name]["ADE_m"]\n    table += f"| {name} | {metric[\'ADE_m\']:.4f} | {metric[\'FDE_m\']:.4f} | {macro:.4f} |\\n"\ndisplay(Markdown(table))\nfor name in report["models"]:\n    gain = 100 * (1 - report["test"][name]["ADE_m"] / report["test"]["constant_velocity"]["ADE_m"])\n    print(f"{name}: ADE reduction versus CV = {gain:.1f}%")\nfor filename in ("learning_curves.png", "trajectories.png", "error_by_horizon.png"):\n    display(Image(filename=str(RUN / filename)))', 'results')
    md('## 6. Forecasting from one history and saving the result\n\nOnly the **20 historical points** of one test window are passed to the model. Adding back the last observed position gives an absolute forecast of shape `(10,2)`.\nSelect the model type by validation ADE, not test error. `forecast_history` can also accept your own historical positions in meters, sampled at 10 Hz in the same coordinate system.\nThe checkpoint is produced by training in this notebook and contains the training-set normalization statistics.\n')
    code('@torch.no_grad()\ndef forecast_history(history, checkpoint_path):\n    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)\n    config = saved["config"]\n    history = torch.as_tensor(history, dtype=torch.float32)\n    if history.shape != (config["obs_len"], 2) or not torch.isfinite(history).all():\n        raise ValueError(f"Expected finite ({config[\'obs_len\']}, 2) history")\n    model = {"elman": ElmanRNN, "gated": MyRNN}[saved["model"]](\n        4, config["hidden"], 2 * config["pred_len"])\n    model.load_state_dict(saved["state_dict"])\n    model.eval()\n    history = history[None]\n    features = (make_features(history, saved["dt"]) - saved["mean"]) / saved["std"]\n    baseline = constant_velocity(history, config["pred_len"], saved["dt"])\n    relative = predict(model, features, baseline, saved["output_scale"])\n    return (relative + history[:, -1:])[0].numpy()\n\nselected_model = min(report["models"], key=lambda n: report["models"][n]["val_ADE_m"])\nwith np.load(RUN / "test_predictions.npz", allow_pickle=False) as saved_predictions:\n    history = saved_predictions["observed"][0].copy()\n    expected = saved_predictions[selected_model + "_relative"][0] + history[-1]\nforecast = forecast_history(history, RUN / f"{selected_model}_best.pt")\nnp.testing.assert_allclose(forecast, expected, atol=1e-5, rtol=1e-5)\nnp.save(RUN / "example_history.npy", history)\nnp.save(RUN / "example_forecast.npy", forecast)\nprint("Selected by validation:", selected_model, "forecast shape:", forecast.shape)\nprint("Future absolute XY positions (meters):\\n", forecast)\nprint("Outputs:", RUN)', 'inference')
    md(descriptions['H'])
    md('### Your analysis\n\n1. Why should source motions be split before windowing, instead of randomly splitting overlapping windows?\n2. What information leakage is prevented by training-only normalization and by constructing velocities from history only?\n3. Compare ADE/FDE for both RNNs and the constant-velocity baseline, and explain the differences.\n4. Select a test case with a relatively large error and discuss the effects of stopping, turning, or motion intent.\n5. Explain the data source and identify practical tasks to which these results cannot be directly transferred.\n\nWrite your analysis here.')
    if not student:
        md('### Instructor discussion notes\n\nReference run with seed=42: CV ADE/FDE=0.1842/0.3754 m; Elman=0.1146/0.2392 m; gated=0.1192/0.2444 m.\nUse the actual outputs from your run. A more complex gated model need not perform better on a small dataset.\nDiscuss parameter counts, overfitting, and validation curves, but do not infer general architectural superiority from one run.\nHistorical velocity provides short-term trends, while the RNN learns deviations from constant velocity. Turning and stopping remain uncertain without motion intent.\nRandomly splitting overlapping windows places highly similar segments in both training and test sets; grouping by source motion before splitting reduces this problem.\nThe reference motions have undergone upstream processing and do not include online sensor noise or a closed-loop controller. Their forecast errors do not directly measure real-robot navigation performance.\n')
    nbf.validate(notebook)
    path = ROOT / f'humanoid_rnn_{kind}.ipynb'
    nbf.write(notebook, path)
    print(path.name, 'cells:', len(cells), 'bytes:', path.stat().st_size)
if __name__ == '__main__':
    build(student=False)
    build(student=True)
