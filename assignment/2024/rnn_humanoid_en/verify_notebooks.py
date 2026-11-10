"""Execute isolated notebooks and verify exact answer-to-TODO correspondence.

Requires nbformat, nbclient and ipykernel. This is an instructor maintenance tool.
"""

import ast
import json
from pathlib import Path
import re
import sys
import tempfile

from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager
import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

ROOT = Path(__file__).resolve().parent


def execute(notebook, directory, kernel_dir):
    directory.mkdir(parents=True)
    manager = KernelManager(kernel_name="python3",
                            kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(kernel_dir)]))
    client = NotebookClient(notebook, km=manager, timeout=300,
                            resources={"metadata": {"path": str(directory)}})
    try:
        return client.execute()
    finally:
        client._cleanup_kernel()


def main():
    instructor_path = ROOT / "humanoid_rnn_instructor.ipynb"
    student_path = ROOT / "humanoid_rnn_student.ipynb"
    instructor = nbformat.read(instructor_path, as_version=4)
    student = nbformat.read(student_path, as_version=4)
    for notebook in (instructor, student):
        nbformat.validate(notebook)
        for cell in notebook.cells:
            if cell.cell_type == "code":
                compile(cell.source, "notebook-cell", "exec")
    assert all(not cell.get("outputs") for cell in student.cells)
    source = (ROOT / "solution.py").read_text()
    answers = dict(re.findall(r"^ *# BEGIN ANSWER (\w+)\n(.*?)^ *# END ANSWER \1$",
                              source, re.M | re.S))
    assert len(answers) == 13
    filled = nbformat.reads(nbformat.writes(student), as_version=4)
    found = []
    teacher_tasks = {cell.id: cell.source for cell in instructor.cells
                     if cell.cell_type == "code" and cell.id.startswith("task-")}
    for cell in filled.cells:
        if cell.cell_type != "code" or not cell.id.startswith("task-"):
            continue

        def replace(match):
            key = match.group(1)
            found.append(key)
            return answers[key].rstrip()

        cell.source = re.sub(r'^ *# TODO \[(\w+)\]:[^\n]*\n *raise NotImplementedError\("TODO: \1"\)',
                             replace, cell.source, flags=re.M)
        assert ast.dump(ast.parse(cell.source)) == ast.dump(ast.parse(teacher_tasks[cell.id])), cell.id
    assert len(found) == 13 and set(found) == set(answers)
    for cell in student.cells:
        if cell.cell_type == "code":
            assert "BEGIN ANSWER" not in cell.source
            assert "from solution" not in cell.source and "import solution" not in cell.source

    with tempfile.TemporaryDirectory(prefix="humanoid-notebook-check-") as temporary:
        temp = Path(temporary)
        kernels = temp / "kernels"
        (kernels / "python3").mkdir(parents=True)
        (kernels / "python3" / "kernel.json").write_text(json.dumps({
            "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
            "display_name": "Python 3", "language": "python"}))
        print("Executing standalone instructor notebook, including 100 epochs per model...", flush=True)
        executed = execute(instructor, temp / "instructor", kernels)
        nbformat.write(executed, instructor_path)
        print("Executing unfilled student notebook; expecting TODO: features...", flush=True)
        try:
            execute(student, temp / "unfilled", kernels)
        except CellExecutionError as exc:
            assert "NotImplementedError" in str(exc) and "TODO: features" in str(exc)
        else:
            raise AssertionError("Unfilled student unexpectedly ran to completion")
        print("Executing student notebook after filling only its 13 answer blocks...", flush=True)
        execute(filled, temp / "filled", kernels)
        reports = []
        for directory in (temp / "instructor", temp / "filled"):
            metrics = list(directory.glob("humanoid_rnn_workspace/outputs/*/metrics.json"))
            assert len(metrics) == 1
            reports.append(json.loads(metrics[0].read_text()))
        assert reports[0]["test"] == reports[1]["test"]
        assert reports[0]["split_source_ids"] == reports[1]["split_source_ids"]
        for name in ("elman", "gated"):
            assert reports[0]["test"][name]["ADE_m"] < 0.15
            assert reports[0]["test"][name]["FDE_m"] < 0.30
            assert reports[0]["test"][name]["ADE_m"] < reports[0]["test"]["constant_velocity"]["ADE_m"]
        result = {"status": "PASS", "answer_blocks": len(found),
                  "standalone_instructor_execution": True,
                  "unfilled_student_expected_failure": "NotImplementedError: TODO: features",
                  "filled_student_execution": True, "filled_student_matches_instructor": True,
                  "python": sys.version.split()[0], "versions": reports[0]["versions"],
                  "test": reports[0]["test"], "test_macro_source": reports[0]["test_macro_source"],
                  "counts": reports[0]["counts"]}
        (ROOT / "notebook_verification.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
