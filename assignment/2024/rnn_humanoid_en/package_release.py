"""Create separate instructor/student releases using explicit file allowlists."""

from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parent
COMMON = ["student.py", "trajectory_utils.py", "check_assignment.py", "predict.py",
          "prepare_phuma.py", "requirements.txt", "requirements-tested.txt",
          "requirements-notebook.txt", "ASSIGNMENT.md", "DATASETS.md",
          "README_student.md", "humanoid_rnn_student.ipynb"]
INSTRUCTOR = ["solution.py", "README.md", "humanoid_rnn_instructor.ipynb",
              "build_notebooks.py", "verify_notebooks.py", "package_release.py",
              "notebook_verification.json", "translation_verification.json"]


def main():
    data_files = [str(p.relative_to(ROOT)) for p in sorted((ROOT / "data").rglob("*"))
                  if p.is_file() and "raw_cache" not in p.parts]
    reference = [str(p.relative_to(ROOT)) for p in sorted((ROOT / "outputs/verified_cpu").iterdir())
                 if p.is_file()]
    for kind in ("student", "instructor"):
        files = COMMON + data_files
        if kind == "instructor":
            files += INSTRUCTOR + reference
        destination = ROOT.parent / f"rnn_humanoid_{kind}_en.zip"
        with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as archive:
            for name in files:
                archive.write(ROOT / name, f"rnn_humanoid_{kind}_en/{name}")
        with zipfile.ZipFile(destination) as archive:
            assert archive.testzip() is None
            if kind == "student":
                assert not any("solution" in n or "instructor" in n or n.endswith(".pt")
                               or "/outputs/" in n for n in archive.namelist())
        print(destination.name, destination.stat().st_size, "bytes", len(files), "files")


if __name__ == "__main__":
    main()
