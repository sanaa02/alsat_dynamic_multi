from pathlib import Path

EXCLUDE_DIRS = {
    ".git",
    "__pycache__",
    ".venv",
    "venv",
    "env",
    "node_modules",
    ".mypy_cache",
    ".pytest_cache",
}

root_dir = Path(".")
output_file = "all_python_files.txt"

with open(output_file, "w", encoding="utf-8") as out:
    for py_file in sorted(root_dir.rglob("*.py")):
        if any(part in EXCLUDE_DIRS for part in py_file.parts):
            continue

        out.write(f"\n\n{'=' * 80}\n")
        out.write(f"FILE: {py_file}\n")
        out.write(f"{'=' * 80}\n\n")

        try:
            out.write(py_file.read_text(encoding="utf-8", errors="ignore"))
        except Exception as e:
            out.write(f"\n[ERROR READING FILE: {e}]\n")

print(f"Saved merged files to {output_file}")