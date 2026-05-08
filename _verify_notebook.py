import json
import sys

path = r"d:\WorkPlace\Pycharm\gemma4-ft\gemma4_multimodal_demo\notebooks\02-data_preparation-labelme_processing.ipynb"

try:
    with open(path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    print(f"Valid JSON: {len(nb['cells'])} cells")
    md = sum(1 for c in nb["cells"] if c["cell_type"] == "markdown")
    cd = sum(1 for c in nb["cells"] if c["cell_type"] == "code")
    print(f"Markdown: {md}, Code: {cd}")

    for i, c in enumerate(nb["cells"]):
        src = "".join(c["source"])[:80].replace("\n", " | ")
        print(f"  [{i}] {c['cell_type']}: {src}...")

    code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
    imports_cell = code_cells[0]
    imports_src = "".join(imports_cell["source"])

    has_os = "import os" in imports_src
    has_random = "import random" in imports_src
    has_trange = "trange" in imports_src
    has_tqdm_available = "TQDM_AVAILABLE" in imports_src
    has_notebook_ui = "NotebookUI" in imports_src
    has_progress_display = "ProgressDisplay" in imports_src
    has_tools_import = "from tools" in imports_src

    print(f"\nImports analysis:")
    print(f"  import os: {has_os} (should be False)")
    print(f"  import random: {has_random} (should be False)")
    print(f"  trange: {has_trange} (should be False)")
    print(f"  TQDM_AVAILABLE: {has_tqdm_available} (should be True)")
    print(f"  NotebookUI: {has_notebook_ui} (should be True)")
    print(f"  ProgressDisplay: {has_progress_display} (should be False)")
    print(f"  from tools: {has_tools_import} (should be True)")

    all_code = "".join("".join(c["source"]) for c in code_cells)
    has_max_workers_call = "max_workers" in all_code
    has_error_handling = "try:" in all_code and "except" in all_code
    has_elapsed_time = "elapsed" in all_code or "duration" in all_code

    print(f"\nFeature analysis:")
    print(f"  max_workers passed: {has_max_workers_call}")
    print(f"  Error handling (try/except): {has_error_handling}")
    print(f"  Elapsed time display: {has_elapsed_time}")

    print("\nAll checks passed!")

except json.JSONDecodeError as e:
    print(f"JSON parse error: {e}")
    sys.exit(1)
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)