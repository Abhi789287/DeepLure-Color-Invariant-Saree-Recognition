from pathlib import Path
required = [
"README.md","APPROACH_NOTE.txt","DATASET.md","REQUIREMENT_TRACEABILITY.md",
"FINAL_RESULTS_TEMPLATE.md","SUBMISSION_CHECKLIST.md","requirements.txt",
"configs/config.yaml","notebooks/DeepLure_Saree_Recognition.ipynb",
"src/dataset.py","src/model.py","src/losses.py","src/train.py",
"src/inference.py","src/evaluate.py","src/efficiency.py"
]
missing=[x for x in required if not Path(x).exists()]
n=len(Path("APPROACH_NOTE.txt").read_text().strip())
print("Missing:", missing)
print("Approach-note characters:", n)
assert not missing
assert n <= 500
print("Project structure validation: PASS")
