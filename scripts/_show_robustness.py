"""Pretty-print robustness.json results."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
d = json.loads((ROOT / "results" / "robustness.json").read_text())
epss = d["epsilons"]

for attack in ("fgsm", "pgd"):
    print(f"\n=== {attack.upper()} accuracy by epsilon (%) ===")
    head = "model".ljust(18) + "  " + "  ".join(f"{e:7.3f}" for e in epss)
    print(head)
    print("-" * len(head))
    for m in d["models"]:
        accs = [a * 100 for a in m[attack]["accuracies"]]
        print(f"{m['model']:<18s}  " + "  ".join(f"{a:7.2f}" for a in accs))
