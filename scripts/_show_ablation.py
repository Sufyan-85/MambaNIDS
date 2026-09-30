"""Pretty-print ablation_results.json using ASCII-only output (Windows-safe)."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
data = json.loads((ROOT / "results" / "ablation_results.json").read_text(encoding="utf-8"))

print(f"{'variant':<18s} {'accuracy':>10s} {'f1_macro':>10s} {'auc_roc':>10s} {'d_acc':>9s} {'params':>13s}")
print("-" * 76)
for r in data:
    m = r["metrics"]
    print(f"{r['variant']:<18s} "
          f"{m['accuracy']*100:9.2f}% "
          f"{m['f1_macro']*100:9.2f}% "
          f"{m['auc_roc_ovr']*100:9.2f}% "
          f"{r['delta_acc_vs_full']*100:+8.2f}% "
          f"{r['n_params']:>12,}")
