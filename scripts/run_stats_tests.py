"""Run full statistical test battery: McNemar + DeLong (ablation variants) + ECE."""
import json, sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from configs import CFG
from utils.stats import run_all_tests_from_files

ft_path  = CFG.RESULTS_DIR / "finetune_results.json"
abl_path = CFG.RESULTS_DIR / "ablation_results.json"
out_path = CFG.RESULTS_DIR / "statistical_tests.json"

print(f"[stats] finetune_results  : {ft_path}")
print(f"[stats] ablation_results  : {abl_path}")

res = run_all_tests_from_files(
    str(ft_path), str(abl_path),
    alpha=0.05, n_comparisons=5,
)

print(f"\n[stats] ECE                  = {res['ece']['ece']:.4f}")
print(f"[stats] N samples (pooled)   = {res['n_samples']}")
m = res['full_model_auroc_mean']
s = res['full_model_auroc_std']
print(f"[stats] Full model AUROC     = {m:.4f} +/- {s:.4f} (fold-mean)")

with open(out_path, "w") as f:
    json.dump(res, f, indent=2)
print(f"\n[stats] Saved -> {out_path}")
