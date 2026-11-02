# Downtown West scoring versions

The current paper uses **3 m on routes 09, 14, 16, 17, 21, 25 and 29, and 6 m on the other 23 routes**. This applies to all methods and heading conditions.

Current displayed results are in the [repository README](../README.md#current-paper-results-revised-success-radii) and [current paper summary](../supplementary/current_paper_results/summary.csv). The evaluated coordinates and radii are in `../supplementary/routes.csv`.

The original route-layout files, frozen batch configurations and `heading_ablation/20261016_heading_prediction/analysis/` preserve historical uniform-6-m records. They are retained as experimental provenance, not as the current paper's results. In particular, the historical full/ablated SR values of 70.3%/21.7% correspond to uniform 6 m; the current revised-radius values are **54.7%/11.3%**.

From the repository root, reproduce the current results with:

```bash
python -B scripts/reproduce_all.py --output reproduced_results
```

Only when comparing historical thresholds, add `--include-original-6m`. The legacy `map2/scripts/validate_map2.py` and archived batch analyzers retain the original batch scoring conventions; use the canonical supplementary replay for current manuscript reporting.
