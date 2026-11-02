# Three-condition cue ablation

This self-contained subset contains 1,320 scored episodes: actual CLIP cue,
uniform random cue, and no cue, with 140 episodes per condition in City Park
and 300 in Downtown West. It reuses 880 benchmark episodes and adds 440 new
random-cue episodes. There are ten repeats per condition per route.

From the submission directory run:

```
python supplementary/reproduce_cue_control.py
```

Python 3.10+ and NumPy are required. The replay checks trajectory SHA-256,
complete route/repeat cells, and invariance to event labels. It scores all
episodes from their XY coordinates and saved endpoints, reconstructs every
route-matched contrast and interval, and verifies equality with the published
full-precision results. It writes to `replay/`, preserving published outputs.

- `episodes_manifest.csv` and `data/`: all scoring inputs and file hashes.
- `results.json`: full-precision means, marginal intervals and three pairwise contrasts.
- `episode_metrics.csv`: 1,320 scored episodes.
- `routewise_metrics.csv`: 132 condition/route means.
- `routewise_differences.csv`: 132 contrast/route differences.
- `plot_ci.csv`: five-metric means and marginal 95% intervals, in plotted units.
- `cue_schedule.json`: frozen formal and functional-check sequences.
- `transmitted_cues.csv`: all 2,941 formal decisions, sent regions and shadow CLIP regions.
- `city/`, `downtown/`: unchanged execution configurations, routes, schedules and camera settings.
- `provenance.json`: original source paths and scoring-trajectory hashes.
- `integrity.json`, `final_verification.json`: the completed experiment's dated integrity checks, before manuscript integration.

The two primary SR contrasts, actual minus random cue, use 97.5% intervals
with a two-map Bonferroni allocation. All other contrasts use exploratory 95%
intervals. Routes are the resampling units; ten repeats remain nested within
each route and condition. Repeat numbers are not episode-level matched pairs.

All three conditions used Qwen3.5-35B-A3B with identical model configuration
and inference parameters. Within each map, route settings, observation
modalities and preprocessing, weighted heading initialization, action
interfaces and execution policies, timing, and evaluation rules were shared;
only the cue condition varied. Only random-cue episodes were newly collected,
on 27–28 October 2026. Random cues are iid uniform nine-region draws with the
original cue sentence. Source records and scoring inputs are retained.
See Supplementary Methods S5 and S8 for scoring, inference and execution
details, and `protocol_confirmation_20261028.txt` for the dated author
clarification. Frozen configurations and experiment outputs are unchanged.

The parent bundle contains 3,780 distinct navigation episodes. The 880 reused
records in this subset duplicate existing scoring records for portability;
they are not additional experiments. Six short functional checks are excluded
from the formal totals. No result was selected or removed based on performance.
