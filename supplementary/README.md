# Offline reproducibility bundle

**Current displayed results use `revised_v2`: Downtown routes 09, 14, 16, 17, 21, 25 and 29 use 3 m; all other Downtown routes use 6 m. City Park radii remain 10 m except route 03 (15 m) and route 05 (20 m).** See `current_paper_results/summary.csv` and the root README for the current tables. Files named `original_6m` and archived batch analyses are historical sensitivity/provenance records.

This archive reproduces 3,780 distinct navigation episodes: 3,340 original benchmark/heading episodes in 17 groups and 440 new uniform random-cue episodes in two groups, without simulation or model calls. All required scoring trajectories are present. The accompanying Supplementary Methods describes the exact endpoint conventions and available camera/prompt records.

Run `python reproduce_results.py` with Python 3.10+ and NumPy. It verifies heading/metric edge cases and writes `results/`. The original replay retains its fixed seeds. Main-text Qwen3.5-35B-A3B cue intervals and three-condition whiskers use the unified replay in `reproduce_cue_control.py`; the Qwen3.5-Flash and heading comparisons retain their original intervals.

- `episodes_manifest.csv`: one row per episode, coordinate inputs, trajectory path and SHA-256.
- `routes.csv`: 44 route configurations, unchanged pre-experiment coordinates, original/revised radii, and the two exact target-text inputs.
- `data/`: unchanged copies of the original XYZ scoring trajectories.
- `heading_route_audit.csv`, `heading_episode_audit.csv`: 44 history/configuration checks and 440 recorded formal execution checks.
- `interfaces/`: archived Chinese prompt templates, exact nine-region phrases, settings and observation examples. English prompt translation is in Supplementary Methods.
- `results/`: full-precision episode, route, aggregate and paired-difference results.
- `threshold_revision.json`: revision-stage rule halving the radius to 3 m on seven initially successful Downtown West routes, retaining 6 m elsewhere; the same rule applies to all methods and heading conditions.
- `episodes_manifest_original_6m.csv`, `routes_original_6m.csv`, `results_original_6m/`: preserved original evaluation. Replay with `python reproduce_results.py --manifest episodes_manifest_original_6m.csv --output results_original_6m` from this directory.
- `interfaces/route_target_texts.csv`: exact English CLIP strings and separate Chinese navigation descriptions for all 44 routes. Route identifiers are not model text. The complete-text examples identify their actual later heading requests.

The radius revision followed observation of the original outcomes. It uses configured start/goal geometry and a fixed halving rule, not a search for a favorable method difference. Raw trajectories and preset coordinates were not changed; no episode was excluded. NE/TL remain identical episode by episode. Both scoring versions are supplied and Supplementary Methods S7 reports their comparison.

The three-condition Qwen3.5-35B-A3B cue ablation uses identical model configuration and inference parameters. Within each map, route settings, observation modalities and preprocessing, heading initialization, action interfaces and execution policies, timing, and evaluation rules are shared; only the cue condition varies. Matching is by route, with ten independently repeated episodes per condition per route. Repeat numbers are not episode-level matched pairs. Source records, scoring conventions and available heading telemetry are documented in Supplementary Methods.

Event labels do not enter scoring or inclusion. Raw trajectory bytes and evaluated coordinate versions are preserved. The new random-cue condition is complete; this bundle replays archived results only and makes no model calls.

For the cue ablations, trajectory recording begins after takeoff and heading initialization, before the first MLLM decision. Path length starts at the first recorded sample; the straight-line SPL reference uses the configured start. Initialization movement before recording is outside the sampled path. Supplementary Methods S4 also identifies the shared default temperature setting; offline reproducibility refers to scores and route-level statistics.

- `cue_prior_control/`: portable 1,320-episode comparison, including 880 reused records, 440 new records, frozen cue schedules, all sent cue indices and scoring hashes.
- `reproduce_cue_control.py`: route-matched 10,000-resample bootstrap; two primary actual-minus-random SR contrasts use 97.5% intervals, all remaining contrasts use exploratory 95% intervals.

