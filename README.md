# CLiQ-Nav experiments

Experimental results and reproduction scripts for **CLiQ-Nav: Image-relative spatial grounding for zero-shot aerial vision-language navigation**.

This repository contains the paper's complete formal scoring results, the trajectories used to compute them, route and observation configurations, heading execution checks, experiment scripts, and result figures. The release was prepared on 2 November 2026 from the current manuscript's experiment records. It is an independent repository with its own Git history.

<!-- CURRENT_PAPER_RESULTS_BEGIN -->
## Current paper results (revised success radii)

**The tables below use the current KBS manuscript scoring version, `revised_v2`.**

- **City Park:** 10 m, except route 03 (15 m) and route 05 (20 m).
- **Downtown West:** **3 m on routes 09, 14, 16, 17, 21, 25 and 29; 6 m on all other routes.**

The same route-specific rule applies to every method and heading condition. SR, SPL and OR are percentages; NE and TL are metres. Values are means, with percentages rounded to one decimal and distances to two. The radius change occurred during manuscript revision, after the original outcomes were available.

### City Park: benchmark and cue conditions

Each row contains 140 episodes (ten per route).

| Method / condition | SR (%) | NE (m) | SPL (%) | OR (%) | TL (m) |
|---|---:|---:|---:|---:|---:|
| Qwen-VL-Max | 8.6 | 62.01 | 8.1 | 86.4 | 97.99 |
| Qwen-VL-Plus | 7.1 | 84.26 | 7.0 | 80.0 | 122.84 |
| Qwen3-VL-Flash | 12.9 | 69.90 | 11.7 | 85.0 | 106.15 |
| Qwen3.5-35B-A3B + CLIP | 80.7 | 9.81 | 68.7 | 87.1 | 41.75 |
| Qwen3.5-35B-A3B | 42.9 | 15.94 | 37.3 | 60.0 | 44.01 |
| Qwen3.5-Flash + CLIP | 54.3 | 13.53 | 45.8 | 77.1 | 45.74 |
| Qwen3.5-Flash | 38.6 | 18.15 | 33.0 | 59.3 | 44.13 |
| CityNavAgent | 32.1 | 20.47 | 31.4 | 37.1 | 35.80 |
| AerialVLA | 17.9 | 17.45 | 16.0 | 73.6 | 39.96 |
| FineCog-Nav | 2.9 | 35.58 | 2.9 | 2.9 | 7.33 |
| Qwen3.5-35B-A3B + random cue | 21.4 | 25.87 | 18.1 | 43.6 | 39.39 |

### Downtown West: benchmark and cue conditions

Each row contains 300 episodes (ten per route).

| Method / condition | SR (%) | NE (m) | SPL (%) | OR (%) | TL (m) |
|---|---:|---:|---:|---:|---:|
| Qwen3.5-35B-A3B + CLIP | 54.7 | 7.27 | 48.9 | 63.3 | 13.15 |
| Qwen3.5-35B-A3B | 35.3 | 8.36 | 31.9 | 46.7 | 11.56 |
| AerialVLA | 7.0 | 56.11 | 5.8 | 57.7 | 67.70 |
| FineCog-Nav | 8.0 | 12.45 | 7.6 | 8.0 | 1.62 |
| CityNavAgent | 22.7 | 10.99 | 17.8 | 58.0 | 21.44 |
| Qwen3.5-35B-A3B + random cue | 30.3 | 9.36 | 26.6 | 45.7 | 14.67 |

### Heading ablation

Both conditions retain Qwen3.5-35B-A3B and the actual CLIP cue. The unablated rows reuse the full-method benchmark episodes and are not additional experiments.

| Environment | Heading condition | SR (%) | NE (m) | SPL (%) | OR (%) | TL (m) |
|---|---|---:|---:|---:|---:|---:|
| City Park | Unablated (weighted-history heading) | 80.7 | 9.81 | 68.7 | 87.1 | 41.75 |
| City Park | Ablated (last-segment heading) | 2.9 | 36.95 | 2.2 | 10.7 | 39.41 |
| Downtown West | Unablated (weighted-history heading) | 54.7 | 7.27 | 48.9 | 63.3 | 13.15 |
| Downtown West | Ablated (last-segment heading) | 11.3 | 12.45 | 9.9 | 20.7 | 13.88 |

Full-precision means and marginal 95% intervals: [current summary](supplementary/current_paper_results/summary.csv), [heading conditions](supplementary/current_paper_results/heading.csv), and [heading differences](supplementary/current_paper_results/heading_differences.csv). The summary covers all 19 distinct formal condition/map groups (3,780 episodes).

The cue rows retain the unified three-condition bootstrap, and heading comparisons retain their own manuscript bootstrap stream. The two primary actual-minus-random SR contrasts use 97.5% intervals, available in `supplementary/cue_prior_control/results.json`; these differ from the marginal 95% intervals in the summary CSV.

**Original uniform-6-m outputs are historical sensitivity records. They are not the displayed current results.** Historical route-layout summaries and frozen batch settings remain unchanged for provenance; use the current result files linked above for manuscript values.

Regenerate these CSV files and this README section with:

```bash
python -B scripts/export_current_paper_results.py --update-readme
```
<!-- CURRENT_PAPER_RESULTS_END -->

## Experiment coverage

There are **3,780 distinct formal navigation episodes**, across 44 routes with ten repetitions per route and condition.

| Environment | Experiment | Conditions | Episodes |
|---|---|---|---:|
| City Park | Backbone and cue benchmark | Seven Qwen configurations | 980 |
| City Park | External comparisons | CityNavAgent, AerialVLA, FineCog-Nav | 420 |
| Downtown West | Cue benchmark | Qwen3.5-35B-A3B with and without CLIP | 600 |
| Downtown West | External comparisons | CityNavAgent, AerialVLA, FineCog-Nav | 900 |
| Both environments | Heading ablation | Last-segment heading: 140 + 300 | 440 |
| Both environments | Random-cue control | Uniform nine-region cue: 140 + 300 | 440 |
| | **Total distinct formal episodes** | | **3,780** |

The seven City Park Qwen conditions are Qwen-VL-Max, Qwen-VL-Plus, Qwen3-VL-Flash, Qwen3.5-35B-A3B, Qwen3.5-35B-A3B + CLIP, Qwen3.5-Flash, and Qwen3.5-Flash + CLIP. The external comparisons are the **adapted implementations used in this paper**; they should not be described as exact reproductions of the original methods.

The portable three-condition cue subset contains 1,320 records. Of these, 880 duplicate the actual-cue/no-cue benchmark records and 440 are the new random-cue episodes. These copies are included for independent reproduction and do not increase the total. Functional checks, unfinished attempts, and discarded historical batches are excluded from the formal totals.

## Contents

| Path | Contents |
|---|---|
| `supplementary/episodes_manifest.csv` | All 3,340 benchmark/heading scoring records, with coordinates, endpoints, radii and trajectory SHA-256 |
| `supplementary/data/` | All 3,340 unchanged XYZ scoring trajectories |
| `supplementary/current_paper_results/` | Displayed current revised-radius summary, heading conditions/differences and scoring-version metadata |
| `supplementary/results/` | Current episode, route and aggregate results; bootstrap intervals and plot data |
| `supplementary/cue_prior_control/` | All 1,320 cue-subset trajectories, metrics, contrasts, frozen cue schedules and transmitted cues |
| `supplementary/routes.csv`, `interfaces/` | All 44 route configurations, exact CLIP/MLLM target texts, Chinese prompts, nine-region mapping and camera settings |
| `supplementary/heading_*_audit.csv` | All 44 route definitions and 440 formal heading execution checks |
| `supplementary/*original_6m*`, `threshold_revision.json` | Original Downtown scoring and the explicitly documented revision-stage radius rule |
| `supplementary/publication_provenance.json` | Source paths and hashes, published hashes and the list of publication-copy code adaptations |
| `map1/`, `map2/` | Original route-layout scoring data, history inputs, final heading records, frozen settings and experiment/analysis scripts |
| `cue_prior_control/` | Random-cue design, execution, monitoring and analysis code |
| `nav_qwen*.py`, `formal*.py`, `benchmark_map2.py`, `fly_on_keyboard.py` | Navigation, controller and batch-execution scripts |
| `city_nav/adapted/`, `aerialvla/adapted/`, `finecog_nav/adapted/` | Baseline implementations used by the experiment runners |
| `scripts/` | Complete offline verification and original MATLAB result-rendering programs |
| `figures/`, `source_data/` | Result plots, native MATLAB figures, numerical figure inputs and the qualitative navigation sequences |

Raw scoring trajectories are unchanged. Publication copies of code remove an embedded credential fallback and a machine-specific CLIP import path, detach manuscript-only file checks, and expose simulator installation paths through environment variables. Historical paths inside experimental provenance remain historical records. Older route-layout summaries and heading-batch analyses may use the original radius convention; `supplementary/results/` and `supplementary/cue_prior_control/results.json` are the authoritative current paper results.

Manuscript sources/PDFs, references, reviews, submission documents, model weights, simulator assets, runtime environments, temporary files, complete per-step sensor dumps and failed-attempt archives are outside this release.

## Reproduce all reported numerical results

Offline reproduction requires Python 3.10+ and NumPy. It makes no model calls and needs no simulator, GPU, credentials or network. The release was verified with Python 3.13.9 and NumPy 2.2.6.

```bash
git -c core.autocrlf=false clone https://github.com/noob32123/cliq-nav-experiments.git
cd cliq-nav-experiments
python -m pip install numpy==2.2.6
python -B scripts/reproduce_all.py --output reproduced_results
```

The wrapper checks every published file hash, complete condition/route/repetition cells, all trajectory hashes, duplicated cue-subset identities, and all 440 heading audits. By default it reproduces only the current revised-radius benchmark/heading results and unified cue analysis, compares all means and interval bounds, and exports the current paper tables to `paper_results/` under the selected output directory. The historical uniform-6-m replay requires the explicit `--include-original-6m` option.

Individual entry points, from the repository root:

```bash
python -B supplementary/reproduce_results.py --output reproduced_results/current
python -B supplementary/reproduce_cue_control.py --output reproduced_results/cue_control
```

Optional historical threshold sensitivity (not the displayed current results):

```bash
python -B scripts/reproduce_all.py --output reproduced_with_sensitivity --include-original-6m
```

The original MATLAB result scripts are in `scripts/`. They require MATLAB with the Cambria Math font. For the currently adopted backbone and heading interval plots, run `addpath('scripts'); render_adopted_result_figures`. Other original result plots use `render_paper_results`. These original rendering programs write their exports into `figures/`; numerical offline verification does not require MATLAB. Qualitative navigation frames and action logs are in `source_data/navigation_maneuvers/`.

## Metrics, intervals and interpretation

SR, NE, SPL, OR and TL are calculated from the final navigation endpoint and horizontal XY trajectory. NE/TL are in metres; SR/SPL/OR are proportions in scoring outputs and percentages in plot tables. SPL uses the configured start-to-goal horizontal straight-line distance as its reference. A saved navigation endpoint cuts off later landing motion. Collision/termination labels never alter the score, inclusion or weighting. For cue experiments, path accumulation starts at the first recorded sample after takeoff and heading initialization.

Bootstrap resampling uses whole routes, retaining ten repetitions within each selected route. Repetition indices are not matched pairs across separately collected conditions. The original benchmark uses 10,000 resamples with seeds 20260823 (City Park) and 20261014 (Downtown); heading comparisons use 20260823. The unified cue analysis uses SeedSequence streams based on 20261027. The two primary actual-minus-random-cue SR contrasts use 97.5% intervals; the other contrasts use exploratory 95% intervals. The unified cue replay supplies the current three-condition means/intervals; remaining benchmark and heading comparisons retain their archived seeds.

During manuscript revision, after original outcomes were available, the Downtown success radius was reduced from 6 m to 3 m on routes 09, 14, 16, 17, 21, 25 and 29 because their configured starts already lay inside the original success region. Other Downtown routes retain 6 m. The same rule applies to every method and heading condition. Both evaluations are included. Coordinates, trajectories and episode inclusion are unchanged; NE and TL are identical between versions.

City Park history tracks were derived from historical navigation motion. Downtown six-point histories are explicitly synthetic; its heading experiment compares an initial-heading perturbation with a historical full configuration. Historical full configurations lack equivalent yaw telemetry. These cross-batch comparisons do not establish an isolated causal effect under independently measured GNSS histories.

## Run new simulator experiments

Online execution uses **Windows, AirSim, Unreal Engine 4.27**, the corresponding City Park/Downtown projects, and compatible model services. Map assets and model weights must be obtained separately from their providers. The frozen scene records retain map hashes and landmark checks. Configure local paths before starting a new batch:

```powershell
$env:UE4_EDITOR = 'D:/UE_4.27/Engine/Binaries/Win64/UE4Editor.exe'
$env:CITYPARK_PROJECT = 'D:/CityPark/CityPark.uproject'
$env:DOWNTOWN_PROJECT = 'D:/Downtown/airsimdowntown.uproject'
$env:CITYPARK_MAP = 'D:/CityPark/Content/CityPark/Maps/Showcase.umap'
$env:DOWNTOWN_MAP = 'D:/Downtown/Content/maps/airsim_downtown.umap'
$env:OPENAI_API_KEY = '<your API key>'
# Legacy DashScope-backed entry points additionally use DASHSCOPE_API_KEY.
```

Online dependencies include NumPy, Pillow, OpenCV, OpenAI's Python client, PyTorch/torchvision, keyboard and the AirSim Python client. Install OpenAI CLIP from its official source (`python -m pip install git+https://github.com/openai/CLIP.git`), rather than an unrelated package named `clip`. Choose a CUDA-compatible PyTorch build for the target machine. AirSim may be installed from the matching Microsoft AirSim PythonClient checkout. Local AerialVLA additionally needs OpenVLA-7B, the XuPeng23/AerialVLA LoRA, transformers 4.46.3, peft 0.13.2, tokenizers 0.20.3 and a compatible bitsandbytes installation. `aerialvla/download_models.py` fetches the separately distributed weights. Their providers' terms still apply.

Use new batch directories. Published checkpoints are partial evidence archives, not resumable live batches; the full original retry/sensor archives are not included.

```powershell
# Heading experiment preflight (no formal navigation)
python -B map1/scripts/run_heading_ablation.py --preflight-only --batch-dir map1/heading_ablation/new_batch
python -B map2/scripts/run_heading_ablation.py --preflight-only --batch-dir map2/heading_ablation/new_batch
# For each initialized batch, run functional checks, then formal experiments:
python -B map1/scripts/run_heading_ablation.py --resume map1/heading_ablation/new_batch --smoke-test
python -B map1/scripts/run_heading_ablation.py --resume map1/heading_ablation/new_batch --formal-only
# Use the equivalent map2 commands for Downtown.

# Random-cue control: create frozen design, then functional checks and formal jobs
python -B -m cue_prior_control.runner --map all --batch-dir experiments/new_cue_batch --preflight
python -B -m cue_prior_control.runner --map all --batch-dir experiments/new_cue_batch --resume --smoke
python -B -m cue_prior_control.runner --map all --batch-dir experiments/new_cue_batch --resume --run
```

Heading/cue runners use the archived Qwen3.5-35B-A3B model configuration, nine-region CLIP ViT-B/32 cue and original action policy. Their historical endpoint is preserved as a default; the runners expose `--endpoint`. Provider availability and repeatability of new stochastic model calls are separate from deterministic offline scoring. The original City Park backbone scripts are parameterized in source; their `main()` scenario/model settings must be selected before collecting a new configuration. Baseline scenario entry points are `city_nav/run_citynav.py`, `aerialvla/run_aerialvla.py` and `finecog_nav/run_finecog.py`.

## Availability and licensing

Code and results are publicly available for inspection. **No license is granted by this repository at present**, as requested by the authors. No third-party model, simulator or dataset license is overridden. This publication should therefore be described as a public code/results release rather than a permissively licensed software distribution.
