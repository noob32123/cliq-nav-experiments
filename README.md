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

## Reproduction protocol

The paper contains the core method, evaluation protocol, three-condition means, and threshold-sensitivity results. This section records the full prompts, preprocessing, camera settings, parser, execution details, and offline analysis. Paths are relative to this repository root.

### R1. Heading convention and execution checks
All horizontal directions use the AirSim North-East-Down frame: $x$ points north, $y$ east, and positive yaw rotates from north toward east. Thus a positive signed yaw change requests a right turn. For six positions ordered oldest to newest, the five horizontal displacements are weighted by $[0.10,0.15,0.20,0.25,0.30]$. The predicted direction is $\psi_{\mathrm{pred}}=\operatorname{atan2}(v_{\mathrm{pred},y},v_{\mathrm{pred},x})$. With current body yaw $\psi_0$, the signed turn is $\operatorname{wrap}(\psi_{\mathrm{pred}}-\psi_0)$, where wrap maps to $[-180^\circ,180^\circ)$. The equations in the main text use radians; control calls use degrees.

The latest displacement defines $\psi_{\mathrm{last}}$, not body yaw. In lateral or backward flight these can differ. The recorded last-segment condition uses $\Delta\psi=\operatorname{wrap}(\psi_{\mathrm{last}}-\psi_{\mathrm{pred}})$ and adds this difference to the stored weighted command $u_{\mathrm W}$. The command branch is retained: for example, $u_{\mathrm W}=160^\circ$ and $\Delta\psi=54.6247^\circ$ give $u_{\mathrm{last}}=214.6247^\circ$, rather than replacing the recorded command by an equivalent shorter rotation.

Full-condition turns were derived from weighted histories before the experiments and stored by route.
The heading ablation used retained or generated six-position histories.
Downtown West histories were constructed to match the stored weighted directions.
Historical XYZ logs omit per-run initial-yaw and post-turn telemetry.
Calibration runs verify the stored commands and additional signed differences; independently measured GNSS histories are needed to evaluate prediction under positioning errors.

The later batches are `20261015_heading_prediction_corrected_frame` in City Park and `20261016_heading_prediction` in Downtown West. The Downtown West full Qwen control comes from `formal_20260909_170427`. Comparisons match the same route indices, retaining ten episodes per condition per route. The endpoint conventions are documented in R5.

`supplementary/heading_route_audit.csv` lists all 44 histories, directions, stored turns, measured calibration yaws and signed differences. `supplementary/heading_episode_audit.csv` contains the 440 formal execution checks. The largest measured angle-difference errors were $0.1477^\circ$ and $0.3341^\circ$, respectively; the acceptance tolerance was $1^\circ$.

| Body yaw | Predicted yaw | Signed turn | Interpretation |
| --- | --- | --- | --- |
| $0^\circ$ | $90^\circ$ | $+90^\circ$ | right |
| $90^\circ$ | $0^\circ$ | $-90^\circ$ | left |
| $179^\circ$ | $-179^\circ$ | $+2^\circ$ | wrap across boundary |
| $-179^\circ$ | $179^\circ$ | $-2^\circ$ | wrap across boundary |
| $35^\circ$ | $90^\circ$ | $+55^\circ$ | reference remains body yaw |

The replay checks these cases, cardinal directions, and disagreement between motion direction and body yaw.
The runner and replay reject non-finite histories and weighted or latest XY vectors with norms below $10^{-5}$ m.
All formal histories passed these vector checks.

### R2. Full navigation prompt and nine-region mapping
The navigation request supplies forward RGB, forward depth visualization, and top-down RGB, followed by a text item in the same user message.
The request uses this single user message without a separate system message.
The English translation below accompanies the exact Chinese UTF-8 templates in `supplementary/interfaces/`.

You are a UAV pilot, currently flying using the cameras mounted on the UAV. Current task: navigate intelligently to {TARGET} and land. Precise positioning is unnecessary; once approximately near the target, you may output stop. I will provide three images from the UAV cameras in sequence: a forward image, a forward depth image, and a top-down image. Use the depth image to avoid nearby objects as much as possible, navigate flexibly to the vicinity of the target, and land. Decide the next action from these images. The available actions are listed below. Your output must be one of the following formats (the values must be numbers):
```text
forward X      (fly forward X metres)
backward X     (fly backward X metres)
left X         (fly left X metres)
right X        (fly right X metres)
up X           (ascend X metres)
down X         (descend X metres)
turn_left Y    (turn left Y degrees)
turn_right Y   (turn right Y degrees)
stop           (target reached; stop navigation)
```
Output only one command, without any explanation.

The actual CLIP cue and random cue conditions append the following sentence after the action instructions:
"To assist your judgment: your navigation target is very likely in your {REGION}."
The template's probability wording is an uncalibrated heuristic; the no cue condition omits the appended sentence.

The mapping uses row-major image coordinates, from top to bottom and left to right. Its exact Chinese phrases are in `supplementary/interfaces/nine_region_mapping.csv`; their image-relative meanings are:

| Row | Left | Center | Right |
| --- | --- | --- | --- |
| 1 | 1: upper left | 2: upper center | 3: upper right |
| 2 | 4: middle left | 5: middle center | 6: middle right |
| 3 | 7: lower left | 8: lower center | 9: lower right |

The labels describe locations within the forward image, using the original Chinese phrases for the upper, middle, and lower regions.
Each route has an English CLIP matching phrase and a Chinese MLLM navigation description.
The exact UTF-8 strings are listed in `supplementary/interfaces/route_target_texts.csv`, and repeated in `supplementary/routes.csv`.
The older City Park target names serve as route identifiers.
For route 1, CLIP receives "orange and white striped umbrella"; the separate Chinese task description refers to the vicinity of that umbrella.
The English phrase is supplied directly to CLIP encoding, without a separately tuned text template.
No automatic translation or underscore replacement is performed.
The two `complete_text_example.json` files contain text from later heading requests, including target substitution and the appended region phrase.
Each example identifies its condition, route, repeat, and step.
For actual CLIP cues, the region with the largest image-text similarity supplies the appended phrase.

### R3. CLIP and camera interface
The spatial module uses OpenAI CLIP ViT-B/32 with the forward image in RGB format.
Lanczos resampling reduces its dimensions to the nearest lower multiples of three; a $640\times480$ image becomes $639\times480$.
The nine equal tiles undergo the checkpoint's standard resize, center crop to $224\times224$, and RGB tensor conversion.
Normalization uses means $(0.48145466,0.4578275,0.40821073)$ and standard deviations $(0.26862954,0.26130258,0.27577711)$.
The highest-scoring normalized image-text feature match selects the tile, without navigation-specific fine-tuning or a target-presence gate.
The local checkpoint checksum is unavailable in the archive.

City Park uses camera identifier `0` for Scene and DepthVis and identifier `3` for top-down Scene; all responses are $640\times480$.
Downtown West requests `front_center` Scene and DepthVis at $640\times480$ and `bottom_center` Scene at $2560\times1440$.
Depth input is AirSim's eight-bit DepthVis display, supplied directly without a documented metric near/far conversion.
Images use the client's default JPEG encoding, with no explicit quality parameter.

Camera JSON files specify global defaults and named-camera overrides for capture type, dimensions, field of view, and mounting.
The global field of view is $110^\circ$, with $90^\circ$ overrides for named front/downward Scene entries.
The settings file does not fully map City Park's numeric camera identifiers to named-camera extrinsics.
Observation examples record response dimensions, camera positions and quaternions, vehicle poses, and simulator timestamps, including timestamp differences among views.

### R4. Action parser, timing and inference
The response is stripped of surrounding whitespace and lowercased. The parser accepts exact `stop`, or the full-string pattern below:
```text
^(forward|backward|left|right|up|down|turn_left|turn_right)\s+(\d+(?:\.\d+)?)$
```
Magnitudes are nonnegative; signs, scientific notation, prose, units and multiple commands do not match this grammar. The parser has no explicit upper-magnitude clamp. Invalid responses do not execute a movement. The three Qwen3.5-35B-A3B cue conditions used identical model configuration and inference parameters, including a 256-token output limit and disabled thinking. Temperature was not explicitly specified; all three conditions used the same model defaults. No fixed seed was specified for individual model responses. The experiment runner records request failures and permits at most three attempts, with a 150 s timeout per request.

Horizontal translation speed is 2.0 m/s, vertical speed is 1.5 m/s, and yaw rate is $30^\circ$/s.
Navigation action duration is magnitude divided by the relevant speed or yaw rate, with a 0.5 s minimum.
Heading initialization precedes the first MLLM action and uses angle divided by yaw rate, without the navigation minimum.

All three City Park cue conditions used 20 decisions, ClockSpeed 5, and a 4 s wall-clock post-action wait, with simulation continuing during inference.
All three Downtown West cue conditions used 30 decisions, ClockSpeed 5, and a 0.8 s wall-clock post-action wait.
The Downtown West wait corresponds to 4 s of simulation time while unpaused.
Simulation pauses during Downtown West inference, so inference latency does not advance simulated physics.

Large-parameter models were run on two NVIDIA A100 GPUs with 80 GB of memory each, while the remaining experiments used an NVIDIA GeForce RTX 5070 Ti workstation.
Some original external methods relied on legacy GPT API versions unavailable during experimentation; their adapted implementations are specified below.

#### Adapted external methods
The AerialVLA implementation deployed `openvla/openvla-7b` locally with the
`XuPeng23/AerialVLA` low-rank adaptation (LoRA) module.
It resized forward and downward RGB images to $224\times224$ pixels each and stacked them vertically into a $224\times448$ mosaic.
The prompt combined the target description with a coarse body-frame direction computed from simulator target coordinates.
This direction cue supplied privileged information beyond the visual inputs available to the other methods.
AerialVLA returned three continuous controls: forward velocity, vertical velocity, and yaw rate.
Their ranges were $[0,5]$~m/s, $[-5,5]$~m/s, and $[-1.1,1.1]$~rad/s, respectively.
Decoding was deterministic (`do_sample=False`, `max_new_tokens=20`), and each episode allowed at most 20 navigation steps.

The City Park adaptation of CityNavAgent supplied forward RGB, forward depth visualization, and top-down RGB to the Qwen client.
This three-image interface replaced the original panorama and omitted the original DINO/SAM visual grounding module.
The prompt included the navigation instruction, target description, recent action history, and eight permitted commands:
`STOP`, `MOVE_FORWARD`, `TURN_LEFT`, `TURN_RIGHT`,
`GO_UP`, `GO_DOWN`, `MOVE_LEFT`, and `MOVE_RIGHT`.
Step sizes were 5~m for horizontal motion, 2~m for vertical motion, and 15$^{\circ}$ for turns.
The adapted pipeline used Qwen3.5-35B-A3B.
It set `max_tokens=200` for viewpoint decisions and allowed at most 20 navigation steps.
Unparseable responses triggered the fallback action `MOVE_FORWARD` in this implementation.
Separate tuning-budget records are unavailable.

The FineCog-Nav adapter used Qwen3.5-27B.
This model served both the text-reasoning and visual-perception modules, with forward RGB supplied to visual perception.
A rule-based collision checker converted the forward depth visualization into warnings for the decision prompt.
The adapter saved top-down RGB images but did not supply them to perception or decision calls.
The pipeline invoked instruction parsing, subgoal extraction, attention guidance, visual perception, imagination, subgoal judging, decision making, and step-memory modules.
It used the same eight discrete actions as CityNavAgent and allowed at most 20 steps.
Step sizes were 5~m horizontally, 2~m vertically, and 15$^{\circ}$ for yaw rotations.
Token limits were `max_tokens=1000` for text modules and `max_tokens=800` for visual calls.
Separate tuning-budget records are unavailable for this implementation.

In Downtown West, CityNavAgent used InternVL2-26B with forward RGB input.

### R5. Endpoint and trajectory scoring
The repository contains 3,780 unique scoring trajectories: the original 3,340 benchmark and heading episodes, plus 140 City Park and 300 Downtown West random-cue episodes.
The three-condition subset in `supplementary/cue_prior_control/` contains 1,320 records, including 880 reused actual CLIP cue and no cue trajectories.
Every configuration contains ten episodes per route; SHA-256 hashes link the reused trajectories to their original files.

`supplementary/routes.csv` supplies all 44 route targets, starts, goals, radii, and horizontal straight-line reference lengths.
Goal coordinates were selected before experimentation, and scoring uses the frozen configurations for both environments.
City Park's evaluated goal file differs from a later JSON on two routes; `supplementary/routes.csv` identifies the evaluated coordinates.
Downtown West uses 3 m radii on seven routes and 6 m on the other 23, as detailed in R7.

For cue ablations, trajectory recording begins after takeoff and heading initialization, before the first MLLM decision. The first recorded position can differ from the configured start because initialization precedes recording. Path length accumulates horizontal displacements from this first sample onward; takeoff, initialization, and preceding movement are outside the recorded path. The straight-line reference length uses the configured start.

For recorded horizontal samples $\mathbf p_0,\ldots,\mathbf p_T$, goal $\mathbf g$, radius $r$, and configured start $\mathbf s$, the episode quantities are
$$
\begin{aligned}
\mathrm{NE}&=\|\mathbf p_T-\mathbf g\|,& S&=\mathbf1[\mathrm{NE}\leq r],\\
P&=\sum_{t=1}^{T}\|\mathbf p_t-\mathbf p_{t-1}\|,& L&=\|\mathbf s-\mathbf g\|,\\
\mathrm{SPL}&=S\,\frac{L}{\max(P,L)},&\mathrm{OR}&=\mathbf1[\min_t\|\mathbf p_t-\mathbf g\|\leq r],\\
\mathrm{TL}&=P.&
\end{aligned}
$$
SR is the mean of $S$. For the degenerate case $P=L=0$, the replay returns SPL 0, consistent with the supplied evaluator. All distances use XY coordinates. SR and NE depend on the final navigation coordinate; SPL also uses path length, OR uses the nearest recorded trajectory point, and TL uses accumulated horizontal displacement. A trajectory may enter the goal region and leave it, yielding OR 1 and SR 0.

All conditions use the same final-position and horizontal-trajectory metric definitions.
For XYZ-only records, the final CSV coordinate supplies the final navigation position.
Where `navigation_end_position` is saved, replay ends at the last matching XY sample within $10^{-6}$ m, excluding subsequent landing samples.
Collision and termination labels do not affect inclusion, success, weighting, or metric values.
The replay verifies this invariance with coordinates and final navigation position held fixed.

### R6. Statistical replay and source files
Episode scores are averaged within each route and then across routes.
Offline replay reconstructs scores and route-level statistics from archived trajectories.
Whole-route bootstrap intervals use 10,000 resamples, retaining all ten repeats and matching sampled route indices across conditions.
Matching occurs at the route level, rather than between individual episode repeats.
The original benchmark replay uses seed 20260823 for City Park cue comparisons, 20261014 for Downtown West cue comparisons, and 20260823 for both heading comparisons.
The unified three-condition analysis in R8 supplies the Qwen3.5-35B-A3B cue intervals used in the paper tables and figures.
These seeds control offline resampling; model-response settings are specified in R4.

Both scoring versions use the same replay program and bootstrap seeds.
The seven radius changes affect SR, SPL, and OR; NE, TL, and all City Park results remain unchanged.
R7 reports the original-threshold sensitivity analysis, while paper tables report route-matched differences and their intervals.
Figure whiskers show marginal intervals for each condition.

Run the following command with Python 3.10 or later and NumPy, from the repository root:
```text
python supplementary/reproduce_results.py
```
The program reads `supplementary/episodes_manifest.csv` and produces `results.json`, episode and route metrics, and `plot_ci.csv`. Its built-in checks cover signed heading, wrap boundaries, zero-vector rejection, SR/OR differences, and invariance to event labels. No model service, simulator, network connection or new experiment is needed.

The MATLAB script `scripts/render_paper_results.m` reproduces the archived bar-chart layouts from the original editable FIG files and replayed confidence limits.
The backbone and Heading Prediction point-interval plots use `scripts/render_adopted_result_figures.m`; `scripts/verify_adopted_result_figures.m` checks their means and confidence limits against the frozen results.
Both scripts use the archived FIG data as their inputs.
In the Heading Prediction ablation plot, open markers identify the last-segment condition (Ablated), and filled markers identify weighted-history initialization (Unablated).
Gray dashed lines connect condition means; capped colored bars show marginal 95% intervals in panels a, c-f and route-paired 95% SR-difference intervals in panel b.
The Figure 3 schematic is supplied as an editable native PPTX. The schematic and both point-interval plots use Cambria Math; the benchmark target and navigation-sequence plates retain Cambria.
The Downtown West plate uses eight archived route reference views, with their route identifiers and native crops documented in the figure-source manifest.
The navigation-sequence plate retains twenty full-frame front-camera observations from four successful recorded episodes and is supplied as an editable PPTX.
The source records in `source_data/navigation_maneuvers/` identify the episodes, capture times, target descriptions, trajectories, and executed lateral or turning actions.
The repository supplies editable PPTX and MATLAB FIG/M sources, LaTeX, scoring trajectories, configurations, and checksums.
Figure 1 is schematic; its source illustration has no retained route/repeat identifiers.

### R7. Revised success radii and original-threshold sensitivity
The initial Downtown West evaluation used a uniform 6 m radius.
The seven affected routes were 09, 14, 16, 17, 21, 25, and 29.
Their configured start-goal distances were 4.431, 5.757, 4.000, 4.265, 5.525, 5.999, and 5.996 m, respectively.
Their starts therefore already satisfied the original criterion.

During manuscript revision, we halved the radius to 3 m on those seven routes and retained 6 m on the other 23.
This post-experiment rule used configured XY geometry and was adopted after outcomes were available.
All methods and heading conditions use the same rule, placing every configured start outside its success region.
Immediate stopping at the configured start therefore scores SR and SPL zero.

Goal coordinates, recorded trajectories, endpoint conventions and sample inclusion are unchanged. All 3,340 original benchmark and heading episodes remain included. The resulting scoring rule was frozen before the 440 random-cue episodes were collected. The revised radii apply to 420 manifest rows across the six Downtown West conditions. The manifest records both the original and revised radii, and `supplementary/threshold_revision.json` documents the rule and timing.

The following table reports Qwen3.5-35B-A3B with and without the cue in Downtown West.
This threshold-sensitivity replay uses seed 20261014 for both scoring versions; R8 separately reports the unified three-condition bootstrap.

| Scoring | Condition | SR (%) | NE (m) | SPL (%) | OR (%) | TL (m) |
| --- | --- | --- | --- | --- | --- | --- |
| Original 6 m | Actual CLIP cue | 70.3 | 7.27 | 61.5 | 78.3 | 13.15 |
| Original 6 m | No cue | 49.3 | 8.36 | 44.1 | 62.7 | 11.56 |
| Revised 3/6 m | Actual CLIP cue | 54.7 | 7.27 | 48.9 | 63.3 | 13.15 |
| Revised 3/6 m | No cue | 35.3 | 8.36 | 31.9 | 46.7 | 11.56 |

The revised cue-minus-no-cue SR difference is 19.3 percentage points (95% route-clustered interval 13.0 to 26.0).
The original-radius difference is 21.0 points (14.3 to 28.0).
Corresponding SPL differences are 16.9 points (11.1 to 23.2) and 17.4 points (11.0 to 23.9).
Both evaluations replay the same recorded episodes under their documented scoring rules.

The complete original results for every method and heading condition are retained in `supplementary/results_original_6m/`. Reproduce that version with
```text
python -B supplementary/reproduce_results.py --manifest supplementary/episodes_manifest_original_6m.csv --output supplementary/results_original_6m
```
The default command in R6 reproduces the revised evaluation.

### R8. Image-independent uniform random-cue control
The three-condition cue ablation used Qwen3.5-35B-A3B on both maps, with ten episodes per route: 140 in City Park and 300 in Downtown West.
Within each map, model settings, routes, observations, preprocessing, weighted heading initialization, action execution, timing, and scoring were fixed across cue conditions.
The analysis reused actual CLIP cue and no cue records and added 440 random cue episodes.
Six two-decision functional checks preceded the formal sample, which includes all planned route/repeat cells.

The random cue condition appends the same sentence as the actual CLIP cue condition (R2).
Before navigation, each decision's region was drawn independently and uniformly from the nine labels, independently of the image.
CLIP regions were also computed and logged, allowing incidental agreement with random cues.
Regional frequencies and temporal patterns could therefore differ between the two cue conditions.

NumPy SeedSequence used $[20261027,1,m,r,k,s]$ for the cue stream, where $m$ is map ID, $r$ route, $k$ repeat, and $s$ indicates a functional check. Route order used $[20261027,2,m]$ and was randomized within each repetition round. Cue sequences were generated before execution and reused on recovery. The archived `cue_schedule.json` and `transmitted_cues.csv` preserve scheduled and sent regions; the latter also records the contemporaneous CLIP region.

The additional runs followed the map-specific timing and decision budgets in R4.
Coordinates, radii, reference lengths, and weighted initialization turns came from the frozen submission configuration.
Recovery retained completed episodes and their originally scheduled cues.
Transport errors allowed at most three requests with the same observation and cue; persistent errors paused execution.
Invalid command text consumed a decision without movement, and scoring followed R5.

The additional random cue episodes were collected on 27-28 October 2026.
Machine-readable configurations and scoring manifests identify the source records for all conditions.

**Analysis.** The complete comparison contains 1,320 episodes and 44 route units. We averaged ten repeats within each route and condition, then resampled whole routes 10,000 times with identical sampled indices across all three conditions. Episode repeat numbers were not treated as matched pairs. Bootstrap streams used $[20261027,3,m]$. The two primary SR contrasts, actual minus random cue, use 97.5% percentile intervals (1.25th and 98.75th percentiles), with a Bonferroni allocation across maps. Other contrasts and metrics use exploratory 95% intervals. Marginal figure whiskers use 95% intervals. Differences are computed before rounding; percentage-metric differences are percentage points.

**Three-condition means.** Each City Park row includes 140 episodes and each Downtown West row includes 300.

| Environment | Condition | SR (%) | NE (m) | SPL (%) | OR (%) | TL (m) |
| --- | --- | --- | --- | --- | --- | --- |
| City Park | Actual CLIP cue | 80.7 | 9.81 | 68.7 | 87.1 | 41.75 |
| City Park | Random cue | 21.4 | 25.87 | 18.1 | 43.6 | 39.39 |
| City Park | No cue | 42.9 | 15.94 | 37.3 | 60.0 | 44.01 |
| Downtown West | Actual CLIP cue | 54.7 | 7.27 | 48.9 | 63.3 | 13.15 |
| Downtown West | Random cue | 30.3 | 9.36 | 26.6 | 45.7 | 14.67 |
| Downtown West | No cue | 35.3 | 8.36 | 31.9 | 46.7 | 11.56 |

**All pairwise contrasts.** Entries are mean difference [lower, upper]. Only SR in the actual-minus-random column uses 97.5% intervals; all remaining entries use exploratory 95% intervals. Negative NE differences favor the first condition. TL differences describe traveled distance and require joint interpretation with success.

**City Park.**

| Metric | Actual $-$ random | Actual $-$ no cue | Random $-$ no cue |
| --- | --- | --- | --- |
| SR (%) | $+59.3$ [$+45.0$, $+73.6$] | $+37.9$ [$+20.7$, $+55.0$] | $-21.4$ [$-37.9$, $-4.3$] |
| NE (m) | $-16.06$ [$-22.81$, $-9.55$] | $-6.13$ [$-11.46$, $-1.04$] | $+9.93$ [$+3.99$, $+15.86$] |
| SPL (%) | $+50.5$ [$+40.7$, $+61.2$] | $+31.3$ [$+15.3$, $+47.4$] | $-19.2$ [$-33.8$, $-4.1$] |
| OR (%) | $+43.6$ [$+30.7$, $+56.4$] | $+27.1$ [$+11.4$, $+44.3$] | $-16.4$ [$-31.4$, $+0.0$] |
| TL (m) | $+2.36$ [$-8.67$, $+12.46$] | $-2.26$ [$-10.28$, $+4.33$] | $-4.62$ [$-14.94$, $+6.45$] |

**Downtown West.**

| Metric | Actual $-$ random | Actual $-$ no cue | Random $-$ no cue |
| --- | --- | --- | --- |
| SR (%) | $+24.3$ [$+15.0$, $+33.7$] | $+19.3$ [$+13.0$, $+26.0$] | $-5.0$ [$-15.0$, $+4.7$] |
| NE (m) | $-2.09$ [$-3.03$, $-1.17$] | $-1.09$ [$-2.12$, $-0.14$] | $+1.00$ [$-0.32$, $+2.21$] |
| SPL (%) | $+22.3$ [$+15.0$, $+30.0$] | $+16.9$ [$+11.2$, $+23.3$] | $-5.3$ [$-14.8$, $+3.9$] |
| OR (%) | $+17.7$ [$+11.0$, $+25.0$] | $+16.7$ [$+10.0$, $+23.7$] | $-1.0$ [$-9.0$, $+7.0$] |
| TL (m) | $-1.52$ [$-3.67$, $+0.47$] | $+1.59$ [$+0.20$, $+3.04$] | $+3.11$ [$+1.11$, $+5.03$] |

Actual CLIP cues had higher SR than random cues and no cue in both maps.
Random cues reduced City Park SR relative to no cue, whereas the Downtown West interval included zero.
SPL showed the same pairwise contrast pattern across the two maps.

**Offline replay.** Run the following command from the repository root with Python 3.10+ and NumPy:
```text
python supplementary/reproduce_cue_control.py
```
The program verifies trajectory hashes, complete route/repeat cells, and invariance of the five scores to event labels.
It reconstructs three-condition means and intervals from coordinates and trajectories, writing outputs to `supplementary/cue_prior_control/replay/`.
Full-precision outputs include 1,320 episode scores, 132 route means, and 132 route-level contrasts.
The MATLAB `cueprior` selection reads `supplementary/cue_prior_control/plot_ci.csv` using the result-figure layout and Cambria Math fonts.

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
