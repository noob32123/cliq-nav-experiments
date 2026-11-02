Raw source images and telemetry for Figure 13.

Twenty original 640x480 front-camera PNGs and four trajectories are byte-preserved. The figure contract records their hashes and timestamps. Executed lateral/turn actions retain raw action and action-end logs. These are qualitative selected episodes; complete scoring datasets remain elsewhere in the archive.

All frozen file fields in figure_contract.json are relative to this folder. Original_contract.json preserves the approved candidate metadata. The figure uses chronological samples, not necessarily adjacent decision steps. No image pixels, source trajectories, scoring settings or quantitative results were changed.

Rebuild the native image plate from this directory with scripts/build_navigation_sequence_figure.mjs, then run export_navigation_sequence_figure.ps1 -Mode normalize, finalize_navigation_sequence_figure.mjs, and export_navigation_sequence_figure.ps1 -Mode export. The native PPTX is also directly editable.
